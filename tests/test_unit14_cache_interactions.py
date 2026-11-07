from collections import Counter

import pytest

from experiments.unit14_cache_interactions import make_scheduler, run_shared_preempt
from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus


def assert_pool_consistent(scheduler, requests):
    bm = scheduler.block_manager
    refs = Counter(block for seq in requests for block in seq.block_table)
    assert [block.ref_count for block in bm.blocks] == [refs[i] for i in range(len(bm.blocks))]
    free, used = set(bm.free_block_ids), bm.used_block_ids
    assert not free & used and free | used == set(range(len(bm.blocks)))
    assert len(free) == len(bm.free_block_ids)
    for block in bm.blocks:
        assert (block.block_id in used) == (block.ref_count > 0)
    for hash_value, block_id in bm.hash_to_block_id.items():
        assert bm.blocks[block_id].hash == hash_value
        assert len(bm.blocks[block_id].token_ids) == bm.block_size


@pytest.mark.parametrize('cache', ['original', 'lru'])
@pytest.mark.parametrize('phase', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('decode', ['original', 'round_robin'])
def test_preempted_request_reuses_prefix_still_referenced_by_peer(cache, phase, decode):
    case = run_shared_preempt(cache, phase=phase, decode=decode)
    rows = {row['event']: row for row in case['trace']}
    preempted = rows['A_preempted_B_holds_prefix']
    a = preempted['requests']['A']
    assert a['cached_tokens'] == 0 and a['block_table'] == []
    assert a['completion_ids'] == [1000] and a['status'] == 'WAITING'
    assert preempted['ref_counts'][case['shared_block']] == 1
    assert case['shared_block'] not in preempted['free']
    assert case['resume_cached_tokens'] == 256
    assert case['resume_computed_tokens'] == 2  # Original tail + one saved output token.
    assert case['A_final_completion_ids'] == [1000, 1001, 1002]
    for row in case['trace']:
        references = Counter(block for req in row['requests'].values() for block in req['block_table'])
        assert row['ref_counts'] == [references[i] for i in range(3)]
        assert not set(row['free']) & set(row['used'])
        assert set(row['free']) | set(row['used']) == {0, 1, 2}
        assert len(set(row['free'])) == len(row['free'])
    assert case['trace'][-1]['used'] == []
    assert all(req['status'] == 'FINISHED' for req in case['trace'][-1]['requests'].values())


@pytest.mark.parametrize('cache', ['original', 'lru'])
def test_hash_chain_requires_matching_context_not_just_current_block_tokens(cache):
    scheduler = make_scheduler(cache)
    bm = scheduler.block_manager
    prompt = list(range(513))
    first = Sequence(prompt, SamplingParams(max_tokens=1))
    scheduler.add(first)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [1000], prefill)
    first_hash = bm.compute_hash(prompt[:256])
    second_hash = bm.compute_hash(prompt[256:512], first_hash)
    second_block = bm.hash_to_block_id[second_hash]
    assert bm.blocks[second_block].token_ids == prompt[256:512]
    same = Sequence(prompt, SamplingParams(max_tokens=1))
    assert bm.can_allocate(same) == 2
    changed_first = prompt.copy()
    changed_first[0] = 9000
    probe = Sequence(changed_first, SamplingParams(max_tokens=1))
    assert probe.block(1) == same.block(1)
    assert bm.can_allocate(probe) == 0  # Equal second-block tokens do not imply equal KV context.
    changed_second = prompt.copy()
    changed_second[256] = 9001
    assert bm.can_allocate(Sequence(changed_second, SamplingParams(max_tokens=1))) == 1
    new_first_hash = bm.compute_hash(changed_first[:256])
    assert bm.compute_hash(changed_first[256:512], new_first_hash) != second_hash
    scheduler.add(probe)
    batch, prefill = scheduler.schedule()
    assert probe.num_cached_tokens == 0 and probe.num_scheduled_tokens == 513
    scheduler.postprocess(batch, [1001], prefill)
    assert probe.is_finished and scheduler.is_finished()
    assert_pool_consistent(scheduler, [first, probe])


@pytest.mark.parametrize('cache', ['original', 'lru'])
def test_recycling_old_duplicate_does_not_delete_index_of_live_newer_copy(cache):
    scheduler = make_scheduler(cache, blocks=4, max_num_seqs=2)
    bm = scheduler.block_manager
    prefix = list(range(256))
    a = Sequence(prefix + [300], SamplingParams(max_tokens=2))
    b = Sequence(prefix + [301], SamplingParams(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)
    batch, prefill = scheduler.schedule()
    assert batch == [a, b] and prefill
    old_copy, new_copy = a.block_table[0], b.block_table[0]
    assert old_copy != new_copy  # Neither prefix was published at batch admission time.
    scheduler.postprocess(batch, [1000, 1001], prefill)
    prefix_hash = bm.compute_hash(prefix)
    assert bm.hash_to_block_id[prefix_hash] == new_copy
    scheduler.max_num_seqs = 1
    batch, prefill = scheduler.schedule()
    assert batch == [a] and not prefill
    scheduler.postprocess(batch, [1002], prefill)
    assert bm.blocks[new_copy].ref_count == 1 and old_copy in bm.free_block_ids
    c = Sequence(list(range(5000, 5257)), SamplingParams(max_tokens=1))
    scheduler.add(c)
    batch, prefill = scheduler.schedule()
    assert old_copy in c.block_table
    assert bm.hash_to_block_id[prefix_hash] == new_copy
    scheduler.postprocess(batch, [1003], prefill)
    d = Sequence(prefix + [302], SamplingParams(max_tokens=1))
    scheduler.add(d)
    batch, prefill = scheduler.schedule()
    assert d.num_cached_tokens == 256 and d.block_table[0] == new_copy
    assert bm.blocks[new_copy].ref_count == 2
    scheduler.postprocess(batch, [1004], prefill)
    assert_pool_consistent(scheduler, [a, b, c, d])
    batch, prefill = scheduler.schedule()
    assert batch == [b] and not prefill
    scheduler.postprocess(batch, [1005], prefill)
    assert scheduler.is_finished()
    assert_pool_consistent(scheduler, [a, b, c, d])


@pytest.mark.parametrize('cache', ['original', 'lru'])
@pytest.mark.parametrize('decode', ['original', 'round_robin'])
def test_automatic_preemption_overwrites_victim_block_and_resumes_without_losing_outputs(cache, decode):
    scheduler = make_scheduler(cache, blocks=2, max_num_seqs=2,
                               phase='bounded_prefill', decode=decode)
    bm = scheduler.block_manager
    a = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    b = Sequence([500], SamplingParams(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [1000, 2000], prefill)
    b_block = b.block_table[0]
    saved = b.token_ids.copy()
    batch, prefill = scheduler.schedule()
    assert batch == [a] and not prefill
    assert b.status == SequenceStatus.WAITING and b.block_table == []
    assert b.token_ids == saved and b.num_cached_tokens == 0
    assert a.block_table[-1] == b_block
    assert_pool_consistent(scheduler, [a, b])
    scheduler.postprocess(batch, [1001], prefill)
    batch, prefill = scheduler.schedule()
    assert batch == [b] and prefill
    assert b.num_cached_tokens == 0 and b.num_scheduled_tokens == 2
    scheduler.postprocess(batch, [2001], prefill)
    assert b.completion_token_ids == [2000, 2001]
    assert scheduler.is_finished() and not bm.used_block_ids
    assert_pool_consistent(scheduler, [a, b])
