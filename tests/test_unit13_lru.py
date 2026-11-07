from collections import Counter, deque
from types import SimpleNamespace

import pytest

from experiments.unit13_lru_compare import run_case
from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cache_policy import LRUCachePolicy
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def test_lru_comparison_exposes_different_victim_and_regression():
    original, lru = run_case('original'), run_case('lru')
    assert original['C_recycled_block'] == 3
    assert lru['C_recycled_block'] == 0
    assert original['D_cached_tokens'] == 256
    assert original['D_computed_prompt_tokens'] == 1
    assert lru['D_cached_tokens'] == 0
    assert lru['D_computed_prompt_tokens'] == 257
    for case in (original, lru):
        for row in case['trace']:
            references = Counter(block for table in row['tables'].values() for block in table)
            assert row['ref_counts'] == [references[i] for i in range(4)]
            assert not set(row['free']) & set(row['used'])
            assert len(set(row['free'])) == len(row['free'])
            assert set(row['free']) | set(row['used']) == set(range(4))
        assert case['trace'][-1]['used'] == []
    rows = {row['event']: row for row in lru['trace']}
    assert rows['A_finished']['last_access'] == {0: 1, 1: 2, 2: 3, 3: 4}
    assert rows['A_finished']['A_prefix_index'] == 0
    assert rows['C_allocated']['A_prefix_index'] is None
    assert rows['D_finished']['A_prefix_index'] is not None  # Recomputed prefix is valid again.


def test_never_acquired_blocks_and_ties_use_queue_order():
    policy = LRUCachePolicy()
    candidates = deque([3, 1, 2])
    assert policy.select_victim(candidates) == 3
    policy.on_acquire(3)
    assert policy.select_victim(candidates) == 1
    assert list(candidates) == [3, 1, 2]


def test_probes_and_releases_do_not_refresh_but_cached_acquisition_does():
    policy = LRUCachePolicy()
    bm = BlockManager(4, 256, eviction_policy=policy)
    a = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    bm.allocate(a, 0)
    a.num_scheduled_tokens = 257
    bm.hash_blocks(a)
    bm.deallocate(a)
    before = policy.last_access.copy()
    clock = policy.clock
    b = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    for _ in range(3):
        assert bm.can_allocate(b) == 1
    assert policy.last_access == before and policy.clock == clock
    bm.allocate(b, 1)
    assert b.block_table == [0, 2]  # Unused 2 is chosen before previously acquired 1.
    assert policy.last_access[0] > before[0]
    after = policy.last_access.copy()
    bm.deallocate(b)
    assert policy.last_access == after


def test_oldest_access_does_not_allow_recycling_a_live_block():
    policy = LRUCachePolicy()
    bm = BlockManager(3, 256, eviction_policy=policy)
    requests = [Sequence([10 + i], SamplingParams(max_tokens=1)) for i in range(4)]
    for seq in requests[:3]:
        bm.allocate(seq, 0)
    bm.deallocate(requests[1])
    bm.deallocate(requests[2])
    bm.allocate(requests[3], 0)
    assert requests[0].block_table == [0]  # Oldest but still referenced.
    assert requests[3].block_table == [1]
    assert bm.blocks[0].ref_count == 1
    bm.deallocate(requests[0])
    bm.deallocate(requests[3])
    assert not bm.used_block_ids


@pytest.mark.parametrize('phase', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('decode', ['original', 'round_robin'])
def test_lru_shared_prefix_stays_live_and_free_cache_is_reused(phase, decode):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=3,
        scheduling_policy=phase, decode_scheduling_policy=decode,
        cache_eviction_policy='lru',
    ))
    assert isinstance(scheduler.block_manager.eviction_policy, LRUCachePolicy)
    bm = scheduler.block_manager
    prompt = list(range(257))
    a, b = [Sequence(prompt, SamplingParams(max_tokens=2)) for _ in range(2)]
    for seq in (a, b):
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        assert batch == [seq] and prefill
        scheduler.postprocess(batch, [1000], prefill)
    shared = a.block_table[0]
    assert b.block_table[0] == shared and bm.blocks[shared].ref_count == 2
    assert not bm.free_block_ids
    for seq, remaining in [(a, 1), (b, 0)]:
        batch, prefill = scheduler.schedule()
        assert batch == [seq] and not prefill
        scheduler.postprocess(batch, [1001], prefill)
        assert bm.blocks[shared].ref_count == remaining
        assert (shared in bm.free_block_ids) == (remaining == 0)
    c = Sequence(prompt, SamplingParams(max_tokens=1))
    scheduler.add(c)
    batch, prefill = scheduler.schedule()
    assert c.num_cached_tokens == 256
    assert c.block_table[0] == shared
    scheduler.postprocess(batch, [1002], prefill)
    assert scheduler.is_finished() and not bm.used_block_ids
    assert all(block.ref_count == 0 for block in bm.blocks)


@pytest.mark.parametrize('invalid', ['typo', None, True])
def test_invalid_scheduler_cache_policy_is_rejected(invalid):
    with pytest.raises(ValueError, match='cache_eviction_policy'):
        Scheduler(SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=512, eos=-1,
                                  kvcache_block_size=256, num_kvcache_blocks=4,
                                  cache_eviction_policy=invalid))
