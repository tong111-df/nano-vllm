from collections import Counter
from types import SimpleNamespace

from experiments.unit11_cache_lifecycle import run_lifecycle
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def test_shared_prefix_stays_used_until_last_reference_finishes():
    case = run_lifecycle()
    events = {event['event']: event for event in case['events']}
    shared = case['prefix_block_id']
    hit = events['B_prefix_hit']
    assert hit['requests']['A']['block_table'][0] == shared
    assert hit['requests']['B']['block_table'][0] == shared
    assert hit['requests']['A']['block_table'][1] != hit['requests']['B']['block_table'][1]
    assert hit['requests']['B']['cached_tokens'] == 256
    assert hit['blocks'][shared]['ref_count'] == 2
    assert hit['free_blocks'] == []
    a_done = events['A_finished']
    assert a_done['blocks'][shared]['ref_count'] == 1
    assert shared in a_done['used_blocks'] and shared not in a_done['free_blocks']
    assert a_done['requests']['A']['block_table'] == []
    assert a_done['requests']['B']['block_table'][0] == shared
    both_done = events['B_finished']
    assert both_done['blocks'][shared]['ref_count'] == 0
    assert shared in both_done['free_blocks']
    assert both_done['prefix_index_block'] == shared
    assert both_done['blocks'][shared]['metadata_token_count'] == 256


def test_free_prefix_is_reused_then_invalidated_when_overwritten():
    case = run_lifecycle()
    events = {event['event']: event for event in case['events']}
    shared = case['prefix_block_id']
    reused = events['C_free_prefix_hit']
    assert reused['requests']['C']['cached_tokens'] == 256
    assert reused['requests']['C']['block_table'][0] == shared
    assert shared not in reused['free_blocks']
    overwritten = events['D_overwrites_prefix']
    assert shared in overwritten['requests']['D']['block_table']
    assert overwritten['prefix_index_block'] is None
    assert overwritten['blocks'][shared]['hash'] == -1
    assert overwritten['blocks'][shared]['metadata_token_count'] == 0
    assert events['E_old_prefix_miss']['requests']['E']['cached_tokens'] == 0


def test_block_reference_and_pool_invariants_at_every_lifecycle_event():
    case = run_lifecycle()
    for event in case['events']:
        references = Counter(block for req in event['requests'].values() for block in req['block_table'])
        free, used = set(event['free_blocks']), set(event['used_blocks'])
        assert len(free) == len(event['free_blocks'])  # No double release.
        assert not free & used
        assert free | used == set(range(3))
        for block in event['blocks']:
            assert block['ref_count'] == references[block['id']]
            assert (block['id'] in used) == (block['ref_count'] > 0)
        for req in event['requests'].values():
            if req['status'] == 'FINISHED':
                assert req['block_table'] == []
    final = case['events'][-1]
    assert not final['used_blocks']
    assert all(req['status'] == 'FINISHED' for req in final['requests'].values())


def test_full_block_becomes_shareable_before_whole_prompt_finishes():
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=128, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=4,
    ))
    a = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    probe = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    scheduler.add(a)
    bm = scheduler.block_manager
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert a.num_cached_tokens == 128 and not a.completion_token_ids
    assert bm.can_allocate(probe) == 0
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert a.num_cached_tokens == 256 and not a.completion_token_ids
    assert bm.can_allocate(probe) == 1
    assert probe.block_table == [] and probe.num_cached_tokens == 0  # Probe does not allocate.
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert a.is_finished and scheduler.is_finished()
    assert not bm.used_block_ids
