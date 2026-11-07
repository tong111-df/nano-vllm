import pytest

from experiments.unit12_reclaim_order import RecordingOriginalPolicy, run_baseline
from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cache_policy import OriginalFreeQueuePolicy
from nanovllm.engine.sequence import Sequence


def test_original_release_order_determines_next_recycled_block():
    case = run_baseline()
    rows = {row['event']: row for row in case['trace']}
    assert rows['B_finished']['free'] == [3, 2]
    assert rows['A_finished']['free'] == [3, 2, 1, 0]
    assert rows['C_allocated']['tables']['C'] == [3]
    assert rows['C_allocated']['free'] == [2, 1, 0]
    assert [e['block'] for e in case['policy_events'] if e['event'] == 'select'] == [0, 1, 2, 3, 3, 2]


def test_cache_probe_is_read_only_but_successful_cache_acquire_updates_events():
    case = run_baseline()
    rows = {row['event']: row for row in case['trace']}
    before, probe, acquired = [rows[name] for name in ('C_finished', 'D_probe_only', 'D_prefix_acquired')]
    assert probe['free'] == before['free']
    assert probe['policy_event_count'] == before['policy_event_count']
    assert acquired['tables']['D'] == [0, 2]
    assert acquired['free'] == [1, 3]
    changes = case['policy_events'][probe['policy_event_count']:acquired['policy_event_count']]
    assert changes == [{'event': 'acquire', 'block': 0},
                       {'event': 'select', 'block': 2, 'candidates': [2, 1, 3]},
                       {'event': 'acquire', 'block': 2}]


def test_release_event_waits_for_last_reference():
    policy = RecordingOriginalPolicy()
    bm = BlockManager(3, 256, eviction_policy=policy)
    a = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    b = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    bm.allocate(a, bm.can_allocate(a))
    # Model-completion metadata only: publish the first complete prefix block.
    a.num_scheduled_tokens = 257
    bm.hash_blocks(a)
    a.num_cached_tokens, a.num_scheduled_tokens = 257, 0
    shared, tail_a = a.block_table
    assert bm.can_allocate(b) == 1
    bm.allocate(b, 1)
    tail_b = b.block_table[1]
    policy.events.clear()
    bm.deallocate(a)
    assert policy.events == [{'event': 'release', 'block': tail_a}]
    assert bm.blocks[shared].ref_count == 1
    bm.deallocate(b)
    assert policy.events[-2:] == [{'event': 'release', 'block': tail_b},
                                 {'event': 'release', 'block': shared}]
    assert all(block.ref_count == 0 for block in bm.blocks)


class TailPolicy(OriginalFreeQueuePolicy):
    # A test policy proving the interface can select a non-head candidate.
    # This is not the LRU implementation.
    def select_victim(self, free_block_ids):
        return free_block_ids[-1]


def test_non_head_selection_still_invalidates_old_hash_safely():
    bm = BlockManager(2, 256, eviction_policy=TailPolicy())
    a = Sequence(list(range(256)), SamplingParams(max_tokens=1))
    bm.allocate(a, 0)
    assert a.block_table == [1]
    a.num_scheduled_tokens = 256
    bm.hash_blocks(a)
    old_hash = bm.blocks[1].hash
    bm.deallocate(a)
    assert list(bm.free_block_ids) == [0, 1]
    b = Sequence([1000], SamplingParams(max_tokens=1))
    bm.allocate(b, 0)
    assert b.block_table == [1]
    assert old_hash not in bm.hash_to_block_id
    assert bm.blocks[1].hash == -1 and bm.blocks[1].token_ids == []
    assert list(bm.free_block_ids) == [0]
    bm.deallocate(b)
    assert len(set(bm.free_block_ids)) == 2


def test_policy_cannot_select_a_still_referenced_block():
    class InvalidPolicy(OriginalFreeQueuePolicy):
        def select_victim(self, free_block_ids):
            return 0

    bm = BlockManager(2, 256)
    a = Sequence([10], SamplingParams(max_tokens=1))
    bm.allocate(a, 0)
    bm.eviction_policy = InvalidPolicy()
    with pytest.raises(RuntimeError, match='not free'):
        bm._allocate_block()
    assert list(bm.free_block_ids) == [1]
    assert bm.used_block_ids == {0}
    assert bm.blocks[0].ref_count == 1
    assert a.block_table == [0]
    bm.deallocate(a)


def test_empty_pool_does_not_ask_policy_to_select_a_victim():
    policy = RecordingOriginalPolicy()
    bm = BlockManager(1, 256, eviction_policy=policy)
    a = Sequence([10], SamplingParams(max_tokens=1))
    bm.allocate(a, 0)
    before = policy.events.copy()
    with pytest.raises(RuntimeError, match='no free KV block'):
        bm._allocate_block()
    assert policy.events == before
    assert bm.blocks[0].ref_count == 1
    bm.deallocate(a)
