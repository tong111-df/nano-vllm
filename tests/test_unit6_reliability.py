from types import SimpleNamespace

import pytest

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler, SchedulerNoProgressError
from nanovllm.engine.sequence import Sequence


@pytest.mark.parametrize('returned_tokens', [[1000], [1000, 1001, 1002]])
def test_wrong_output_count_does_not_partially_update_requests(returned_tokens):
    scheduler = make_scheduler(2)
    requests = [Sequence([10 + i], SamplingParams(max_tokens=1)) for i in range(2)]
    for seq in requests:
        scheduler.add(seq)
    seqs, prefill = scheduler.schedule()
    assert seqs == requests
    tables_before = [s.block_table.copy() for s in requests]
    with pytest.raises(ValueError, match='expected 2 output tokens'):
        scheduler.postprocess(seqs, returned_tokens, prefill)
    # Error must not finish the first request before noticing the mismatch.
    assert list(scheduler.running) == requests
    assert [s.completion_token_ids for s in requests] == [[], []]
    assert [s.num_cached_tokens for s in requests] == [0, 0]
    assert [s.num_scheduled_tokens for s in requests] == [1, 1]
    assert [s.block_table for s in requests] == tables_before
    assert len(scheduler.block_manager.used_block_ids) == 2

    # Correct results can still be applied to the same scheduled batch.
    scheduler.postprocess(seqs, [1000, 1001], prefill)
    assert scheduler.is_finished()
    assert all(s.is_finished for s in requests)
    assert not scheduler.block_manager.used_block_ids
    assert all(b.ref_count == 0 for b in scheduler.block_manager.blocks)


@pytest.mark.parametrize('max_tokens', [0, -1])
def test_nonpositive_output_limit_is_rejected(max_tokens):
    with pytest.raises(ValueError, match='max_tokens must be a positive integer'):
        SamplingParams(max_tokens=max_tokens, ignore_eos=True)


def test_empty_prompt_has_clear_error():
    with pytest.raises(ValueError, match='prompt must contain at least one token'):
        Sequence([], SamplingParams(max_tokens=1))


@pytest.mark.parametrize('field', [
    'max_num_seqs', 'max_num_batched_tokens', 'num_kvcache_blocks',
])
def test_configuration_that_cannot_make_progress_is_rejected(field):
    config = dict(max_num_seqs=2, max_num_batched_tokens=512, eos=-1,
                  kvcache_block_size=256, num_kvcache_blocks=2)
    config[field] = 0
    with pytest.raises(ValueError, match=field):
        Scheduler(SimpleNamespace(**config))


def make_scheduler(blocks):
    return Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=512, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks,
    ))


def test_scheduling_empty_engine_has_clear_error():
    scheduler = make_scheduler(2)
    with pytest.raises(SchedulerNoProgressError, match='no pending requests'):
        scheduler.schedule()
    assert scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids


@pytest.mark.parametrize('phase_policy', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('decode_policy', ['original', 'round_robin'])
def test_temporary_block_shortage_waits_then_recovers(phase_policy, decode_policy):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=512, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=2,
        scheduling_policy=phase_policy, max_consecutive_prefills=2,
        decode_scheduling_policy=decode_policy,
    ))
    a = Sequence(list(range(257)), SamplingParams(max_tokens=1))
    b = Sequence(list(range(500, 757)), SamplingParams(max_tokens=1))
    scheduler.add(a)
    scheduler.add(b)  # Each fits alone; admission does not reserve both at once.
    seqs, prefill = scheduler.schedule()
    assert seqs == [a] and prefill is True
    assert not scheduler.block_manager.free_block_ids
    assert scheduler.block_manager.can_allocate(b) == -1
    assert list(scheduler.waiting) == [b]
    scheduler.postprocess(seqs, [1000], prefill)
    assert a.is_finished

    seqs, prefill = scheduler.schedule()
    assert seqs == [b] and prefill is True
    scheduler.postprocess(seqs, [1001], prefill)
    assert b.is_finished and scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids
    assert len(scheduler.block_manager.free_block_ids) == 2
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


def test_no_runnable_batch_reports_resources_and_does_not_return_empty(monkeypatch):
    scheduler = make_scheduler(2)
    seq = Sequence([10], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    # Fault injection: allocation refuses a valid request. This is not a
    # claim that healthy metadata would reject this particular workload.
    with monkeypatch.context() as fault:
        fault.setattr(scheduler.block_manager, 'can_allocate', lambda seq: -1)
        with pytest.raises(SchedulerNoProgressError) as exc:
            scheduler.schedule()
        assert 'waiting=1, running=0' in str(exc.value)
        assert 'free_blocks=2, used_blocks=0, total_blocks=2' in str(exc.value)

    assert list(scheduler.waiting) == [seq]
    assert not seq.block_table
    assert seq.completion_token_ids == []
    # Retry only after the injected cause is removed, not in a blind loop.
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert scheduler.is_finished()
    assert seq.is_finished


@pytest.mark.parametrize('blocks, prompt_length', [(1, 257), (2, 513)])
def test_prompt_larger_than_entire_pool_is_rejected_without_mutation(blocks, prompt_length):
    scheduler = make_scheduler(blocks)
    valid = Sequence([10], SamplingParams(max_tokens=1))
    scheduler.add(valid)
    oversized = Sequence(list(range(prompt_length)), SamplingParams(max_tokens=1))
    with pytest.raises(ValueError, match='prompt requires'):
        scheduler.add(oversized)

    assert list(scheduler.waiting) == [valid]
    assert not scheduler.running
    assert not oversized.block_table
    assert len(scheduler.block_manager.free_block_ids) == blocks
    assert not scheduler.block_manager.used_block_ids

    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert valid.is_finished
    assert scheduler.is_finished()


def test_exact_capacity_prompt_can_generate_final_token_without_extra_kv_block():
    scheduler = make_scheduler(1)
    seq = Sequence(list(range(256)), SamplingParams(max_tokens=1))
    scheduler.add(seq)
    seqs, prefill = scheduler.schedule()
    assert seqs == [seq] and prefill is True
    assert len(seq.block_table) == 1
    scheduler.postprocess(seqs, [1000], prefill)
    # The last output token is never fed back: no second KV block is needed.
    assert len(seq) == 257
    assert seq.is_finished
    assert scheduler.is_finished()
    assert not seq.block_table
    assert not scheduler.block_manager.used_block_ids
    assert list(scheduler.block_manager.free_block_ids) == [0]


@pytest.mark.parametrize('prompt_length, max_tokens', [(256, 2), (255, 3), (1, 257)])
def test_generation_that_cannot_fit_is_rejected_before_execution(prompt_length, max_tokens):
    scheduler = make_scheduler(1)
    seq = Sequence(list(range(prompt_length)), SamplingParams(max_tokens=max_tokens))
    with pytest.raises(ValueError, match='maximum generation requires 2 KV blocks'):
        scheduler.add(seq)
    assert scheduler.is_finished()
    assert seq.completion_token_ids == []
    assert seq.block_table == []
    assert list(scheduler.block_manager.free_block_ids) == [0]


def test_final_output_may_cross_capacity_without_needing_more_kv():
    scheduler = make_scheduler(1)
    seq = Sequence(list(range(255)), SamplingParams(max_tokens=2))
    scheduler.add(seq)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert len(seq) == 256 and not seq.is_finished
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert len(seq.block_table) == 1
    scheduler.postprocess(seqs, [1001], prefill)
    assert len(seq) == 257
    assert seq.completion_token_ids == [1000, 1001]
    assert seq.is_finished and scheduler.is_finished()
    assert list(scheduler.block_manager.free_block_ids) == [0]
