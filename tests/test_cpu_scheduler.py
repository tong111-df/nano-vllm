"""Exercise real control-plane code; token IDs stand in for model output."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler


def make_scheduler(budget=512, blocks=4, *, policy="prefill_first", limit=2,
                   decode_policy="original"):
    # Match Sequence.block_size; do not shrink it independently.
    return Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=99,
        kvcache_block_size=256, num_kvcache_blocks=blocks,
        scheduling_policy=policy, max_consecutive_prefills=limit,
        decode_scheduling_policy=decode_policy,
    ))


def step(scheduler, token):
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [token] * len(seqs), prefill)
    return prefill


def assert_released(scheduler, seq):
    bm = scheduler.block_manager
    assert scheduler.is_finished()
    assert seq.status == SequenceStatus.FINISHED
    assert seq.block_table == []
    assert not bm.used_block_ids
    assert len(set(bm.free_block_ids)) == len(bm.blocks)
    assert all(block.ref_count == 0 for block in bm.blocks)


def test_import_does_not_load_inference_dependencies():
    subprocess.run([sys.executable, '-c', '''
import sys
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
assert not {'torch', 'triton', 'transformers', 'flash_attn',
            'nanovllm.engine.model_runner'} & sys.modules.keys()
'''], check=True)


def test_request_lifecycle():
    scheduler = make_scheduler()
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=2))
    scheduler.add(seq)
    assert seq.status == SequenceStatus.WAITING
    assert step(scheduler, 20) is True
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_cached_tokens == 3
    assert seq.completion_token_ids == [20]
    assert len(seq.block_table) == 1
    assert step(scheduler, 21) is False
    assert seq.completion_token_ids == [20, 21]
    assert_released(scheduler, seq)


def test_chunked_prefill_only_samples_after_last_chunk():
    scheduler = make_scheduler(budget=2)
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
    scheduler.add(seq)
    assert step(scheduler, 20) is True
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_cached_tokens == 2
    assert seq.completion_token_ids == []
    assert step(scheduler, 21) is True
    assert seq.completion_token_ids == [21]
    assert_released(scheduler, seq)


@pytest.mark.parametrize('ignore_eos', [False, True])
def test_eos(ignore_eos):
    scheduler = make_scheduler()
    seq = Sequence([1], SamplingParams(max_tokens=2, ignore_eos=ignore_eos))
    scheduler.add(seq)
    step(scheduler, 99)
    assert seq.is_finished == (not ignore_eos)
    if ignore_eos:
        step(scheduler, 20)
    assert_released(scheduler, seq)


def test_decode_crosses_block_boundary():
    scheduler = make_scheduler(blocks=2)
    seq = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    scheduler.add(seq)
    step(scheduler, 300)
    assert len(seq) == 257
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert len(seq.block_table) == 2
    scheduler.postprocess(seqs, [301], prefill)
    assert_released(scheduler, seq)


def test_freed_full_prefix_is_reused():
    scheduler = make_scheduler(blocks=2)
    prompt = list(range(257))

    first = Sequence(prompt, SamplingParams(max_tokens=1))
    scheduler.add(first)
    step(scheduler, 300)
    assert_released(scheduler, first)

    # 第二个请求使用完全相同的 prompt。
    second = Sequence(prompt, SamplingParams(max_tokens=1))
    scheduler.add(second)

    seqs, prefill = scheduler.schedule()
    assert second.num_cached_tokens == 256
    assert second.num_scheduled_tokens == 1

    scheduler.postprocess(seqs, [301], prefill)
    assert_released(scheduler, second)
    
def test_preempt_preserves_generated_tokens():
    scheduler = make_scheduler(blocks=2)
    seq = Sequence([10, 11, 12], SamplingParams(max_tokens=3))
    scheduler.add(seq)

    # 完成 Prefill，生成第一个 token。
    step(scheduler, 20)
    assert seq.status == SequenceStatus.RUNNING
    tokens_before = seq.token_ids.copy()

    # 模拟 schedule() 先取出请求，再抢占它。
    scheduler.running.remove(seq)
    scheduler.preempt(seq)

    assert seq.status == SequenceStatus.WAITING
    assert scheduler.waiting[0] is seq
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert not scheduler.block_manager.used_block_ids

    # KV 分配被释放，但生成进度保留。
    assert seq.token_ids == tokens_before
    assert seq.completion_token_ids == [20]
    assert seq.num_completion_tokens == 1
    assert seq.max_tokens == 3
    assert not seq.is_finished
        # 恢复时先执行 Prefill，处理保留下来的序列。
    assert step(scheduler, 21) is True
    assert seq.status == SequenceStatus.RUNNING
    assert seq.completion_token_ids == [20, 21]
    assert seq.num_completion_tokens == 2

    # 再执行一次 Decode，累计生成 3 个 token 后结束。
    assert step(scheduler, 22) is False
    assert seq.completion_token_ids == [20, 21, 22]
    assert_released(scheduler, seq)
    
def test_scheduler_automatically_preempts():
    scheduler = make_scheduler(blocks=2)

    a = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    b = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)

    # 两个请求完成 Prefill，各占一块，各生成一个 token。
    assert step(scheduler, 300) is True
    assert list(scheduler.running) == [a, b]
    assert len(a) == 257
    assert len(b) == 2
    assert len(scheduler.block_manager.free_block_ids) == 0
    b_tokens = b.token_ids.copy()

    # A 需要第二块：调度器自动抢占队尾的 B。
    seqs, prefill = scheduler.schedule()

    assert prefill is False
    assert seqs == [a]
    assert len(a.block_table) == 2
    assert b.status == SequenceStatus.WAITING
    assert list(scheduler.waiting) == [b]
    assert b.block_table == []
    assert b.token_ids == b_tokens

    # A 生成第二个 token，结束并释放两块。
    scheduler.postprocess(seqs, [301], prefill)
    assert a.is_finished
    assert len(scheduler.block_manager.free_block_ids) == 2

    # B 恢复 Prefill，生成第二个 token，也结束。
    assert step(scheduler, 302) is True
    assert b.completion_token_ids == [300, 302]
    assert_released(scheduler, b)


@pytest.mark.parametrize('decode_policy', ['original', 'round_robin'])
def test_bounded_decode_can_preempt_another_running_request(decode_policy):
    scheduler = make_scheduler(budget=512, blocks=3, policy="bounded_prefill",
                               decode_policy=decode_policy)
    a = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    b = Sequence([500], SamplingParams(max_tokens=2))
    scheduler.add(a)
    scheduler.add(b)
    assert step(scheduler, 600) is True

    # Use a smaller budget after setup to create a partially prefetched C.
    scheduler.max_num_batched_tokens = 2
    c = Sequence([700, 701, 702, 703, 704], SamplingParams(max_tokens=1))
    scheduler.add(c)
    assert step(scheduler, 800) is True
    assert step(scheduler, 800) is True
    assert not scheduler.block_manager.free_block_ids

    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    assert len(a.block_table) == 2
    assert b.status == SequenceStatus.WAITING
    assert b.block_table == []
    assert b.completion_token_ids == [600]
    scheduler.postprocess(seqs, [601], prefill)
    assert a.is_finished

    assert step(scheduler, 602) is True  # Restore B and finish it.
    assert b.is_finished
    assert step(scheduler, 801) is True  # Finish C's last Prefill chunk.
    assert_released(scheduler, c)


def test_bounded_prefill_can_finish_when_no_decode_has_space():
    scheduler = make_scheduler(budget=512, blocks=2, policy="bounded_prefill")
    a = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 600) is True

    scheduler.max_num_batched_tokens = 2
    b = Sequence([700, 701, 702, 703, 704], SamplingParams(max_tokens=1))
    scheduler.add(b)
    assert step(scheduler, 800) is True
    assert step(scheduler, 800) is True
    assert not scheduler.block_manager.can_append(a)

    # A needs a new block. B owns the only other block but is still WAITING.
    # Let B finish its last chunk rather than forcing an impossible Decode.
    assert step(scheduler, 801) is True
    assert b.is_finished
    assert a.status == SequenceStatus.RUNNING
    assert step(scheduler, 601) is False
    assert_released(scheduler, a)


def test_prefill_delays_running_decode():
    scheduler = make_scheduler(budget=2)

    # 先让 A 完成 Prefill，生成第一个 token。
    a = Sequence([10], SamplingParams(max_tokens=3))
    scheduler.add(a)
    assert step(scheduler, 20) is True
    assert a.completion_token_ids == [20]

    # A 等待 Decode 时，B 到达。
    b = Sequence([30, 31, 32, 33, 34], SamplingParams(max_tokens=1))
    scheduler.add(b)

    # B 连续占用三轮 Prefill，A 始终没有生成进展。
    for expected_cached in [2, 4]:
        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [b]
        scheduler.postprocess(seqs, [40], prefill)

        assert b.num_cached_tokens == expected_cached
        assert b.status == SequenceStatus.WAITING
        assert a.status == SequenceStatus.RUNNING
        assert a.completion_token_ids == [20]

    # 第三轮完成 B 的 Prefill，B 生成一个 token 后结束。
    seqs, prefill = scheduler.schedule()
    assert prefill is True
    assert seqs == [b]
    assert b.num_scheduled_tokens == 1
    scheduler.postprocess(seqs, [41], prefill)

    assert b.is_finished
    assert b.completion_token_ids == [41]
    assert a.completion_token_ids == [20]

    # 第四轮，A 才恢复 Decode。
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    scheduler.postprocess(seqs, [21], prefill)
    assert a.completion_token_ids == [20, 21]

    # 再一次 Decode，完成 A 并检查资源释放。
    assert step(scheduler, 22) is False
    assert a.completion_token_ids == [20, 21, 22]
    assert_released(scheduler, a)
@pytest.mark.parametrize(
    "budget, expected_rounds",
    [(1, 5), (2, 3), (5, 1)],
)
def test_prefill_budget_changes_decode_wait(budget, expected_rounds):
    scheduler = make_scheduler(budget=budget)

    a = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    b = Sequence([30, 31, 32, 33, 34], SamplingParams(max_tokens=1))
    scheduler.add(b)

    blocked_rounds = 0

    # 限制循环次数，避免出错时测试无限运行。
    for _ in range(10):
        if b.is_finished:
            break

        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [b]
        scheduler.postprocess(seqs, [40], prefill)

        # 每轮 B 执行期间，A 都没有生成新 token。
        assert a.completion_token_ids == [20]
        blocked_rounds += 1

    assert b.is_finished
    assert blocked_rounds == expected_rounds

    # B 结束后，A 才恢复 Decode。
    assert step(scheduler, 21) is False
    assert a.completion_token_ids == [20, 21]
    assert_released(scheduler, a)

    print(f"budget={budget}, A等待轮数={blocked_rounds}")

def test_new_arrivals_keep_delaying_decode():
    scheduler = make_scheduler(budget=2)

    a = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    # 连续三轮，每轮调度前都到达一个短请求。
    for i in range(3):
        newcomer = Sequence(
            [30 + i],
            SamplingParams(max_tokens=1),
        )
        scheduler.add(newcomer)

        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [newcomer]
        scheduler.postprocess(seqs, [40 + i], prefill)

        assert newcomer.is_finished
        assert a.status == SequenceStatus.RUNNING
        assert a.completion_token_ids == [20]

    # 停止加入新请求后，A 才获得 Decode 机会。
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    scheduler.postprocess(seqs, [21], prefill)

    assert a.completion_token_ids == [20, 21]
    assert_released(scheduler, a)

@pytest.mark.parametrize("limit", [1, 2, 3])
def test_bounded_prefill_configurable_limit(limit):
    scheduler = make_scheduler(budget=2, policy="bounded_prefill", limit=limit)
    a = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    # Enough prompt tokens to remain in Prefill after the configured limit.
    b = Sequence(list(range(30, 30 + 2 * limit + 1)),
                 SamplingParams(max_tokens=1))
    scheduler.add(b)
    for _ in range(limit):
        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [b]
        scheduler.postprocess(seqs, [40], prefill)
        assert a.completion_token_ids == [20]

    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    scheduler.postprocess(seqs, [21], prefill)
    assert a.is_finished
    assert step(scheduler, 41) is True
    assert_released(scheduler, b)


def test_bounded_prefill_counts_batches_not_requests():
    scheduler = make_scheduler(budget=2, policy="bounded_prefill")
    a = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    # Each batch contains two different short Prefill requests.
    for batch in range(2):
        newcomers = [Sequence([30 + batch * 2 + i], SamplingParams(max_tokens=1))
                     for i in range(2)]
        for seq in newcomers:
            scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == newcomers
        scheduler.postprocess(seqs, [40, 41], prefill)
        assert all(seq.is_finished for seq in newcomers)
        assert scheduler.consecutive_prefills == batch + 1

    # A new arrival is waiting, so Decode must win because of the limit.
    last = Sequence([50], SamplingParams(max_tokens=1))
    scheduler.add(last)
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    scheduler.postprocess(seqs, [21], prefill)
    assert step(scheduler, 51) is True
    assert_released(scheduler, last)


def test_bounded_prefill_gives_decode_a_turn():
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=2,
        eos=99,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
        scheduling_policy="bounded_prefill",
        max_consecutive_prefills=2,
    )
    scheduler = Scheduler(config)

    # 准备阶段：此时没有其他请求等待 Decode，不计入连续让行次数。
    a = Sequence([10], SamplingParams(max_tokens=2))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    b = Sequence([30, 31, 32, 33, 34], SamplingParams(max_tokens=1))
    scheduler.add(b)

    # 前两轮允许 B 执行 Prefill。
    for cached in [2, 4]:
        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [b]
        scheduler.postprocess(seqs, [40], prefill)
        assert b.num_cached_tokens == cached
        assert a.completion_token_ids == [20]

    # 第三轮必须让给 A 执行 Decode。
    seqs, prefill = scheduler.schedule()
    assert prefill is False
    assert seqs == [a]
    scheduler.postprocess(seqs, [21], prefill)
    assert a.is_finished
    assert b.status == SequenceStatus.WAITING

    # A 完成后，B 继续处理剩余的一个 prompt token。
    assert step(scheduler, 41) is True
    assert b.completion_token_ids == [41]
    assert_released(scheduler, b)


def test_bounded_prefill_without_decode_waiters():
    scheduler = make_scheduler(budget=2, policy="bounded_prefill")
    b = Sequence([30, 31, 32, 33, 34], SamplingParams(max_tokens=1))
    scheduler.add(b)

    # No Decode waiter: allow all three chunks, despite a limit of two.
    for _ in range(3):
        seqs, prefill = scheduler.schedule()
        assert prefill is True
        assert seqs == [b]
        scheduler.postprocess(seqs, [40], prefill)
        assert scheduler.consecutive_prefills == 0

    assert b.completion_token_ids == [40]
    assert_released(scheduler, b)


def test_bounded_prefill_resets_after_each_decode():
    scheduler = make_scheduler(budget=2, policy="bounded_prefill")
    a = Sequence([10], SamplingParams(max_tokens=3))
    scheduler.add(a)
    assert step(scheduler, 20) is True

    b = Sequence(list(range(30, 39)), SamplingParams(max_tokens=1))
    scheduler.add(b)

    # Two full cycles must each permit two Prefill rounds before Decode.
    for cycle in range(2):
        for count in (1, 2):
            seqs, prefill = scheduler.schedule()
            assert prefill is True
            assert seqs == [b]
            scheduler.postprocess(seqs, [40], prefill)
            assert scheduler.consecutive_prefills == count

        seqs, prefill = scheduler.schedule()
        assert prefill is False
        assert seqs == [a]
        scheduler.postprocess(seqs, [21 + cycle], prefill)
        assert scheduler.consecutive_prefills == 0
        assert a.num_completion_tokens == cycle + 2

    assert a.is_finished
    assert b.num_cached_tokens == 8
    assert step(scheduler, 41) is True
    assert b.completion_token_ids == [41]
    assert_released(scheduler, b)
