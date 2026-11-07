from types import SimpleNamespace
import pytest

from experiments.gpu_timing import run_workload
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class CPUControlPlane:
    """Validate only the workload harness, NOT the GPU timing implementation."""
    def __init__(self):
        self.model_runner = SimpleNamespace(config=SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=32, eos=-1,
            kvcache_block_size=256, num_kvcache_blocks=8))
        self.scheduler = Scheduler(self.model_runner.config)

    def is_finished(self):
        return self.scheduler.is_finished()

    def add_request(self, prompt, params):
        self.scheduler.add(Sequence(prompt, params))

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        count = sum(s.num_scheduled_tokens for s in seqs) if prefill else -len(seqs)
        self.scheduler.postprocess(seqs, [1000 + s.num_completion_tokens for s in seqs], prefill)
        return [], count


@pytest.mark.parametrize('policy,first_decode', [('prefill_first', 21), ('bounded_prefill', 3)])
def test_timing_workload_keeps_fixed_lengths_arrivals_and_cold_scheduler(policy, first_decode):
    engine = CPUControlPlane()
    for _ in range(2):
        result = run_workload(engine, policy, 10, lambda: None)
        assert result['A_first_decode_round'] == first_decode
        assert {n: len(ids) for n, ids in result['outputs'].items()} == {
            'A': 16, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1}
        assert len(result['trace']) == 36
        assert sum(row['input_tokens'] for row in result['trace']) == 532
        assert len(set(engine.scheduler.block_manager.free_block_ids)) == 8
