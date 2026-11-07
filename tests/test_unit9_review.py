import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.run import ROOT
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def make_scheduler(policy='prefill_first', decode_policy='original', model_limit=256):
    return Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=512, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=4, max_model_len=model_limit,
        scheduling_policy=policy, decode_scheduling_policy=decode_policy,
    ))


@pytest.mark.parametrize('policy', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('decode_policy', ['original', 'round_robin'])
def test_model_context_limit_is_enforced_even_with_free_kv_blocks(policy, decode_policy):
    scheduler = make_scheduler(policy, decode_policy)
    valid = Sequence([10], SamplingParams(max_tokens=1))
    scheduler.add(valid)
    too_long = Sequence(list(range(256)), SamplingParams(max_tokens=2))
    with pytest.raises(ValueError, match='257 KV positions, but max_model_len is 256'):
        scheduler.add(too_long)
    assert list(scheduler.waiting) == [valid]
    assert not scheduler.running
    assert not too_long.block_table
    assert not scheduler.block_manager.used_block_ids
    assert len(scheduler.block_manager.free_block_ids) == 4
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [1000], prefill)
    assert valid.is_finished and scheduler.is_finished()


def test_final_output_can_exceed_input_context_limit_without_being_fed_back():
    scheduler = make_scheduler()
    seq = Sequence(list(range(255)), SamplingParams(max_tokens=2))
    scheduler.add(seq)
    for token in [1000, 1001]:
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [token], prefill)
    assert len(seq) == 257
    assert seq.is_finished and seq.completion_token_ids == [1000, 1001]
    assert scheduler.is_finished() and not scheduler.block_manager.used_block_ids


def test_prompt_alone_cannot_exceed_model_context():
    scheduler = make_scheduler()
    with pytest.raises(ValueError, match='max_model_len is 256'):
        scheduler.add(Sequence(list(range(257)), SamplingParams(max_tokens=1)))
    assert scheduler.is_finished()


@pytest.mark.parametrize('limit', [0, -1, True])
def test_invalid_model_limit_rejected(limit):
    with pytest.raises(ValueError, match='max_model_len must be a positive integer'):
        make_scheduler(model_limit=limit)


def test_environment_recording_does_not_import_or_run_gpu_stack(tmp_path):
    script = '''
import runpy, sys
sys.argv = ['experiments.gpu_smoke', '--output-root', sys.argv[1]]
try:
    runpy.run_module('experiments.gpu_smoke', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'triton', 'transformers', 'flash_attn'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report_paths = list(tmp_path.glob('*/report.json'))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text())
    assert report['status'] == 'environment_recorded_gpu_not_run'
    assert report['requested']['run'] is False
    assert report['trace'] == []
    assert 'nanovllm/engine/scheduler.py' in report['source_sha256']
