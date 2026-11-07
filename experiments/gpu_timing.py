"""Synchronized, repeated real-model timing on a round-based arrival workload."""
import argparse
import atexit
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

from experiments.gpu_smoke import environment
from experiments.run import ROOT, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler


def run_workload(engine, policy, token, synchronize):
    if not engine.is_finished():
        raise RuntimeError('Previous workload did not finish')
    config = copy(engine.model_runner.config)
    config.scheduling_policy = policy
    engine.scheduler = Scheduler(config)  # Cold metadata; old GPU slots overwritten before use.
    requests, arrivals, generations, trace = {}, {}, {}, []
    synchronize()
    start = perf_counter()

    def elapsed():
        return (perf_counter() - start) * 1000

    def add(name, prompt_len, outputs):
        arrivals[name] = elapsed()
        engine.add_request([token] * prompt_len,
                           SamplingParams(max_tokens=outputs, ignore_eos=True))
        requests[name] = engine.scheduler.waiting[-1]
        generations[name] = []

    add('A', 1, 16)
    for round_id in range(80):
        if 1 <= round_id <= 4:
            add(f'B{round_id}', 129, 1)
        before = {n: s.num_completion_tokens for n, s in requests.items()}
        _, count = engine.step()
        synchronize()
        ended = elapsed()
        for name, seq in requests.items():
            if seq.num_completion_tokens > before[name]:
                generations[name].append({'round': round_id, 'ms': ended})
        trace.append({'round': round_id, 'phase': 'prefill' if count > 0 else 'decode',
                      'input_tokens': abs(count), 'ended_ms': ended})
        if round_id >= 4 and engine.is_finished():
            break
    else:
        raise RuntimeError('Workload failed to drain')
    total = elapsed()
    bm = engine.scheduler.block_manager
    if bm.used_block_ids or any(b.ref_count for b in bm.blocks):
        raise RuntimeError('Blocks remain referenced')
    for name, seq in requests.items():
        expected = 16 if name == 'A' else 1
        if seq.num_completion_tokens != expected or not seq.is_finished:
            raise RuntimeError(f'Incorrect completion: {name}')
    return {'policy': policy, 'metrics': {
        'A_first_decode_ms': generations['A'][1]['ms'],
        'A_finish_ms': generations['A'][-1]['ms'],
        'B_max_first_token_wait_ms': max(generations[n][0]['ms'] - arrivals[n]
                                       for n in requests if n != 'A'),
        'total_ms': total,
    }, 'A_first_decode_round': generations['A'][1]['round'],
        'arrival_ms': arrivals, 'generation_events': generations, 'trace': trace,
        'outputs': {n: s.completion_token_ids for n, s in requests.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be >= 2')
    now = datetime.now(timezone.utc)
    output = ROOT / 'results/unit18/gpu_timing' / (now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'gpu_timing_failed', 'environment': environment(),
              'source_sha256': source_manifest(), 'samples': [], 'warmups': [],
              'config': {'model': str(args.model), 'budget': 32, 'max_num_seqs': 4,
                         'max_model_len': 256, 'limit': 2, 'decode': 'original', 'cache': 'original',
                         'eager': True, 'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.5},
              'workload': 'A arrives round 0: prompt=1, output=16; B1..4 arrive rounds 1..4: prompt=129, output=1; ignore_eos=True.',
              'scope': 'Synchronized wall time including CPU scheduling, GPU steps and instrumentation. Model loading/initialization and warmups excluded. Round-based arrivals, NOT fixed wall-clock arrival rate; tiny synthetic-token prompts; no universal speedup claim.',
              'cache_reset': 'New Scheduler and cold block metadata for every run; shared loaded model.',
              'sampling': 'Seed 0 per run; fixed lengths, but policy changes may allocate random draws differently.'}
    try:
        import torch
        from nanovllm import LLM
        engine = LLM(str(args.model.resolve()), enforce_eager=True, tensor_parallel_size=1,
                     max_model_len=256, max_num_batched_tokens=32, max_num_seqs=4,
                     gpu_memory_utilization=0.5)
        try:
            report['model_config_sha256'] = hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest()
            report['kv_blocks'] = engine.model_runner.config.num_kvcache_blocks
            token = engine.tokenizer.encode('Hello', add_special_tokens=False)[0]
            report['prompt_token_id'] = token
            policies = ('prefill_first', 'bounded_prefill')
            # Warm BOTH policy paths twice; exclude all warmups from statistics.
            for _ in range(2):
                for policy in policies:
                    torch.manual_seed(0)
                    sample = run_workload(engine, policy, token, torch.cuda.synchronize)
                    report['warmups'].append(sample['metrics'] | {'policy': policy})
            for repeat in range(args.repeats):
                for policy in (policies if repeat % 2 == 0 else policies[::-1]):
                    print(f'Measuring repeat={repeat + 1}, policy={policy}', flush=True)
                    torch.manual_seed(0)
                    sample = run_workload(engine, policy, token, torch.cuda.synchronize)
                    sample['repeat'] = repeat + 1
                    report['samples'].append(sample)
            report['summary'] = {}
            for policy in policies:
                samples = [s for s in report['samples'] if s['policy'] == policy]
                report['summary'][policy] = {
                    metric: {'min': min(values), 'median': median(values), 'max': max(values)}
                    for metric in samples[0]['metrics']
                    for values in [[s['metrics'][metric] for s in samples]]}
        finally:
            atexit.unregister(engine.exit)
            engine.exit()
        report['status'] = 'gpu_timing_completed'
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Status: {report['status']}")
    print(json.dumps(report.get('summary', report.get('error')), indent=2))
    print(f'Saved: {output}')
    return 0 if report['status'] == 'gpu_timing_completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
