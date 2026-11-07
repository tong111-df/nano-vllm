"""Record environment by default; --run explicitly enables local-model GPU smoke."""
import argparse
import atexit
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def environment():
    packages = {}
    for name in ('torch', 'triton', 'transformers', 'flash-attn', 'numpy',
                 'xxhash', 'safetensors', 'tqdm'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
            capture_output=True, text=True, timeout=10, check=True)
        nvidia_smi = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        nvidia_smi = f'unavailable: {type(error).__name__}'
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'nvidia_smi': nvidia_smi}


def run_gpu(args, report):
    if platform.system() != 'Linux':
        raise RuntimeError('This NCCL-based smoke entry targets Linux with NVIDIA CUDA')
    if args.model is None or not (args.model / 'config.json').is_file():
        raise ValueError('--run requires --model pointing to a local model directory with config.json')
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError('CUDA and the NCCL backend must be available')
    from nanovllm import LLM

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    report['cuda'] = {'torch_cuda_version': torch.version.cuda,
                      'device': torch.cuda.get_device_name(0)}
    # Start small: eager execution and one GPU; no CUDA graph or TP claims.
    llm = LLM(str(args.model.resolve()), enforce_eager=True, tensor_parallel_size=1,
              max_model_len=256, max_num_batched_tokens=16, max_num_seqs=4,
              gpu_memory_utilization=0.5, scheduling_policy=args.policy,
              decode_scheduling_policy=args.decode_policy, max_consecutive_prefills=2,
              cache_eviction_policy=args.cache_policy)
    config = llm.model_runner.config
    report['effective_config'] = {
        name: getattr(config, name) for name in (
            'max_model_len', 'max_num_batched_tokens', 'max_num_seqs',
            'kvcache_block_size', 'num_kvcache_blocks', 'enforce_eager',
            'tensor_parallel_size', 'scheduling_policy', 'decode_scheduling_policy',
            'max_consecutive_prefills', 'gpu_memory_utilization', 'cache_eviction_policy')}
    report['model_config_sha256'] = hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest()
    try:
        exercise_engine(llm, report)
    finally:
        # The engine normally registers this at process exit. Close once here
        # so a successful report also requires successful runner cleanup.
        atexit.unregister(llm.exit)
        llm.exit()


def exercise_engine(llm, report):
    from nanovllm import SamplingParams
    requests = {}
    token_ids = llm.tokenizer.encode('Hello', add_special_tokens=False)
    if not token_ids:
        raise RuntimeError('Tokenizer returned no tokens for smoke prompt')
    token = token_ids[0]

    def add(name, prompt_length, outputs):
        llm.add_request([token] * prompt_length,
                        SamplingParams(max_tokens=outputs, ignore_eos=True))
        requests[name] = llm.scheduler.waiting[-1]

    add('A', 1, 4)
    add('C', 1, 4)
    for round_id in range(32):
        if round_id == 1:
            add('B', 33, 1)  # Three Prefill chunks with budget 16.
        before = {name: seq.num_completion_tokens for name, seq in requests.items()}
        outputs, num_tokens = llm.step()
        report['trace'].append({
            'round': round_id, 'phase': 'prefill' if num_tokens > 0 else 'decode',
            'input_token_count': abs(num_tokens),
            'completion_counts_before': before,
            'completion_counts_after': {n: s.num_completion_tokens for n, s in requests.items()},
            'finished_request_ids': [seq_id for seq_id, _ in outputs],
        })
        if round_id >= 1 and llm.is_finished():
            break
    else:
        raise RuntimeError('Smoke workload did not finish within 32 rounds')
    for name, length in [('A', 4), ('B', 1), ('C', 4)]:
        seq = requests[name]
        if not seq.is_finished or seq.num_completion_tokens != length or seq.block_table:
            raise RuntimeError(f'{name}: incorrect completion or unreleased block table')
    bm = llm.scheduler.block_manager
    if bm.used_block_ids or any(b.ref_count for b in bm.blocks):
        raise RuntimeError('KV blocks still referenced after all requests finished')
    if not any(row['phase'] == 'decode' for row in report['trace']):
        raise RuntimeError('No Decode exercised')
    if not any(row['phase'] == 'prefill' and row['completion_counts_before'] ==
               row['completion_counts_after'] for row in report['trace']):
        raise RuntimeError('No partial Prefill exercised')
    report['outputs'] = {n: s.completion_token_ids for n, s in requests.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Actually load model and execute GPU inference')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--policy', choices=['prefill_first', 'bounded_prefill'], default='bounded_prefill')
    parser.add_argument('--decode-policy', choices=['original', 'round_robin'], default='round_robin')
    parser.add_argument('--cache-policy', choices=['original', 'lru'], default='original')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results' / 'gpu_smoke')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    output = args.output_root / (now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / 'nanovllm').rglob('*.py')) + [Path(__file__).resolve()]
    report = {
        'created_at_utc': now.isoformat(), 'environment': environment(),
        'requested': {'run': args.run, 'model': str(args.model) if args.model else None,
                      'policy': args.policy, 'decode_policy': args.decode_policy,
                      'cache_policy': args.cache_policy},
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        'status': 'environment_recorded_gpu_not_run', 'trace': [],
        'scope': 'Compatibility smoke only; not a performance or numerical-equivalence benchmark.',
    }
    code = 0
    if args.run:
        try:
            run_gpu(args, report)
            report['status'] = 'gpu_smoke_passed'
        except Exception as error:
            report['status'] = 'gpu_smoke_failed'
            report['error'] = {'type': type(error).__name__, 'message': str(error)}
            code = 1
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Status: {report['status']}")
    if 'error' in report:
        print(f"Error: {report['error']['type']}: {report['error']['message']}")
    print(f'Saved: {output}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
