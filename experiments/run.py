"""Run fixed CPU workloads from JSON and archive each successful comparison."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

from experiments.day4_arrivals import run_case as run_arrivals
from experiments.unit5_fairness import run_case as run_mixed


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'mixed': {'leading_tokens': 8, 'short_arrival_rounds': 6, 'budget': 8},
    'arrivals': {'arrival_count': 6, 'a_max_tokens': 4,
                 'arrivals_per_round': 1, 'budget': 2, 'short_prompt_tokens': 1},
}


def check_keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'{label}: unknown fields {sorted(unknown)}')


def resolve_config(raw):
    """Reject typos before running; expand defaults into an auditable config."""
    check_keys(raw, {'schema_version', 'workload', 'parameters', 'strategies'}, 'config')
    if type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    workload = raw.get('workload')
    if not isinstance(workload, str) or workload not in DEFAULTS:
        raise ValueError('workload must be mixed or arrivals')
    params = raw.get('parameters', {})
    check_keys(params, DEFAULTS[workload].keys(), 'parameters')
    params = DEFAULTS[workload] | params
    for name, value in params.items():
        minimum = 0 if name == 'short_arrival_rounds' else 1
        if name == 'a_max_tokens':
            minimum = 2
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    strategies = raw.get('strategies')
    if not isinstance(strategies, list) or not strategies:
        raise ValueError('strategies must be a nonempty list')
    resolved = []
    for strategy in strategies:
        check_keys(strategy, {'policy', 'decode_policy', 'max_consecutive_prefills',
                              'cache_eviction_policy'}, 'strategy')
        item = {'decode_policy': 'original', 'max_consecutive_prefills': 2,
                'cache_eviction_policy': 'original'} | strategy
        if item.get('policy') not in ('prefill_first', 'bounded_prefill'):
            raise ValueError('policy must be prefill_first or bounded_prefill')
        if item['decode_policy'] not in ('original', 'round_robin'):
            raise ValueError('decode_policy must be original or round_robin')
        if item['cache_eviction_policy'] not in ('original', 'lru'):
            raise ValueError('cache_eviction_policy must be original or lru')
        limit = item['max_consecutive_prefills']
        if type(limit) is not int or limit < 1:
            raise ValueError('max_consecutive_prefills must be a positive integer')
        resolved.append(item)
    return {'schema_version': 1, 'workload': workload,
            'parameters': params, 'strategies': resolved}


def source_manifest():
    # Include package initializers and all engine files, not just committed HEAD.
    paths = {ROOT / 'requirements-cpu.txt'}
    for directory in ('nanovllm', 'experiments'):
        paths.update((ROOT / directory).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def git_state():
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    try:
        return {'head': git('rev-parse', 'HEAD'),
                'dirty': bool(git('status', '--porcelain', '--untracked-files=normal'))}
    except (OSError, subprocess.CalledProcessError):
        return {'head': None, 'dirty': None}


def run_experiment(raw, output_root=None):
    config = resolve_config(raw)
    manifest = source_manifest()
    runner = run_mixed if config['workload'] == 'mixed' else run_arrivals
    cases = []
    for strategy in config['strategies']:
        result = runner(**config['parameters'], **strategy)
        cases.append({'strategy': deepcopy(strategy), 'result': result})
    if source_manifest() != manifest:
        raise RuntimeError('Source changed during experiment; rerun before archiving')
    created = datetime.now(timezone.utc)
    report = {
        'schema_version': 1, 'created_at_utc': created.isoformat(),
        'upstream_base': 'bb823b3e06983d71485a8e1f23715ebd87d98ef8',
        'config': config,
        'fixed_scheduler': {'max_num_seqs': 2 if config['workload'] == 'mixed' else 4,
                            'num_kvcache_blocks': 16, 'kvcache_block_size': 256, 'eos': -1},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: version(name) for name in ('numpy', 'xxhash')}},
        'git': git_state(), 'source_sha256': manifest,
        'units': 'Scheduling rounds, synthetic output, no GPU timing; same-round wait=0.',
        'cases': cases,
    }
    parent = Path(output_root) if output_root is not None else ROOT / 'results' / 'unit7'
    output = parent / (created.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in [('config.json', config), ('report.json', report)]:
        (output / filename).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, help='Parent for a NEW run directory')
    args = parser.parse_args()
    try:
        raw = json.loads(args.config.read_text(encoding='utf-8-sig'))
        output, report = run_experiment(raw, args.output_root)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Experiment failed: {error}\n')
    for case in report['cases']:
        strategy, requests = case['strategy'], case['result']['requests']
        target = 'C' if report['config']['workload'] == 'mixed' else 'A'
        metric = requests[target]
        print(f"{strategy['policy']}/{strategy['decode_policy']} "
              f"cache={strategy['cache_eviction_policy']} "
              f"limit={strategy['max_consecutive_prefills']}: "
              f"{target} decode_rounds={metric['decode_rounds']}, "
              f"finish_round={metric['finish_round']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
