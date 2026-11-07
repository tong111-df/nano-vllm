"""Fixed cache loads: token accounting, selection work and CPU management timing."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace
from uuid import uuid4

from experiments.run import ROOT, check_keys, git_state, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


DEFAULT_CONFIG = ROOT / 'experiments' / 'configs' / 'unit15_workloads.json'


class MeasuredPolicy:
    """Measure delegate calls only, not BlockManager queue removal or hashing."""

    def __init__(self, delegate, name):
        self.delegate, self.name = delegate, name
        self.select_calls = self.acquire_calls = self.release_calls = 0
        self.candidate_entries = self.policy_ns = 0

    def select_victim(self, candidates):
        self.select_calls += 1
        # Algorithmic entries inspected by these two known implementations:
        # queue head vs min scanning the complete free candidate deque.
        self.candidate_entries += 1 if self.name == 'original' else len(candidates)
        start = perf_counter_ns()
        result = self.delegate.select_victim(candidates)
        self.policy_ns += perf_counter_ns() - start
        return result

    def on_acquire(self, block_id):
        self.acquire_calls += 1
        start = perf_counter_ns()
        self.delegate.on_acquire(block_id)
        self.policy_ns += perf_counter_ns() - start

    def on_release(self, block_id):
        self.release_calls += 1
        start = perf_counter_ns()
        self.delegate.on_release(block_id)
        self.policy_ns += perf_counter_ns() - start


def run_case(prefix_keys, policy, blocks):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks,
        cache_eviction_policy=policy,
    ))
    bm = scheduler.block_manager
    measured = MeasuredPolicy(bm.eviction_policy, policy)
    bm.eviction_policy = measured
    trace, control_plane_ns = [], 0
    for index, key in enumerate(prefix_keys):
        # Fixed prompt per workload position, independent of policy or timing.
        prompt = list(range(key * 1000, key * 1000 + 256)) + [1000000 + index]
        seq = Sequence(prompt, SamplingParams(max_tokens=1))
        start = perf_counter_ns()
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        control_plane_ns += perf_counter_ns() - start
        assert batch == [seq] and prefill
        cached, computed = seq.num_cached_tokens, seq.num_scheduled_tokens
        table = seq.block_table.copy()
        start = perf_counter_ns()
        scheduler.postprocess(batch, [2000000 + index], prefill)
        control_plane_ns += perf_counter_ns() - start
        assert seq.is_finished and seq.completion_token_ids == [2000000 + index]
        assert scheduler.is_finished() and not bm.used_block_ids
        assert len(set(bm.free_block_ids)) == blocks
        assert all(block.ref_count == 0 for block in bm.blocks)
        trace.append({'request_index': index, 'prefix_key': key, 'prompt_tokens': len(prompt),
                      'cached_tokens': cached, 'computed_prompt_tokens': computed,
                      'block_table_before_completion': table, 'free_after': list(bm.free_block_ids)})
    total = sum(row['prompt_tokens'] for row in trace)
    cached = sum(row['cached_tokens'] for row in trace)
    computed = sum(row['computed_prompt_tokens'] for row in trace)
    assert cached + computed == total
    metrics = {
        'request_count': len(trace), 'total_prompt_tokens': total,
        'cached_tokens': cached, 'computed_prompt_tokens': computed,
        'prompt_token_hit_rate': cached / total,
        'requests_with_prefix_hit': sum(row['cached_tokens'] > 0 for row in trace),
        'select_calls': measured.select_calls, 'acquire_calls': measured.acquire_calls,
        'release_calls': measured.release_calls,
        'selection_candidate_entries': measured.candidate_entries,
        'recency_metadata_entries': len(getattr(measured.delegate, 'last_access', {})),
    }
    return {'metrics': metrics, 'trace': trace,
            'timing_ns': {'policy_delegate': measured.policy_ns,
                          'instrumented_control_plane': control_plane_ns}}


def validate_config(raw):
    check_keys(raw, {'schema_version', 'block_counts', 'policies', 'workloads'}, 'config')
    if type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    capacities = raw.get('block_counts')
    if not isinstance(capacities, list) or not capacities or any(
            type(n) is not int or n < 2 for n in capacities):
        raise ValueError('block_counts must be a nonempty list of integers >= 2')
    policies = raw.get('policies')
    if not isinstance(policies, list) or not policies or any(
            name not in ('original', 'lru') for name in policies):
        raise ValueError('policies must contain original or lru')
    workloads = raw.get('workloads')
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError('workloads must be a nonempty object')
    for name, keys in workloads.items():
        if not isinstance(name, str) or not name or not isinstance(keys, list) or not keys or any(
                type(key) is not int or not 0 <= key <= 100000 for key in keys):
            raise ValueError('workload values must be nonempty lists of prefix IDs in 0..100000')
    return raw


def run_suite(raw, *, repeats=5, warmup=1, output_root=None):
    config = validate_config(raw)
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError('repeats must be positive and warmup nonnegative integers')
    manifest = source_manifest()
    cases = []
    for blocks in config['block_counts']:
        for workload, keys in config['workloads'].items():
            for policy in config['policies']:
                for _ in range(warmup):
                    run_case(keys, policy, blocks)
                samples = [run_case(keys, policy, blocks) for _ in range(repeats)]
                first = samples[0]
                assert all(sample['metrics'] == first['metrics'] and sample['trace'] == first['trace']
                           for sample in samples)
                timing = {name: {'samples_ns': [sample['timing_ns'][name] for sample in samples],
                                 'min_ns': min(sample['timing_ns'][name] for sample in samples),
                                 'median_ns': median(sample['timing_ns'][name] for sample in samples),
                                 'max_ns': max(sample['timing_ns'][name] for sample in samples)}
                          for name in first['timing_ns']}
                cases.append({'workload': workload, 'blocks': blocks, 'policy': policy,
                              'metrics': first['metrics'], 'trace': first['trace'], 'timing': timing})
    if source_manifest() != manifest:
        raise RuntimeError('Source changed during experiment; rerun before archiving')
    report = {
        'config': config, 'repeats': repeats, 'warmup_runs_per_case': warmup,
        'cache_reset': 'Fresh scheduler and cold block pool on EVERY warmup and measured run.',
        'prompt_rule': 'prefix ID k => tokens range(k*1000,k*1000+256), then tail token 1000000+request_index; output length=1.',
        'fixed_scheduler': {'max_num_seqs': 1, 'budget': 1024, 'block_size': 256,
                            'phase': 'prefill_first', 'decode': 'original', 'eos': -1},
        'access_definition': 'Successful block reference acquisition; not queries, releases or attention reads.',
        'timing_scope': 'Instrumented CPU-only control-plane and delegate-call timing, including timer overhead; no model computation or GPU inference. Trace construction and scheduler initialization excluded.',
        'candidate_entries_definition': 'Algorithmic entries inspected: 1 per original selection, len(free_candidates) per LRU selection; not elapsed time.',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': {name: version(name) for name in ('numpy', 'xxhash')}},
        'source_sha256': manifest, 'git': git_state(), 'cases': cases,
    }
    now = datetime.now(timezone.utc)
    report['created_at_utc'] = now.isoformat()
    parent = Path(output_root) if output_root is not None else ROOT / 'results' / 'unit15'
    output = parent / (now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = [{key: case[key] for key in ('workload', 'blocks', 'policy')} |
               case['metrics'] | {'policy_cpu_us_median': case['timing']['policy_delegate']['median_ns'] / 1000}
               for case in cases]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()
    try:
        output, report = run_suite(json.loads(args.config.read_text(encoding='utf-8-sig')),
                                    repeats=args.repeats, warmup=args.warmup)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Experiment failed: {error}\n')
    print('CPU-only management timings; NOT GPU inference latency. Fresh pool per run.')
    print('blocks workload policy   cached computed candidate_entries policy_cpu_us_med')
    for case in report['cases']:
        m = case['metrics']
        elapsed_us = case['timing']['policy_delegate']['median_ns'] / 1000
        print(f"{case['blocks']:>6} {case['workload']:<8} {case['policy']:<8} "
              f"{m['cached_tokens']:>6} {m['computed_prompt_tokens']:>8} "
              f"{m['selection_candidate_entries']:>17} {elapsed_us:>17.2f}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
