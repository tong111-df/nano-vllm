"""Expose Decode queue bias with three requests and a two-request batch limit."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_case(policy, leading_tokens=4, *, decode_policy="original", short_arrival_rounds=0,
             budget=8, max_consecutive_prefills=2, cache_eviction_policy="original"):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=budget, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=16,
        scheduling_policy=policy, max_consecutive_prefills=max_consecutive_prefills,
        decode_scheduling_policy=decode_policy,
        cache_eviction_policy=cache_eviction_policy,
    ))
    requests = {
        name: Sequence([10 + i], SamplingParams(max_tokens=length))
        for i, (name, length) in enumerate(
            [('A', leading_tokens), ('B', leading_tokens), ('C', 2)]
        )
    }
    names = {seq.seq_id: name for name, seq in requests.items()}
    arrivals = {name: 0 for name in requests}
    generated = {name: [] for name in requests}
    decoded = {name: [] for name in requests}
    trace = []
    for seq in requests.values():
        scheduler.add(seq)

    for round_id in range(2 * leading_tokens + short_arrival_rounds + 10):
        # Keep rounds 0/1 as common setup, then add one short request per round.
        if 2 <= round_id < 2 + short_arrival_rounds:
            name = f'D{round_id - 1}'
            seq = Sequence([100 + round_id], SamplingParams(max_tokens=1))
            requests[name] = seq
            names[seq.seq_id] = name
            arrivals[name] = round_id
            generated[name], decoded[name] = [], []
            scheduler.add(seq)
        before = {n: s.num_completion_tokens for n, s in requests.items()}
        running_before = [names[s.seq_id] for s in scheduler.running]
        waiting_before = [names[s.seq_id] for s in scheduler.waiting]
        seqs, prefill = scheduler.schedule()
        selected = [names[s.seq_id] for s in seqs]
        scheduler.postprocess(seqs, [1000 + s.num_completion_tokens for s in seqs], prefill)
        for name in selected:
            if not prefill:
                decoded[name].append(round_id)
            if requests[name].num_completion_tokens > before[name]:
                generated[name].append(round_id)
        trace.append({
            'round': round_id, 'phase': 'prefill' if prefill else 'decode',
            'running_before': running_before, 'waiting_before': waiting_before,
            'selected': selected,
            'running_after': [names[s.seq_id] for s in scheduler.running],
            'completion_counts_after': {
                n: s.num_completion_tokens for n, s in requests.items()
            },
        })
        if round_id >= 1 + short_arrival_rounds and scheduler.is_finished():
            break
    else:
        raise RuntimeError('Workload failed to drain')

    metrics = {}
    for name, seq in requests.items():
        assert seq.is_finished and seq.num_completion_tokens == seq.max_tokens
        rounds = generated[name]
        metrics[name] = {
            'arrival_round': arrivals[name],
            'first_token_round': rounds[0],
            'first_token_wait_rounds': rounds[0] - arrivals[name],
            'first_decode_round': decoded[name][0] if decoded[name] else None,
            'finish_round': rounds[-1],
            'generation_rounds': rounds, 'decode_rounds': decoded[name],
            'max_generation_gap': max((b - a for a, b in zip(rounds, rounds[1:])), default=0),
        }
    bm = scheduler.block_manager
    assert not bm.used_block_ids
    assert len(set(bm.free_block_ids)) == len(bm.blocks)
    assert all(block.ref_count == 0 for block in bm.blocks)
    return {'policy': policy, 'decode_policy': decode_policy, 'leading_tokens': leading_tokens,
            'short_arrival_rounds': short_arrival_rounds,
            'requests': metrics, 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--compare', action='store_true', help='Include round-robin Decode')
    modes.add_argument('--mixed', action='store_true', help='Add six short arrivals to all four policy combinations')
    args = parser.parse_args()
    cases = [run_case(p, length, decode_policy=d, short_arrival_rounds=6 if args.mixed else 0)
             for length in ((8,) if args.mixed else (4, 8, 16))
             for d in (('original', 'round_robin') if args.compare or args.mixed else ('original',))
             for p in ('prefill_first', 'bounded_prefill')]
    sources = ['experiments/unit5_fairness.py', 'nanovllm/engine/scheduler.py',
               'nanovllm/engine/sequence.py', 'nanovllm/engine/block_manager.py',
               'nanovllm/sampling_params.py', 'nanovllm/config.py', 'requirements-cpu.txt']
    report = {
        'upstream_base': 'bb823b3e06983d71485a8e1f23715ebd87d98ef8',
        'python': platform.python_version(), 'platform': platform.system(),
        'config': {'max_num_seqs': 2, 'budget': 8, 'blocks': 16,
                   'block_size': 256, 'max_consecutive_prefills': 2},
        'workload': 'A/B/C arrive before round 0, prompt=1, A/B output=L, C output=2. '
                    'If short_arrival_rounds=N, D1..DN arrive in rounds 2..N+1, prompt=1, output=1.',
        'units': 'Scheduling rounds with synthetic output; no GPU timing.',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in sources},
        'cases': cases,
    }
    filename = ('unit5_mixed.json' if args.mixed else
                'unit5_round_robin.json' if args.compare else 'unit5_fairness.json')
    output = ROOT / 'results' / filename
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for case in cases:
        short_waits = [m['first_token_wait_rounds'] for n, m in case['requests'].items()
                       if n.startswith('D')]
        print(f"policy={case['policy']}, decode={case['decode_policy']}, "
              f"A/B output={case['leading_tokens']}, "
              f"C first_decode={case['requests']['C']['first_decode_round']}, "
              f"C max_gap={case['requests']['C']['max_generation_gap']}, "
              f"A/B finish={case['requests']['A']['finish_round']}/{case['requests']['B']['finish_round']}, "
              f"A first_decode={case['requests']['A']['first_decode_round']}, "
              f"short_max_wait={max(short_waits, default=0)}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
