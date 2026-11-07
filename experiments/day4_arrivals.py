"""Finite stream of arrivals: compare per-request scheduling rounds."""
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


def run_case(policy, arrival_count=6, a_max_tokens=4, *, arrivals_per_round=1,
             decode_policy="original", budget=2, max_consecutive_prefills=2,
             short_prompt_tokens=1, cache_eviction_policy="original"):
    if arrival_count < 1 or a_max_tokens < 2 or arrivals_per_round < 1:
        raise ValueError('Need at least one arrival and two A output tokens')
    if type(short_prompt_tokens) is not int or short_prompt_tokens < 1:
        raise ValueError('short_prompt_tokens must be a positive integer')
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=16,
        scheduling_policy=policy, max_consecutive_prefills=max_consecutive_prefills,
        decode_scheduling_policy=decode_policy,
        cache_eviction_policy=cache_eviction_policy,
    ))
    requests, names, arrivals = {}, {}, {}
    generated, decoded, finished = {}, {}, {}
    trace = []

    def add(name, round_id, max_tokens, prompt_tokens=1):
        seq = Sequence([10 + round_id + i for i in range(prompt_tokens)],
                       SamplingParams(max_tokens=max_tokens))
        requests[name] = seq
        names[seq.seq_id] = name
        arrivals[name] = round_id
        generated[name], decoded[name] = [], []
        scheduler.add(seq)

    add('A', 0, a_max_tokens)
    total_arrivals = arrival_count * arrivals_per_round
    # Allow fully serialized prompt chunks even at budget=1, plus arrival time.
    round_limit = total_arrivals * short_prompt_tokens + a_max_tokens + arrival_count + 10
    for round_id in range(round_limit):
        # Arrival schedule is fixed independently of the selected policy.
        if 1 <= round_id <= arrival_count:
            for offset in range(arrivals_per_round):
                index = (round_id - 1) * arrivals_per_round + offset + 1
                add(f'B{index}', round_id, 1, short_prompt_tokens)
        before = {n: s.num_completion_tokens for n, s in requests.items()}
        queues = {
            'waiting': [names[s.seq_id] for s in scheduler.waiting],
            'running': [names[s.seq_id] for s in scheduler.running],
        }
        seqs, prefill = scheduler.schedule()
        scheduled = {names[s.seq_id]: s.num_scheduled_tokens for s in seqs}
        # Fixed synthetic output by completion position, independent of round.
        tokens = [1000 + s.num_completion_tokens for s in seqs]
        scheduler.postprocess(seqs, tokens, prefill)
        for seq in seqs:
            name = names[seq.seq_id]
            if not prefill:
                decoded[name].append(round_id)
            if seq.num_completion_tokens > before[name]:
                generated[name].append(round_id)
            if seq.is_finished:
                finished[name] = round_id
        trace.append({
            'round': round_id, 'queues_before': queues,
            'phase': 'prefill' if prefill else 'decode',
            'scheduled_tokens': scheduled,
            'completion_counts_after': {
                n: s.num_completion_tokens for n, s in requests.items()
            },
            'states_after': {n: s.status.name for n, s in requests.items()},
            'waiting_after': [names[s.seq_id] for s in scheduler.waiting],
        })
        if round_id >= arrival_count and scheduler.is_finished():
            break
    else:
        raise RuntimeError(f'Workload did not finish within {round_limit} rounds')

    def gaps(rounds):
        return [right - left for left, right in zip(rounds, rounds[1:])]

    metrics = {}
    for name, seq in requests.items():
        assert seq.is_finished and seq.num_completion_tokens == seq.max_tokens
        metrics[name] = {
            'arrival_round': arrivals[name],
            'first_token_round': generated[name][0],
            'first_token_wait_rounds': generated[name][0] - arrivals[name],
            'finish_round': finished[name],
            'arrival_to_finish_rounds': finished[name] - arrivals[name],
            'generation_rounds': generated[name],
            'generation_gaps_rounds': gaps(generated[name]),
            'decode_rounds': decoded[name],
            'decode_gaps_rounds': gaps(decoded[name]),
        }
    bm = scheduler.block_manager
    assert not bm.used_block_ids
    assert len(set(bm.free_block_ids)) == len(bm.blocks)
    assert all(block.ref_count == 0 for block in bm.blocks)
    waits = [metrics[f'B{i}']['first_token_wait_rounds']
             for i in range(1, total_arrivals + 1)]
    summary = {
        'A_first_decode_round': decoded['A'][0],
        'A_max_generation_gap': max(metrics['A']['generation_gaps_rounds']),
        'A_finish_round': finished['A'],
        'B_max_first_token_wait': max(waits),
        'B_mean_first_token_wait': sum(waits) / len(waits),
        'max_waiting_before': max(len(r['queues_before']['waiting']) for r in trace),
        'max_waiting_after': max(len(r['waiting_after']) for r in trace),
        'waiting_after_last_arrival_round': len(trace[arrival_count]['waiting_after']),
        'waiting_clear_round_after_arrivals': next(
            r['round'] for r in trace
            if r['round'] >= arrival_count and not r['waiting_after']
        ),
        'all_finished_round': trace[-1]['round'],
    }
    return {'policy': policy, 'arrival_count': arrival_count,
            'short_prompt_tokens': short_prompt_tokens,
            'arrivals_per_round': arrivals_per_round, 'total_arrivals': total_arrivals,
            'a_max_tokens': a_max_tokens, 'summary': summary,
            'requests': metrics, 'trace': trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--sweep', action='store_true',
                        help='Compare 6/12/24 arrivals with A output length fixed at 32')
    modes.add_argument('--overload', action='store_true',
                       help='Compare one/two arrivals per round over 6/12/24 rounds')
    args = parser.parse_args()
    expanded = args.sweep or args.overload
    cases = [run_case(p, n, 32 if expanded else 4, arrivals_per_round=rate)
             for n in ((6, 12, 24) if expanded else (6,))
             for rate in ((1, 2) if args.overload else (1,))
             for p in ('prefill_first', 'bounded_prefill')]
    sources = ['experiments/day4_arrivals.py', 'nanovllm/engine/scheduler.py',
               'nanovllm/engine/block_manager.py', 'nanovllm/engine/sequence.py',
               'nanovllm/sampling_params.py', 'requirements-cpu.txt']
    report = {
        'upstream_base': 'bb823b3e06983d71485a8e1f23715ebd87d98ef8',
        'python': platform.python_version(), 'platform': platform.system(),
        'config': {'budget': 2, 'blocks': 16, 'block_size': 256,
                   'max_num_seqs': 4, 'max_consecutive_prefills': 2},
        'workload': 'A: round 0, prompt=1. Short requests: prompt=1, output=1. '
                    'arrival_count is the number of arrival rounds (1..N); '
                    'rate, total_arrivals and A output length are recorded per case.',
        'units': 'Scheduling rounds, not wall time. Same-round service has wait=0.',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in sources},
        'cases': cases,
    }
    filename = ('day4_overload.json' if args.overload else
                'day4_sweep.json' if args.sweep else 'day4_arrivals.json')
    output = ROOT / 'results' / filename
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for case in cases:
        if expanded:
            print(f"arrival_rounds={case['arrival_count']}, rate={case['arrivals_per_round']}, "
                  f"policy={case['policy']}, "
                  f"summary={case['summary']}")
            continue
        print(case['policy'])
        for name, metric in case['requests'].items():
            print(f"  {name}: arrival={metric['arrival_round']}, "
                  f"first={metric['first_token_round']}, "
                  f"wait={metric['first_token_wait_rounds']}, "
                  f"finish={metric['finish_round']}, "
                  f"generation_rounds={metric['generation_rounds']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
