"""Compare scheduling rounds, not GPU latency or throughput."""
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_case(policy):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=2, eos=99,
        kvcache_block_size=256, num_kvcache_blocks=4,
        scheduling_policy=policy, max_consecutive_prefills=2,
    ))
    a = Sequence([10], SamplingParams(max_tokens=3))
    b = Sequence([30, 31, 32, 33, 34], SamplingParams(max_tokens=1))
    requests = {'A': a, 'B': b}
    names = {seq.seq_id: name for name, seq in requests.items()}
    trace = []

    def advance(round_id):
        before = {
            'waiting': [names[s.seq_id] for s in scheduler.waiting],
            'running': [names[s.seq_id] for s in scheduler.running],
            'consecutive_prefills': scheduler.consecutive_prefills,
        }
        seqs, prefill = scheduler.schedule()
        scheduled = {names[s.seq_id]: s.num_scheduled_tokens for s in seqs}
        scheduler.postprocess(seqs, [1000 + round_id] * len(seqs), prefill)
        trace.append({
            'round': round_id, 'before': before,
            'phase': 'prefill' if prefill else 'decode',
            'scheduled_tokens': scheduled,
            'consecutive_prefills_after': scheduler.consecutive_prefills,
            'completion_counts_after': {
                n: s.num_completion_tokens for n, s in requests.items()
            },
            'states_after': {n: s.status.name for n, s in requests.items()},
        })

    scheduler.add(a)
    advance(0)  # A's initial Prefill is setup; B arrives before round 1.
    scheduler.add(b)
    for round_id in range(1, 21):
        advance(round_id)
        if scheduler.is_finished():
            break
    else:
        raise RuntimeError('Requests did not finish within 20 rounds')

    first_decode = next(r['round'] for r in trace
                        if r['phase'] == 'decode' and 'A' in r['scheduled_tokens'])
    b_first = next(r['round'] for r in trace
                   if r['completion_counts_after']['B'] > 0)
    finished = {
        n: next(r['round'] for r in trace if r['states_after'][n] == 'FINISHED')
        for n in requests
    }
    expected = (4, 3) if policy == 'prefill_first' else (3, 4)
    assert (first_decode, b_first) == expected
    assert a.num_completion_tokens == 3 and b.num_completion_tokens == 1
    bm = scheduler.block_manager
    assert not bm.used_block_ids
    assert len(set(bm.free_block_ids)) == len(bm.blocks)
    assert all(block.ref_count == 0 for block in bm.blocks)
    return {
        'policy': policy, 'A_first_decode_round': first_decode,
        'B_first_token_round': b_first, 'finish_rounds': finished,
        'trace': trace,
    }


def main():
    cases = [run_case(policy) for policy in ('prefill_first', 'bounded_prefill')]
    sources = ['nanovllm/config.py', 'nanovllm/engine/scheduler.py',
               'nanovllm/engine/block_manager.py', 'nanovllm/engine/sequence.py',
               'experiments/day3_compare.py', 'requirements-cpu.txt']
    report = {
        'upstream_base': 'bb823b3e06983d71485a8e1f23715ebd87d98ef8',
        'python': platform.python_version(), 'platform': platform.system(),
        'config': {'prefill_budget': 2, 'block_size': 256, 'blocks': 4,
                   'max_num_seqs': 4, 'max_consecutive_prefills': 2},
        'note': 'Synthetic model outputs; round 0 is setup. No GPU timing measured.',
        'source_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        'cases': cases,
    }
    output = ROOT / 'results' / 'day3_compare.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for case in cases:
        print(f"{case['policy']}: A_first_decode_round={case['A_first_decode_round']}, "
              f"B_first_token_round={case['B_first_token_round']}, "
              f"finish_rounds={case['finish_rounds']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
