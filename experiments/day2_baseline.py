"""Record real scheduler decisions with synthetic model outputs; no timing claims."""
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


ROOT = Path(__file__).resolve().parents[1]


def run_case(budget, arrivals=False):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=budget, eos=99,
        kvcache_block_size=256, num_kvcache_blocks=4,
    ))
    requests = {}
    names = {}
    rows = []

    def add(name, prompt, max_tokens):
        seq = Sequence(prompt, SamplingParams(max_tokens=max_tokens))
        requests[name] = seq
        names[seq.seq_id] = name
        scheduler.add(seq)
        return seq

    def advance(round_id):
        waiting_before = [names[s.seq_id] for s in scheduler.waiting]
        running_before = [names[s.seq_id] for s in scheduler.running]
        seqs, prefill = scheduler.schedule()
        # Capture now: postprocess resets num_scheduled_tokens to zero.
        scheduled = {names[s.seq_id]: s.num_scheduled_tokens for s in seqs}
        scheduler.postprocess(seqs, [1000 + round_id] * len(seqs), prefill)
        rows.append({
            'round': round_id,
            'phase': 'prefill' if prefill else 'decode',
            'waiting_before': waiting_before, 'running_before': running_before,
            'scheduled_tokens': scheduled,
            'states_after': {n: s.status.name for n, s in requests.items()},
            'completion_counts_after': {
                n: s.num_completion_tokens for n, s in requests.items()
            },
        })

    a = add('A', [10], 3)
    advance(0)  # Setup is recorded but excluded from A's blocked-round count.
    if not arrivals:
        add('B', [30, 31, 32, 33, 34], 1)

    for round_id in range(1, 21):
        if arrivals and round_id <= 3:
            add(f'B{round_id}', [30 + round_id], 1)
        advance(round_id)
        if scheduler.is_finished():
            break
    else:
        raise RuntimeError('Workload did not finish within 20 rounds')

    first_decode = next(row['round'] for row in rows
                        if row['phase'] == 'decode'
                        and 'A' in row['scheduled_tokens'])
    blocked = sum(1 for row in rows if 0 < row['round'] < first_decode
                  and row['completion_counts_after']['A'] == 1)
    expected = 3 if arrivals else (5 + budget - 1) // budget
    assert blocked == expected
    assert a.num_completion_tokens == 3
    bm = scheduler.block_manager
    assert not bm.used_block_ids
    assert len(set(bm.free_block_ids)) == len(bm.blocks)
    assert all(b.ref_count == 0 for b in bm.blocks)
    return {
        'case': 'three_new_arrivals' if arrivals else 'long_prefill',
        'budget': budget, 'A_blocked_rounds': blocked,
        'A_first_decode_round': first_decode, 'trace': rows,
    }


def main():
    cases = [run_case(budget) for budget in (1, 2, 5)]
    cases.append(run_case(2, arrivals=True))
    sources = ['nanovllm/engine/scheduler.py', 'nanovllm/engine/sequence.py',
               'nanovllm/engine/block_manager.py', 'requirements-cpu.txt',
               'experiments/day2_baseline.py']
    report = {
        'upstream_base': 'bb823b3e06983d71485a8e1f23715ebd87d98ef8',
        'python': platform.python_version(), 'platform': platform.system(),
        'note': 'Synthetic token IDs; real control-plane logic; rounds are not latency.',
        'source_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        'cases': cases,
    }
    output = ROOT / 'results' / 'day2_baseline.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for case in cases:
        print(f"{case['case']}: budget={case['budget']}, "
              f"A_blocked_rounds={case['A_blocked_rounds']}, "
              f"A_first_decode_round={case['A_first_decode_round']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
