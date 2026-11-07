"""Controlled shared-prefix preemption using real Scheduler and BlockManager."""
from datetime import datetime, timezone
import json
import platform
from types import SimpleNamespace
from uuid import uuid4

from experiments.run import ROOT, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def make_scheduler(cache_policy, *, blocks=3, max_num_seqs=1,
                   phase='prefill_first', decode='original'):
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks,
        cache_eviction_policy=cache_policy,
        scheduling_policy=phase, decode_scheduling_policy=decode,
    ))


def run_shared_preempt(cache_policy, *, phase='prefill_first', decode='original'):
    scheduler = make_scheduler(cache_policy, phase=phase, decode=decode)
    bm = scheduler.block_manager
    prefix = list(range(256))
    a = Sequence(prefix + [300], SamplingParams(max_tokens=3))
    b = Sequence(prefix + [301], SamplingParams(max_tokens=2))
    requests, trace = {'A': a, 'B': b}, []

    def record(event):
        trace.append({'event': event, 'free': list(bm.free_block_ids),
                      'used': sorted(bm.used_block_ids),
                      'ref_counts': [block.ref_count for block in bm.blocks],
                      'requests': {name: {'status': seq.status.name,
                                          'block_table': seq.block_table.copy(),
                                          'cached_tokens': seq.num_cached_tokens,
                                          'scheduled_tokens': seq.num_scheduled_tokens,
                                          'completion_ids': seq.completion_token_ids.copy()}
                                   for name, seq in requests.items()}})

    for seq, name in ((a, 'A'), (b, 'B')):
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        assert batch == [seq] and prefill
        scheduler.postprocess(batch, [1000], prefill)
        record(f'{name}_prefill_done')
    shared = a.block_table[0]
    saved_ids = a.token_ids.copy()
    # A controlled trigger, matching schedule()'s pop-before-preempt contract.
    # Automatic shortage-triggered preemption is tested separately.
    scheduler.running.remove(a)
    scheduler.preempt(a)
    record('A_preempted_B_holds_prefix')
    batch, prefill = scheduler.schedule()
    assert batch == [a] and prefill and a.token_ids == saved_ids
    cached, recomputed = a.num_cached_tokens, a.num_scheduled_tokens
    record('A_resume_scheduled')
    scheduler.postprocess(batch, [1001], prefill)
    record('A_resumed_prefill_done')
    for _ in range(8):
        if scheduler.is_finished():
            break
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [1000 + seq.num_completion_tokens for seq in batch], prefill)
        record('drain_step')
    assert scheduler.is_finished() and not bm.used_block_ids
    return {'cache_policy': cache_policy, 'phase_policy': phase, 'decode_policy': decode,
            'shared_block': shared, 'resume_cached_tokens': cached,
            'resume_computed_tokens': recomputed, 'A_final_completion_ids': a.completion_token_ids,
            'trace': trace}


def main():
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'config': {'block_size': 256, 'blocks': 3, 'budget': 1024, 'max_num_seqs': 1},
        'trigger': 'Controlled manual preemption of A while B still references the prefix.',
        'units': 'CPU metadata, synthetic model output; not GPU KV verification.',
        'source_sha256': source_manifest(),
        'cases': [run_shared_preempt(name) for name in ('original', 'lru')],
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / 'unit14' / (stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('policy   event                            A_cached shared_ref A_generated')
    for case in report['cases']:
        for row in case['trace']:
            a = row['requests']['A']
            print(f"{case['cache_policy']:<8} {row['event']:<32} {a['cached_tokens']:>8} "
                  f"{row['ref_counts'][case['shared_block']]:>10} {len(a['completion_ids']):>11}")
        print(f"  resume: cached={case['resume_cached_tokens']}, "
              f"computed={case['resume_computed_tokens']}, "
              f"final_outputs={case['A_final_completion_ids']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
