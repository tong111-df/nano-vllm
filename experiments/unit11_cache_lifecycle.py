"""Trace shared prefix references, release, reuse and overwrite on real CPU code."""
from datetime import datetime, timezone
import json
import platform
from types import SimpleNamespace
from uuid import uuid4

from experiments.run import ROOT, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def run_lifecycle():
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=3,
    ))
    bm = scheduler.block_manager
    prefix = list(range(256))
    prefix_hash = bm.compute_hash(prefix)
    requests, events = {}, []

    def record(event):
        events.append({
            'event': event,
            'prefix_index_block': bm.hash_to_block_id.get(prefix_hash),
            'free_blocks': list(bm.free_block_ids),
            'used_blocks': sorted(bm.used_block_ids),
            'blocks': [{'id': b.block_id, 'ref_count': b.ref_count,
                        'hash': b.hash, 'metadata_token_count': len(b.token_ids)} for b in bm.blocks],
            'requests': {name: {'block_table': seq.block_table.copy(),
                                'cached_tokens': seq.num_cached_tokens,
                                'completion_tokens': seq.num_completion_tokens,
                                'status': seq.status.name} for name, seq in requests.items()},
        })

    def begin(name, prompt, max_tokens):
        seq = Sequence(prompt, SamplingParams(max_tokens=max_tokens))
        requests[name] = seq
        scheduler.add(seq)
        seqs, prefill = scheduler.schedule()
        assert prefill and seqs == [seq]
        return seqs

    def finish_prefill(seqs, token, event):
        scheduler.postprocess(seqs, [token], True)
        record(event)

    record('initial')
    finish_prefill(begin('A', prefix + [300], 2), 1000, 'A_prefill_done')
    prefix_block = requests['A'].block_table[0]

    batch = begin('B', prefix + [301], 2)
    record('B_prefix_hit')
    finish_prefill(batch, 1001, 'B_prefill_done')
    for name in ('A', 'B'):
        seqs, prefill = scheduler.schedule()
        assert not prefill and seqs == [requests[name]]
        scheduler.postprocess(seqs, [2000], prefill)
        record(f'{name}_finished')

    batch = begin('C', prefix + [302], 1)
    record('C_free_prefix_hit')
    finish_prefill(batch, 1002, 'C_finished')

    # D needs all three blocks, so it must eventually recycle the old prefix
    # regardless of its position in the original free queue.
    batch = begin('D', list(range(10000, 10513)), 1)
    record('D_overwrites_prefix')
    finish_prefill(batch, 1003, 'D_finished')

    batch = begin('E', prefix + [303], 1)
    record('E_old_prefix_miss')
    finish_prefill(batch, 1004, 'E_finished')
    return {'config': {'blocks': 3, 'block_size': 256, 'budget': 1024, 'max_num_seqs': 1},
            'prefix_block_id': prefix_block, 'prefix_hash': prefix_hash, 'events': events}


def main():
    case = run_lifecycle()
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'units': 'CPU control-plane metadata; synthetic model outputs; no GPU KV tensors.',
              'source_sha256': source_manifest(), 'case': case}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / 'unit11' / (stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    shared = case['prefix_block_id']
    print(f'Tracking physical block {shared}; prefix_index refers to the ORIGINAL prefix.')
    print('event                    block_ref free_blocks prefix_index')
    for event in case['events']:
        print(f"{event['event']:<24} {event['blocks'][shared]['ref_count']:>10} "
              f"{str(event['free_blocks']):<11} {event['prefix_index_block']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
