"""Compare release-queue recycling with acquisition-based LRU on a fixed load."""
from datetime import datetime, timezone
import json
import platform
from types import SimpleNamespace
from uuid import uuid4

from experiments.run import ROOT, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def run_case(cache_policy):
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=4,
        cache_eviction_policy=cache_policy,
    ))
    bm = scheduler.block_manager
    prefix = list(range(256))
    prefix_hash = bm.compute_hash(prefix)
    requests, trace = {}, []

    def record(event):
        trace.append({'event': event, 'free': list(bm.free_block_ids),
                      'used': sorted(bm.used_block_ids),
                      'ref_counts': [b.ref_count for b in bm.blocks],
                      'tables': {n: s.block_table.copy() for n, s in requests.items()},
                      'A_prefix_index': bm.hash_to_block_id.get(prefix_hash),
                      'last_access': dict(getattr(bm.eviction_policy, 'last_access', {}))})

    def begin(name, prompt, outputs):
        seq = Sequence(prompt, SamplingParams(max_tokens=outputs))
        requests[name] = seq
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        assert prefill and batch == [seq]
        return batch

    batch = begin('A', prefix + [300], 2)
    scheduler.postprocess(batch, [1000], True)
    record('A_prefill_done')
    batch = begin('B', list(range(1000, 1257)), 1)
    scheduler.postprocess(batch, [1001], True)
    record('B_finished')
    batch, prefill = scheduler.schedule()
    assert not prefill and batch == [requests['A']]
    scheduler.postprocess(batch, [1002], False)
    record('A_finished')

    batch = begin('C', [2000], 1)
    recycled = requests['C'].block_table[0]
    record('C_allocated')
    scheduler.postprocess(batch, [1003], True)
    record('C_finished')

    batch = begin('D', prefix + [301], 1)
    cached = requests['D'].num_cached_tokens
    scheduled = requests['D'].num_scheduled_tokens
    record('D_scheduled')
    scheduler.postprocess(batch, [1004], True)
    record('D_finished')
    assert scheduler.is_finished() and not bm.used_block_ids
    assert all(block.ref_count == 0 for block in bm.blocks)
    return {'cache_policy': cache_policy, 'C_recycled_block': recycled,
            'D_cached_tokens': cached, 'D_computed_prompt_tokens': scheduled,
            'trace': trace}


def main():
    cases = [run_case(name) for name in ('original', 'lru')]
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'config': {'block_size': 256, 'blocks': 4, 'budget': 1024, 'max_num_seqs': 1},
        'access_definition': 'Successful block reference acquisition; probes, releases and per-step attention reads do not refresh it.',
        'units': 'CPU metadata and prompt token counts; synthetic output; no GPU timing.',
        'source_sha256': source_manifest(), 'cases': cases,
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / 'unit13' / (stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('policy   C_recycled_block D_cached_tokens D_computed_prompt_tokens')
    for case in cases:
        print(f"{case['cache_policy']:<8} {case['C_recycled_block']:>16} "
              f"{case['D_cached_tokens']:>15} {case['D_computed_prompt_tokens']:>24}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
