"""Record original free-queue order and successful reference events."""
from datetime import datetime, timezone
import json
import platform
from types import SimpleNamespace
from uuid import uuid4

from experiments.run import ROOT, source_manifest
from nanovllm import SamplingParams
from nanovllm.engine.cache_policy import OriginalFreeQueuePolicy
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class RecordingOriginalPolicy(OriginalFreeQueuePolicy):
    def __init__(self):
        self.events = []

    def select_victim(self, free_block_ids):
        block_id = super().select_victim(free_block_ids)
        self.events.append({'event': 'select', 'block': block_id,
                            'candidates': list(free_block_ids)})
        return block_id

    def on_acquire(self, block_id):
        self.events.append({'event': 'acquire', 'block': block_id})

    def on_release(self, block_id):
        self.events.append({'event': 'release', 'block': block_id})


def run_baseline():
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=1, max_num_batched_tokens=1024, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=4,
    ))
    bm = scheduler.block_manager
    policy = RecordingOriginalPolicy()
    bm.eviction_policy = policy
    trace, requests = [], {}

    def record(event):
        trace.append({'event': event, 'free': list(bm.free_block_ids),
                      'used': sorted(bm.used_block_ids),
                      'tables': {name: seq.block_table.copy() for name, seq in requests.items()},
                      'ref_counts': [b.ref_count for b in bm.blocks],
                      'policy_event_count': len(policy.events)})

    def begin(name, prompt, outputs):
        seq = Sequence(prompt, SamplingParams(max_tokens=outputs))
        requests[name] = seq
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        assert prefill and batch == [seq]
        return batch

    record('initial')
    prefix = list(range(256))
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
    record('C_allocated')
    scheduler.postprocess(batch, [1003], True)
    record('C_finished')

    # A read-only cache lookup must not be treated as successful acquisition.
    probe = Sequence(prefix + [301], SamplingParams(max_tokens=1))
    event_count = len(policy.events)
    cached_blocks = bm.can_allocate(probe)
    assert cached_blocks == 1 and len(policy.events) == event_count
    record('D_probe_only')
    batch = begin('D', prefix + [301], 1)
    record('D_prefix_acquired')
    scheduler.postprocess(batch, [1004], True)
    record('D_finished')
    assert scheduler.is_finished()
    return {'config': {'blocks': 4, 'block_size': 256, 'budget': 1024,
                       'max_num_seqs': 1, 'eviction_policy': 'original_free_queue'},
            'trace': trace, 'policy_events': policy.events}


def main():
    case = run_baseline()
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'units': 'CPU block metadata and logical reference events; no GPU timing.',
              'source_sha256': source_manifest(), 'case': case}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / 'unit12' / (stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('event                  free_queue       used_blocks')
    for row in case['trace']:
        print(f"{row['event']:<22} {str(row['free']):<16} {row['used']}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
