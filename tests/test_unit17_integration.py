import json

import pytest

from experiments.run import ROOT, resolve_config, run_experiment
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.cache_policy import LRUCachePolicy, OriginalFreeQueuePolicy


@pytest.mark.parametrize('workload', ['mixed', 'arrivals'])
def test_unified_runner_passes_all_eight_policy_combinations_to_scheduler(tmp_path, monkeypatch, workload):
    config = json.loads((ROOT / 'experiments/configs/integrated.json').read_text())
    config['workload'] = workload
    if workload == 'arrivals':
        config['parameters'] = {'arrival_count': 6, 'a_max_tokens': 4}
    seen = []
    original = Scheduler.__init__
    def observe(self, cfg):
        original(self, cfg)
        seen.append((self.scheduling_policy, self.decode_scheduling_policy,
                     self.cache_eviction_policy, type(self.block_manager.eviction_policy)))
    monkeypatch.setattr(Scheduler, '__init__', observe)
    output, report = run_experiment(config, tmp_path)
    assert len(seen) == len(set(seen)) == 8
    for case, actual in zip(report['cases'], seen):
        strategy = case['strategy']
        cache = strategy['cache_eviction_policy']
        assert actual == (strategy['policy'], strategy['decode_policy'], cache,
                          LRUCachePolicy if cache == 'lru' else OriginalFreeQueuePolicy)
    assert json.loads((output / 'report.json').read_text()) == report
    if workload == 'mixed':
        assert [case['result']['requests']['C']['first_decode_round']
                for case in report['cases']] == [15, 13, 9, 6] * 2
    else:
        assert [case['result']['summary']['A_first_decode_round']
                for case in report['cases']] == [7, 3, 7, 3] * 2


def test_legacy_config_defaults_to_original_cache_and_rejects_typo():
    raw = {'schema_version': 1, 'workload': 'mixed', 'strategies': [{'policy': 'prefill_first'}]}
    assert resolve_config(raw)['strategies'][0]['cache_eviction_policy'] == 'original'
    raw['strategies'][0]['cache_eviction_policy'] = 'typo'
    with pytest.raises(ValueError, match='cache_eviction_policy'):
        resolve_config(raw)
