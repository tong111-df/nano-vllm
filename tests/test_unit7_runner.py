from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import pytest

from experiments.run import ROOT, resolve_config, run_experiment


def example(name='mixed'):
    return json.loads((ROOT / 'experiments' / 'configs' / f'{name}.json').read_text())


def test_configured_comparison_reproduces_mixed_results_and_archives(tmp_path):
    config = example()
    original = deepcopy(config)
    first, report = run_experiment(config, tmp_path)
    second, repeated = run_experiment(config, tmp_path)
    assert config == original
    assert first != second
    assert report['cases'] == repeated['cases']
    assert [c['result']['requests']['C']['first_decode_round']
            for c in report['cases']] == [15, 13, 9, 6]
    for case in report['cases']:
        metrics = case['result']['requests']
        assert len(metrics) == 9
        assert [metrics[f'D{i}']['arrival_round'] for i in range(1, 7)] == list(range(2, 8))
    assert json.loads((first / 'report.json').read_text()) == report
    assert json.loads((first / 'config.json').read_text()) == report['config']
    path = 'nanovllm/engine/scheduler.py'
    assert report['source_sha256'][path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert report['environment']['packages']['numpy']


def test_cli_runs_arrival_config_without_inference_dependencies(tmp_path):
    script = '''
import runpy, sys
sys.argv = ['experiments.run', '--config', 'experiments/configs/arrivals.json',
            '--output-root', sys.argv[1]]
runpy.run_module('experiments.run', run_name='__main__')
assert not {'torch', 'triton', 'transformers', 'flash_attn'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = list(tmp_path.glob('*/report.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert [c['result']['summary']['A_first_decode_round'] for c in report['cases']] == [7, 3]


def test_configured_budget_and_limit_change_real_schedule(tmp_path):
    config = example('arrivals')
    config['parameters']['arrivals_per_round'] = 2
    config['parameters']['budget'] = 1
    _, narrow = run_experiment(config, tmp_path)
    config['parameters']['budget'] = 2
    _, wide = run_experiment(config, tmp_path)
    def batch_size(report):
        return max(len(row['scheduled_tokens']) for row in report['cases'][0]['result']['trace'])
    assert batch_size(narrow) == 1
    assert batch_size(wide) == 2
    config['parameters']['arrivals_per_round'] = 1
    config['strategies'] = [{'policy': 'bounded_prefill', 'max_consecutive_prefills': 1}]
    _, report = run_experiment(config, tmp_path)
    assert report['cases'][0]['result']['summary']['A_first_decode_round'] == 2


def test_short_leaders_do_not_drop_future_arrivals(tmp_path):
    config = example()
    config['parameters']['leading_tokens'] = 1
    _, report = run_experiment(config, tmp_path)
    for case in report['cases']:
        assert len(case['result']['requests']) == 9
        assert case['result']['requests']['D6']['finish_round'] >= 7


@pytest.mark.parametrize('section,key,value,message', [
    (None, 'budegt', 2, 'unknown fields'),
    (None, 'schema_version', True, 'schema_version'),
    ('parameters', 'budegt', 2, 'unknown fields'),
    ('parameters', 'budget', 0, 'budget'),
    ('parameters', 'short_arrival_rounds', -1, 'short_arrival_rounds'),
    ('strategy', 'policy', 'typo', 'policy'),
    ('strategy', 'decode_policy', 'typo', 'decode_policy'),
    ('strategy', 'max_consecutive_prefills', True, 'max_consecutive_prefills'),
])
def test_bad_configuration_is_rejected_before_creating_output(tmp_path, section, key, value, message):
    config = example()
    target = config if section is None else (
        config['strategies'][0] if section == 'strategy' else config[section])
    target[key] = value
    with pytest.raises(ValueError, match=message):
        run_experiment(config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_defaults_are_saved_explicitly():
    config = resolve_config({'schema_version': 1, 'workload': 'mixed',
                             'strategies': [{'policy': 'bounded_prefill'}]})
    assert config['parameters'] == {'leading_tokens': 8, 'short_arrival_rounds': 6, 'budget': 8}
    assert config['strategies'][0]['max_consecutive_prefills'] == 2
    assert config['strategies'][0]['decode_policy'] == 'original'
