import csv
import json

import pytest

from experiments.day4_arrivals import run_case
from experiments.run import resolve_config
from experiments.unit8_sweep import DEFAULT_CONFIG, run_sweep


def test_long_arrival_prompt_samples_only_after_all_chunks():
    result = run_case('prefill_first', arrival_count=1, a_max_tokens=2,
                      budget=2, short_prompt_tokens=5)
    assert result['requests']['B1']['generation_rounds'] == [3]
    assert [row['scheduled_tokens']['B1'] for row in result['trace'][1:4]] == [2, 2, 1]
    assert result['trace'][1]['completion_counts_after']['B1'] == 0
    assert result['trace'][2]['completion_counts_after']['B1'] == 0
    assert result['trace'][3]['completion_counts_after']['B1'] == 1


def test_sweep_preserves_workload_and_exposes_both_benefit_and_cost(tmp_path):
    raw = json.loads(DEFAULT_CONFIG.read_text())
    output, rows = run_sweep(raw, output_root=tmp_path)
    assert len(rows) == 12
    assert json.loads((output / 'summary.json').read_text())['rows'] == rows
    with (output / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 12
    for row in rows:
        report = json.loads((output / row['report']).read_text())
        params = report['config']['parameters']
        assert params == raw['parameters'] | {'budget': row['budget']}
        case = report['cases'][row['case_index']]['result']
        assert case['summary'] == {key: row[key] for key in case['summary']}
        assert len(case['requests']['A']['generation_rounds']) == 8
        for i in range(1, 7):
            name = f'B{i}'
            assert case['requests'][name]['arrival_round'] == i
            computed = 0
            for turn in case['trace']:
                computed += turn['scheduled_tokens'].get(name, 0)
                if name in turn['completion_counts_after']:
                    # These one-token outputs appear only after all prompt work.
                    assert turn['completion_counts_after'][name] == int(computed == 5)
            assert computed == 5
        for turn in case['trace']:
            if turn['phase'] == 'prefill':
                assert sum(turn['scheduled_tokens'].values()) <= row['budget']
        assert all(state == 'FINISHED' for state in case['trace'][-1]['states_after'].values())

    def select(budget, policy, limit=2):
        return next(row for row in rows if row['budget'] == budget
                    and row['policy'] == policy and row['max_consecutive_prefills'] == limit)

    for budget in (1, 2, 5):
        baseline = select(budget, 'prefill_first')
        # Six prompts, each requiring ceil(5 / budget) Prefill rounds.
        assert baseline['A_first_decode_round'] == 6 * ((5 + budget - 1) // budget) + 1
        for limit in (1, 2, 3):
            bounded = select(budget, 'bounded_prefill', limit)
            assert bounded['A_first_decode_round'] == limit + 1
            assert bounded['B_max_first_token_wait'] > baseline['B_max_first_token_wait']
            assert bounded['all_finished_round'] == baseline['all_finished_round']
    assert select(2, 'bounded_prefill', 1)['A_finish_round'] < select(2, 'prefill_first')['A_finish_round']
    # First Decode improves at budget=5 while completion does not.
    assert select(5, 'bounded_prefill', 1)['A_finish_round'] == select(5, 'prefill_first')['A_finish_round']


@pytest.mark.parametrize('value', [0, -1, True])
def test_invalid_arrival_prompt_length_rejected(value):
    raw = json.loads(DEFAULT_CONFIG.read_text())
    raw['parameters']['short_prompt_tokens'] = value
    with pytest.raises(ValueError, match='short_prompt_tokens'):
        resolve_config(raw)
