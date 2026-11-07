import pytest

from experiments.unit5_fairness import run_case


@pytest.mark.parametrize('policy', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('length', [4, 8, 16])
def test_decode_queue_bias_baseline(policy, length):
    result = run_case(policy, length)
    trace = result['trace']
    # Prefill admission can create more RUNNING requests than the batch limit.
    assert trace[0]['selected'] == ['A', 'B']
    assert trace[1]['selected'] == ['C']
    assert trace[1]['running_after'] == ['A', 'B', 'C']
    # Existing behavior: the first two retain priority until both finish.
    for row in trace[2:length + 1]:
        assert row['phase'] == 'decode'
        assert row['selected'] == ['A', 'B']
        assert row['completion_counts_after']['C'] == 1
    c = result['requests']['C']
    assert c['generation_rounds'] == [1, length + 1]
    assert c['max_generation_gap'] == length
    assert c['finish_round'] == length + 1
    for name in ('A', 'B'):
        assert len(result['requests'][name]['generation_rounds']) == length


@pytest.mark.parametrize('policy', ['prefill_first', 'bounded_prefill'])
@pytest.mark.parametrize('length', [4, 8, 16])
def test_round_robin_serves_tail_before_leaders_finish(policy, length):
    result = run_case(policy, length, decode_policy='round_robin')
    trace = result['trace']
    assert trace[2]['selected'] == ['A', 'B']
    assert trace[2]['running_after'] == ['C', 'A', 'B']
    assert trace[3]['selected'] == ['C', 'A']
    assert result['requests']['C']['generation_rounds'] == [1, 3]
    assert result['requests']['C']['max_generation_gap'] == 2
    assert result['requests']['A']['finish_round'] == length
    assert result['requests']['B']['finish_round'] == length + 1
    assert len(result['requests']['A']['generation_rounds']) == length
    assert len(result['requests']['B']['generation_rounds']) == length


@pytest.mark.parametrize('policy, decode_policy, a_first, c_first, short_wait', [
    ('prefill_first', 'original', 8, 15, 0),
    ('bounded_prefill', 'original', 3, 13, 1),
    ('prefill_first', 'round_robin', 8, 9, 0),
    ('bounded_prefill', 'round_robin', 3, 6, 1),
])
def test_mixed_arrivals_and_decode_order(policy, decode_policy, a_first, c_first, short_wait):
    result = run_case(policy, 8, decode_policy=decode_policy, short_arrival_rounds=6)
    metrics = result['requests']
    assert metrics['A']['first_decode_round'] == a_first
    assert metrics['C']['first_decode_round'] == c_first
    assert metrics['C']['generation_rounds'] == [1, c_first]
    assert len(metrics) == 9
    assert len(metrics['A']['generation_rounds']) == 8
    assert len(metrics['B']['generation_rounds']) == 8
    assert max(metrics[f'D{i}']['first_token_wait_rounds'] for i in range(1, 7)) == short_wait
    for i in range(1, 7):
        m = metrics[f'D{i}']
        assert m['arrival_round'] == i + 1
        assert m['first_decode_round'] is None
        assert len(m['generation_rounds']) == 1
        assert m['finish_round'] == m['first_token_round']
    for row in result['trace']:
        assert len(row['selected']) == len(set(row['selected'])) <= 2
    if policy == 'bounded_prefill' and decode_policy == 'round_robin':
        assert result['trace'][3]['selected'] == ['A', 'B']
        assert result['trace'][4]['selected'] == ['D2', 'D3']
        assert result['trace'][6]['selected'] == ['C', 'A']
