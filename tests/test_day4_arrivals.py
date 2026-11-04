import pytest

from experiments.day4_arrivals import run_case


@pytest.mark.parametrize('policy, a_rounds, b_waits', [
    ('prefill_first', [0, 7, 8, 9], [0, 0, 0, 0, 0, 0]),
    ('bounded_prefill', [0, 3, 6, 8], [0, 0, 1, 0, 0, 1]),
])
def test_continuous_arrivals_tradeoff(policy, a_rounds, b_waits):
    result = run_case(policy)
    metrics = result['requests']
    assert metrics['A']['generation_rounds'] == a_rounds
    assert metrics['A']['finish_round'] == a_rounds[-1]
    assert metrics['A']['decode_rounds'] == a_rounds[1:]
    assert metrics['A']['generation_gaps_rounds'] == [
        right - left for left, right in zip(a_rounds, a_rounds[1:])
    ]
    assert [metrics[f'B{i}']['first_token_wait_rounds']
            for i in range(1, 7)] == b_waits
    for i in range(1, 7):
        m = metrics[f'B{i}']
        assert m['arrival_round'] == i
        assert m['finish_round'] == m['first_token_round'] == i + b_waits[i - 1]
    # Both strategies drain the complete finite stream, including late arrivals.
    assert len(metrics) == 7


@pytest.mark.parametrize('arrival_count', [6, 12, 24])
def test_longer_stream_repeated_service_and_backlog(arrival_count):
    baseline = run_case('prefill_first', arrival_count, a_max_tokens=32)
    bounded = run_case('bounded_prefill', arrival_count, a_max_tokens=32)
    old, new = baseline['summary'], bounded['summary']
    assert old['A_first_decode_round'] == arrival_count + 1
    assert new['A_first_decode_round'] == 3
    # Applies to this workload: no capacity pressure and only one Decode waiter.
    assert new['A_max_generation_gap'] <= 3
    assert new['B_max_first_token_wait'] == 1
    assert new['max_waiting_after'] == 1
    assert old['max_waiting_after'] == 0
    assert new['A_finish_round'] < old['A_finish_round']
    for result in (baseline, bounded):
        assert len(result['requests']) == arrival_count + 1
        assert len(result['requests']['A']['generation_rounds']) == 32
        for i in range(1, arrival_count + 1):
            metric = result['requests'][f'B{i}']
            assert metric['arrival_round'] == i
            assert len(metric['generation_rounds']) == 1


@pytest.mark.parametrize('rounds', [6, 12, 24])
def test_heavy_arrivals_accumulate_then_drain(rounds):
    old = run_case('prefill_first', rounds, 32, arrivals_per_round=2)
    new = run_case('bounded_prefill', rounds, 32, arrivals_per_round=2)
    # Over each three arrival rounds, six arrive and only four get Prefill.
    assert new['summary']['waiting_after_last_arrival_round'] == 2 * (rounds // 3)
    assert old['summary']['waiting_after_last_arrival_round'] == 0
    assert new['summary']['B_max_first_token_wait'] > 1
    assert new['summary']['A_first_decode_round'] == 3
    for row in new['trace']:
        if 1 <= row['round'] <= rounds and row['round'] % 3 == 0:
            assert len(row['waiting_after']) == 2 * (row['round'] // 3)
    for result in (old, new):
        assert result['total_arrivals'] == 2 * rounds
        assert len(result['requests']) == 2 * rounds + 1
        assert not result['trace'][-1]['waiting_after']
        assert all(state == 'FINISHED'
                   for state in result['trace'][-1]['states_after'].values())
        for i in range(1, 2 * rounds + 1):
            assert result['requests'][f'B{i}']['arrival_round'] == (i - 1) // 2 + 1
    assert new['summary']['waiting_clear_round_after_arrivals'] > rounds
