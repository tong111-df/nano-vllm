from copy import deepcopy
import json

import pytest

from experiments.unit15_cache_workloads import DEFAULT_CONFIG, run_case, run_suite
from experiments.unit14_cache_interactions import make_scheduler
from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence


def example():
    return json.loads(DEFAULT_CONFIG.read_text())


def test_fixed_load_suite_accounts_tokens_and_preserves_equal_policy_results(tmp_path):
    raw = example()
    before = deepcopy(raw)
    output, report = run_suite(raw, repeats=2, warmup=1, output_root=tmp_path)
    assert raw == before
    assert json.loads((output / 'report.json').read_text()) == report
    assert len(json.loads((output / 'summary.json').read_text())) == 12
    assert len(report['cases']) == 12
    expected_hits = {(4, 'hot'): 2816, (6, 'hot'): 2816,
                     (4, 'scan'): 0, (6, 'scan'): 0,
                     (4, 'mixed'): 0, (6, 'mixed'): 768}
    paired = {}
    for case in report['cases']:
        blocks, workload, policy = case['blocks'], case['workload'], case['policy']
        metrics = case['metrics']
        assert metrics['cached_tokens'] == expected_hits[blocks, workload]
        assert metrics['computed_prompt_tokens'] == 3084 - metrics['cached_tokens']
        assert metrics['total_prompt_tokens'] == 3084 and metrics['request_count'] == 12
        assert metrics['acquire_calls'] == metrics['release_calls'] == 24
        assert metrics['select_calls'] == 24 - metrics['requests_with_prefix_hit']
        if policy == 'original':
            assert metrics['selection_candidate_entries'] == metrics['select_calls']
            assert metrics['recency_metadata_entries'] == 0
        else:
            assert metrics['selection_candidate_entries'] > metrics['select_calls']
            assert metrics['recency_metadata_entries'] == blocks
        assert [row['prefix_key'] for row in case['trace']] == raw['workloads'][workload]
        for row in case['trace']:
            assert row['cached_tokens'] + row['computed_prompt_tokens'] == 257
            assert row['cached_tokens'] in (0, 256)
            assert len(row['free_after']) == len(set(row['free_after'])) == blocks
        for timing in case['timing'].values():
            assert len(timing['samples_ns']) == 2
            assert 0 <= timing['min_ns'] <= timing['median_ns'] <= timing['max_ns']
        key = blocks, workload
        if key in paired:
            assert paired[key] == (metrics['cached_tokens'], metrics['computed_prompt_tokens'])
        paired[key] = metrics['cached_tokens'], metrics['computed_prompt_tokens']
    # Hot: one cold 257-token request plus eleven 1-token tails.
    assert expected_hits[4, 'hot'] == 11 * 256
    assert 3084 - expected_hits[4, 'hot'] == 257 + 11


@pytest.mark.parametrize('policy', ['original', 'lru'])
def test_timing_wrapper_does_not_change_cache_behavior(policy):
    keys = [0, 1, 0, 2, 0]
    measured = run_case(keys, policy, 4)
    scheduler = make_scheduler(policy, blocks=4)
    expected = []
    for index, key in enumerate(keys):
        prompt = list(range(key * 1000, key * 1000 + 256)) + [1000000 + index]
        seq = Sequence(prompt, SamplingParams(max_tokens=1))
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        cached, computed = seq.num_cached_tokens, seq.num_scheduled_tokens
        scheduler.postprocess(batch, [2000000 + index], prefill)
        expected.append((cached, computed, list(scheduler.block_manager.free_block_ids)))
    assert [(row['cached_tokens'], row['computed_prompt_tokens'], row['free_after'])
            for row in measured['trace']] == expected


@pytest.mark.parametrize('field,value', [
    ('block_counts', [1]), ('policies', ['typo']),
    ('workloads', {'hot': []}), ('workloads', {'hot': [True]}),
])
def test_invalid_workload_config_fails_without_creating_report(tmp_path, field, value):
    raw = example()
    raw[field] = value
    with pytest.raises(ValueError):
        run_suite(raw, output_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
