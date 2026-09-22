import pytest

from job_shop_demo.baselines import compare_dispatching_rules, solve_with_rule


def test_dispatching_rules_return_positive_makespans():
    results = compare_dispatching_rules(
        rules=("shortest_processing_time", "most_work_remaining")
    )
    assert set(results) == {"shortest_processing_time", "most_work_remaining"}
    assert all(isinstance(value, int) and value > 0 for value in results.values())


def test_baseline_is_deterministic():
    first = solve_with_rule("most_work_remaining")
    second = solve_with_rule("most_work_remaining")
    assert first == second


def test_invalid_dispatching_rule_is_rejected():
    with pytest.raises(Exception):
        solve_with_rule("definitely_not_a_rule")
