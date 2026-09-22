from __future__ import annotations

from collections.abc import Iterable

from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

DEFAULT_RULES = (
    "shortest_processing_time",
    "most_work_remaining",
    "most_operations_remaining",
    "first_come_first_served",
)


def solve_with_rule(rule: str, instance_name: str = "ft06") -> int:
    """Solves a benchmark instance with one dispatching rule and returns makespan."""
    instance = load_benchmark_instance(instance_name)
    schedule = DispatchingRuleSolver(rule)(instance)
    return int(schedule.makespan())


def compare_dispatching_rules(
    rules: Iterable[str] = DEFAULT_RULES,
    instance_name: str = "ft06",
) -> dict[str, int]:
    """Returns deterministic makespans for a set of dispatching heuristics."""
    return {rule: solve_with_rule(rule, instance_name) for rule in rules}
