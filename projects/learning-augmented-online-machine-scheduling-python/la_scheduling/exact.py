from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .schedule import true_lpt


@dataclass(frozen=True)
class ExactResult:
    makespan: int
    machine_loads: tuple
    status: str
    explored_nodes: int


def exact_makespan_branch_and_bound(actual, n_machines: int) -> ExactResult:
    """Exact P||Cmax solver for small/medium integer instances."""
    p = np.asarray(actual, dtype=np.int64)
    if np.any(p <= 0) or n_machines < 1:
        raise ValueError("invalid instance")

    jobs = np.sort(p)[::-1]
    incumbent_result = true_lpt(p, n_machines)
    incumbent = int(incumbent_result.makespan)
    best_loads = tuple(int(x) for x in incumbent_result.machine_loads)
    explored = 0
    loads = np.zeros(n_machines, dtype=np.int64)
    remaining_sum = np.cumsum(jobs[::-1])[::-1]

    def recurse(i: int):
        nonlocal incumbent, best_loads, explored
        explored += 1
        if i == len(jobs):
            value = int(loads.max(initial=0))
            if value < incumbent:
                incumbent = value
                best_loads = tuple(int(x) for x in loads)
            return

        current_max = int(loads.max(initial=0))
        total_future = int(loads.sum() + remaining_sum[i])
        lower = max(
            current_max,
            int(np.ceil(total_future / n_machines)),
            int(jobs[i]),
        )
        if lower >= incumbent:
            return

        job = int(jobs[i])
        seen_loads = set()
        for m in sorted(range(n_machines), key=lambda k: (loads[k], k)):
            old = int(loads[m])
            if old in seen_loads:
                continue
            seen_loads.add(old)
            new = old + job
            if new >= incumbent:
                continue
            loads[m] = new
            recurse(i+1)
            loads[m] = old

    recurse(0)
    return ExactResult(
        makespan=incumbent,
        machine_loads=best_loads,
        status="OPTIMAL",
        explored_nodes=int(explored),
    )


def exhaustive_assignment_oracle(actual, n_machines: int) -> ExactResult:
    """Independent full m^n enumeration; only for tiny test fixtures."""
    p = np.asarray(actual, dtype=np.int64)
    if len(p) > 9:
        raise ValueError("tiny exhaustive oracle limited to <=9 jobs")

    best = 10**18
    best_loads = None
    explored = 0
    loads = np.zeros(n_machines, dtype=np.int64)

    def rec(j):
        nonlocal best, best_loads, explored
        if j == len(p):
            explored += 1
            value = int(loads.max(initial=0))
            if value < best:
                best = value
                best_loads = tuple(int(x) for x in loads)
            return
        for m in range(n_machines):
            loads[m] += int(p[j])
            rec(j+1)
            loads[m] -= int(p[j])

    rec(0)
    return ExactResult(int(best), best_loads, "OPTIMAL_EXHAUSTIVE", explored)
