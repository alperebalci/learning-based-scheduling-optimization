from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np


@dataclass(frozen=True)
class Assignment:
    job: int
    machine: int
    start: int
    end: int


@dataclass(frozen=True)
class ScheduleResult:
    makespan: int
    assignments: tuple
    machine_loads: np.ndarray
    order: tuple


def list_schedule(actual, order, n_machines: int) -> ScheduleResult:
    """
    Dynamic non-preemptive list scheduling.

    The order is decided before true processing times are known. Whenever a
    machine becomes idle, it takes the next unscheduled job from the priority
    list. Actual duration is revealed only when that job completes.
    """
    p = np.asarray(actual, dtype=np.int64)
    order = tuple(int(j) for j in order)
    n = len(p)
    if n_machines < 1:
        raise ValueError("n_machines must be positive")
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of all jobs")

    heap = [(0, m) for m in range(n_machines)]
    heapq.heapify(heap)
    assignments = []
    loads = np.zeros(n_machines, dtype=np.int64)

    for j in order:
        start, m = heapq.heappop(heap)
        end = int(start + p[j])
        assignments.append(Assignment(j, m, int(start), end))
        loads[m] = end
        heapq.heappush(heap, (end, m))

    return ScheduleResult(
        makespan=int(loads.max(initial=0)),
        assignments=tuple(assignments),
        machine_loads=loads,
        order=order,
    )


def predicted_lpt(actual, predicted, n_machines: int) -> ScheduleResult:
    q = np.asarray(predicted, dtype=float)
    p = np.asarray(actual)
    if q.shape != p.shape:
        raise ValueError("prediction shape mismatch")
    order = tuple(sorted(range(len(q)), key=lambda j: (-q[j], j)))
    return list_schedule(p, order, n_machines)


def fixed_order_list(actual, n_machines: int) -> ScheduleResult:
    return list_schedule(actual, range(len(actual)), n_machines)


def true_lpt(actual, n_machines: int) -> ScheduleResult:
    p = np.asarray(actual)
    order = tuple(sorted(range(len(p)), key=lambda j: (-p[j], j)))
    return list_schedule(p, order, n_machines)


def static_predicted_partition(actual, predicted, n_machines: int) -> ScheduleResult:
    """
    Brittle prediction-only comparator.

    It commits every job to a machine using predicted loads before processing
    begins. Unlike dynamic list scheduling, wrong predicted machine loads are
    never corrected when actual jobs finish.
    """
    p = np.asarray(actual, dtype=np.int64)
    q = np.asarray(predicted, dtype=float)
    if p.shape != q.shape:
        raise ValueError("prediction shape mismatch")
    predicted_load = np.zeros(n_machines, dtype=float)
    machine_jobs = [[] for _ in range(n_machines)]

    order = tuple(sorted(range(len(p)), key=lambda j: (-q[j], j)))
    for j in order:
        m = min(range(n_machines), key=lambda k: (predicted_load[k], k))
        machine_jobs[m].append(j)
        predicted_load[m] += q[j]

    assignments = []
    actual_load = np.zeros(n_machines, dtype=np.int64)
    for m, jobs in enumerate(machine_jobs):
        t = 0
        for j in jobs:
            end = t + int(p[j])
            assignments.append(Assignment(j, m, t, end))
            t = end
        actual_load[m] = t

    return ScheduleResult(
        makespan=int(actual_load.max(initial=0)),
        assignments=tuple(assignments),
        machine_loads=actual_load,
        order=order,
    )


def audit_schedule(actual, result: ScheduleResult, n_machines: int) -> float:
    p = np.asarray(actual, dtype=np.int64)
    violation = 0.0
    if len(result.assignments) != len(p):
        violation = max(violation, 1.0)

    seen = set()
    by_machine = {m: [] for m in range(n_machines)}
    for a in result.assignments:
        if a.job in seen:
            violation = max(violation, 1.0)
        seen.add(a.job)
        if not (0 <= a.job < len(p) and 0 <= a.machine < n_machines):
            violation = max(violation, 1.0)
            continue
        violation = max(
            violation,
            float(a.start < 0),
            float(abs((a.end-a.start)-int(p[a.job]))),
        )
        by_machine[a.machine].append(a)

    for ops in by_machine.values():
        ops.sort(key=lambda x: x.start)
        for a, b in zip(ops, ops[1:]):
            violation = max(violation, float(max(a.end-b.start, 0)))

    if seen != set(range(len(p))):
        violation = max(violation, 1.0)
    if result.assignments:
        violation = max(
            violation,
            float(abs(max(a.end for a in result.assignments)-result.makespan)),
        )
    return float(violation)


def list_scheduling_robustness_bound(n_machines: int) -> float:
    if n_machines < 1:
        raise ValueError("n_machines must be positive")
    return 2.0 - 1.0/n_machines


def lpt_consistency_bound(n_machines: int) -> float:
    if n_machines < 2:
        return 1.0
    return 4.0/3.0 - 1.0/(3.0*n_machines)
