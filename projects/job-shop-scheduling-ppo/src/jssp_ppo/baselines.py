from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .environment import JobShopEnv

Policy = Callable[[JobShopEnv], int]


def shortest_processing_time(env: JobShopEnv) -> int:
    valid = env.valid_actions()
    if not valid:
        env._advance_time()
        valid = env.valid_actions()
    return min(valid, key=lambda j: env.jobs[j].operations[int(env.job_next_op[j])].duration)


def earliest_due_date(env: JobShopEnv) -> int:
    valid = env.valid_actions()
    if not valid:
        env._advance_time()
        valid = env.valid_actions()
    return min(valid, key=lambda j: env.jobs[j].due_date)


def most_work_remaining(env: JobShopEnv) -> int:
    valid = env.valid_actions()
    if not valid:
        env._advance_time()
        valid = env.valid_actions()

    def remaining(j: int) -> int:
        idx = int(env.job_next_op[j])
        return sum(op.duration for op in env.jobs[j].operations[idx:])

    return max(valid, key=remaining)


def random_valid(env: JobShopEnv) -> int:
    valid = env.valid_actions()
    if not valid:
        env._advance_time()
        valid = env.valid_actions()
    return int(env.np_random.choice(np.asarray(valid, dtype=np.int64)))
