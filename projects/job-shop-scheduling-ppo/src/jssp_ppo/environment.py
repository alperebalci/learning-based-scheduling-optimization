from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Operation:
    machine: int
    duration: int


@dataclass(frozen=True)
class Job:
    operations: tuple[Operation, ...]
    due_date: int


def default_instance() -> tuple[Job, ...]:
    return (
        Job((Operation(0, 3), Operation(1, 2), Operation(2, 2)), due_date=10),
        Job((Operation(1, 2), Operation(2, 4), Operation(0, 2)), due_date=12),
        Job((Operation(2, 4), Operation(0, 3), Operation(1, 1)), due_date=13),
        Job((Operation(0, 2), Operation(2, 3), Operation(1, 3)), due_date=14),
    )


class JobShopEnv(gym.Env):
    """Event-driven job-shop scheduling environment.

    An action selects a job. If its next operation can be scheduled at the current
    decision epoch, the operation is assigned. Invalid actions receive a penalty.
    When no operation is schedulable, time advances to the next machine/job release.
    """

    metadata = {"render_modes": []}

    def __init__(self, jobs: Iterable[Job] | None = None, invalid_action_penalty: float = 2.0):
        super().__init__()
        self.jobs = tuple(jobs or default_instance())
        self.num_jobs = len(self.jobs)
        self.num_machines = 1 + max(op.machine for job in self.jobs for op in job.operations)
        self.max_ops = max(len(job.operations) for job in self.jobs)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.action_space = spaces.Discrete(self.num_jobs)
        # job progress, job ready times, machine ready times, current time, due dates
        obs_dim = self.num_jobs + self.num_jobs + self.num_machines + 1 + self.num_jobs
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._horizon_scale = float(sum(op.duration for job in self.jobs for op in job.operations) + max(j.due_date for j in self.jobs))
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_time = 0
        self.job_next_op = np.zeros(self.num_jobs, dtype=np.int32)
        self.job_ready = np.zeros(self.num_jobs, dtype=np.int32)
        self.machine_ready = np.zeros(self.num_machines, dtype=np.int32)
        self.completion_times = np.zeros(self.num_jobs, dtype=np.int32)
        self.schedule: list[dict[str, int]] = []
        return self._obs(), self._info()

    def _done(self) -> bool:
        return all(self.job_next_op[j] >= len(self.jobs[j].operations) for j in range(self.num_jobs))

    def valid_actions(self) -> list[int]:
        valid: list[int] = []
        for j, job in enumerate(self.jobs):
            idx = int(self.job_next_op[j])
            if idx >= len(job.operations):
                continue
            op = job.operations[idx]
            if self.job_ready[j] <= self.current_time and self.machine_ready[op.machine] <= self.current_time:
                valid.append(j)
        return valid

    def _advance_time(self) -> None:
        if self._done():
            return
        candidates: list[int] = []
        for j, job in enumerate(self.jobs):
            idx = int(self.job_next_op[j])
            if idx >= len(job.operations):
                continue
            op = job.operations[idx]
            candidates.append(max(int(self.job_ready[j]), int(self.machine_ready[op.machine])))
        if candidates:
            self.current_time = max(self.current_time, min(candidates))

    def step(self, action: int):
        if self._done():
            raise RuntimeError("Episode is finished; call reset().")

        valid = self.valid_actions()
        if not valid:
            self._advance_time()
            valid = self.valid_actions()

        reward = -0.05
        if int(action) not in valid:
            reward -= self.invalid_action_penalty
        else:
            j = int(action)
            idx = int(self.job_next_op[j])
            op = self.jobs[j].operations[idx]
            start = self.current_time
            end = start + op.duration
            self.machine_ready[op.machine] = end
            self.job_ready[j] = end
            self.job_next_op[j] += 1
            self.schedule.append({"job": j, "operation": idx, "machine": op.machine, "start": start, "end": end})
            reward -= 0.01 * op.duration

            if self.job_next_op[j] >= len(self.jobs[j].operations):
                self.completion_times[j] = end
                tardiness = max(0, end - self.jobs[j].due_date)
                reward += 2.0 - 0.1 * tardiness

        if not self._done() and not self.valid_actions():
            self._advance_time()

        terminated = self._done()
        if terminated:
            makespan = int(max(self.completion_times))
            total_tardiness = int(sum(max(0, int(self.completion_times[j]) - self.jobs[j].due_date) for j in range(self.num_jobs)))
            reward += 10.0 - 0.05 * makespan - 0.1 * total_tardiness

        return self._obs(), float(reward), terminated, False, self._info()

    def _obs(self) -> np.ndarray:
        progress = np.array([
            self.job_next_op[j] / len(self.jobs[j].operations) for j in range(self.num_jobs)
        ], dtype=np.float32)
        job_ready = np.clip(self.job_ready / self._horizon_scale, 0.0, 1.0).astype(np.float32)
        machine_ready = np.clip(self.machine_ready / self._horizon_scale, 0.0, 1.0).astype(np.float32)
        current = np.array([min(self.current_time / self._horizon_scale, 1.0)], dtype=np.float32)
        due_dates = np.array([min(j.due_date / self._horizon_scale, 1.0) for j in self.jobs], dtype=np.float32)
        return np.concatenate([progress, job_ready, machine_ready, current, due_dates]).astype(np.float32)

    def _info(self) -> dict[str, float | int | list[int]]:
        completed = int(sum(self.job_next_op[j] >= len(self.jobs[j].operations) for j in range(self.num_jobs)))
        makespan = int(max(self.completion_times)) if completed else int(self.current_time)
        tardiness = int(sum(max(0, int(self.completion_times[j]) - self.jobs[j].due_date) for j in range(self.num_jobs) if self.completion_times[j] > 0))
        busy = sum(op["end"] - op["start"] for op in self.schedule)
        denom = max(1, self.num_machines * max(1, makespan))
        utilization = busy / denom
        return {
            "current_time": int(self.current_time),
            "completed_jobs": completed,
            "makespan": makespan,
            "total_tardiness": tardiness,
            "machine_utilization": float(utilization),
            "valid_actions": self.valid_actions(),
        }
