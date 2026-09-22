from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ParallelMachineInstance:
    """Non-clairvoyant identical-parallel-machine makespan instance.

    All jobs are available at time zero.
    actual[j] is unknown to the scheduler until job j completes.
    features[j] are observable before scheduling and can be used for prediction.
    """
    actual: np.ndarray
    features: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.actual, dtype=np.int64)
        x = np.asarray(self.features, dtype=np.float64)
        if p.ndim != 1 or x.ndim != 2 or len(p) != len(x):
            raise ValueError("actual must be [n] and features [n,f]")
        if np.any(p <= 0):
            raise ValueError("processing times must be positive")

    @property
    def n_jobs(self) -> int:
        return int(len(self.actual))


def generate_parallel_machine_instance(
    *,
    seed: int,
    n_jobs: int = 12,
    feature_dim: int = 7,
) -> ParallelMachineInstance:
    if n_jobs < 2 or feature_dim < 7:
        raise ValueError("n_jobs >=2 and feature_dim >=7 required")

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n_jobs, feature_dim))

    mean = (
        4.0
        + 28.0*x[:, 0]
        + 14.0*x[:, 1]**2
        + 9.0*x[:, 2]*x[:, 3]
        + 8.0*np.sin(np.pi*x[:, 4])**2
        + 6.0*np.maximum(x[:, 5]-0.45, 0.0)
        + 5.0*x[:, 6]*x[:, 0]
    )
    sigma = 1.0 + 4.0*x[:, 1] + 2.0*x[:, 5]
    noise = rng.normal(0.0, sigma)
    rare = (rng.random(n_jobs) < 0.08) * rng.uniform(8.0, 22.0, n_jobs)
    actual = np.maximum(np.rint(mean + noise + rare), 1).astype(np.int64)
    return ParallelMachineInstance(actual=actual, features=x)


def generate_instance_batch(
    n_instances: int,
    *,
    seed: int,
    n_jobs: int,
    feature_dim: int = 7,
):
    if n_instances < 1:
        raise ValueError("n_instances must be positive")
    return tuple(
        generate_parallel_machine_instance(
            seed=seed + 104729*i,
            n_jobs=n_jobs,
            feature_dim=feature_dim,
        )
        for i in range(n_instances)
    )
