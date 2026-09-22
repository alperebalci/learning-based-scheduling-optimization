from __future__ import annotations

import argparse
from statistics import mean

from .baselines import earliest_due_date, most_work_remaining, random_valid, shortest_processing_time
from .environment import JobShopEnv


def run_policy(policy_name: str, episodes: int = 20) -> dict[str, float]:
    policies = {
        "spt": shortest_processing_time,
        "edd": earliest_due_date,
        "mwr": most_work_remaining,
        "random": random_valid,
    }
    policy = policies[policy_name]
    makespans, tardiness, utilization = [], [], []

    for seed in range(episodes):
        env = JobShopEnv()
        env.reset(seed=seed)
        terminated = False
        info = {}
        while not terminated:
            action = policy(env)
            _, _, terminated, _, info = env.step(action)
        makespans.append(float(info["makespan"]))
        tardiness.append(float(info["total_tardiness"]))
        utilization.append(float(info["machine_utilization"]))

    return {
        "mean_makespan": mean(makespans),
        "mean_total_tardiness": mean(tardiness),
        "mean_machine_utilization": mean(utilization),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["spt", "edd", "mwr", "random"], default="spt")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    metrics = run_policy(args.policy, args.episodes)
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
