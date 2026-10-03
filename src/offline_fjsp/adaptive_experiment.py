"""CLI for adaptive disruption-aware job-shop rescheduling."""

from __future__ import annotations

import argparse

from .adaptive_rescheduling import run_adaptive_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_seeds = range(3000, 3000 + args.eval_episodes)
    results = run_adaptive_benchmark(
        train_episodes=args.train_episodes,
        eval_seeds=eval_seeds,
        seed=args.seed,
    )

    print("Adaptive job-shop rescheduling benchmark")
    print(
        f"{'method':<30}{'cost':>10}{'tardiness':>12}"
        f"{'setups':>10}{'complete':>11}{'shield':>10}"
    )
    for method, metrics in results.items():
        print(
            f"{method:<30}"
            f"{metrics['total_cost']:>10.3f}"
            f"{metrics['weighted_tardiness']:>12.3f}"
            f"{metrics['setup_changes']:>10.3f}"
            f"{metrics['completion_rate']:>11.3f}"
            f"{metrics['shield_intervention_rate']:>10.3f}"
        )


if __name__ == "__main__":
    main()
