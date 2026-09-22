from __future__ import annotations

from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching import DispatcherObserverConfig
from job_shop_lib.dispatching.feature_observers import FeatureObserverType, FeatureType
from job_shop_lib.graphs import build_disjunctive_graph
from job_shop_lib.reinforcement_learning import MakespanReward, SingleJobShopGraphEnv


def make_ft06_env() -> SingleJobShopGraphEnv:
    """Builds an RL-ready Gymnasium environment for the classic FT06 instance."""
    instance = load_benchmark_instance("ft06")
    graph = build_disjunctive_graph(instance)
    feature_configs = [
        DispatcherObserverConfig(
            FeatureObserverType.IS_READY,
            kwargs={"feature_types": [FeatureType.JOBS]},
        )
    ]
    return SingleJobShopGraphEnv(
        job_shop_graph=graph,
        feature_observer_configs=feature_configs,
        reward_function_config=DispatcherObserverConfig(MakespanReward),
        render_mode=None,
    )


def rollout_first_available(env: SingleJobShopGraphEnv | None = None) -> dict[str, float | int]:
    """Runs a deterministic legal-action rollout through the RL environment.

    This is deliberately a policy/environment integration example rather than a
    trained agent. At every step it selects the lexicographically first legal
    action returned by JobShopLib.
    """
    environment = env or make_ft06_env()
    _, info = environment.reset()
    terminated = False
    total_reward = 0.0
    steps = 0

    while not terminated:
        available = info["available_operations_with_ids"]
        if not available:
            raise RuntimeError("environment returned no legal action before termination")

        _, machine_id, job_id = min(available, key=lambda item: (item[2], item[1], item[0]))
        _, reward, terminated, truncated, info = environment.step((job_id, machine_id))
        if truncated:
            raise RuntimeError("unexpected truncation in deterministic rollout")
        total_reward += float(reward)
        steps += 1

    return {
        "steps": steps,
        "makespan": int(environment.current_makespan()),
        "total_reward": total_reward,
    }
