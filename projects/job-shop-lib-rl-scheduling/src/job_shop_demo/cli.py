from .baselines import compare_dispatching_rules
from .rl_env import rollout_first_available


def main() -> None:
    print("Dispatching baselines (FT06)")
    for rule, makespan in compare_dispatching_rules().items():
        print(f"{rule}: {makespan}")

    rollout = rollout_first_available()
    print("\nRL-ready Gymnasium rollout")
    print(f"steps: {rollout['steps']}")
    print(f"makespan: {rollout['makespan']}")
    print(f"total_reward: {rollout['total_reward']:.2f}")


if __name__ == "__main__":
    main()
