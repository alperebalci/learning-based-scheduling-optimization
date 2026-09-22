from __future__ import annotations

import argparse

from .environment import JobShopEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model-path", default="ppo_jssp")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    env = JobShopEnv()
    model = PPO("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")


if __name__ == "__main__":
    main()
