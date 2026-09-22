# JobShopLib RL Scheduling

A compact Job Shop Scheduling Problem (JSSP) demo built with `job-shop-lib==1.7.0`.

The repository covers two complementary layers:

- deterministic dispatching-rule baselines on the classic FT06 benchmark;
- an RL-ready `SingleJobShopGraphEnv` Gymnasium environment with a complete legal-action rollout.

## Why this example

Before training a reinforcement-learning agent, it is useful to verify that the environment, action semantics, rewards, termination logic, and classical baselines are all correct. This project provides that reproducible foundation without introducing a second RL framework.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
job-shop-demo
```

The CLI prints makespans for Shortest Processing Time, Most Work Remaining, Most Operations Remaining, and First Come First Served, then executes a deterministic rollout through JobShopLib's Gymnasium environment.

## Tests

```bash
pytest
```

The suite checks dispatching-rule integration, deterministic baseline behaviour, environment reset/action availability, full FT06 episode termination, CLI execution, and at least 90% project coverage.

GitHub Actions runs the suite on Python 3.10, 3.11, 3.12, and 3.13.

## RL extension path

`make_ft06_env()` returns the native JobShopLib Gymnasium environment. A learned policy can therefore replace `rollout_first_available()` later while retaining the same environment and baseline comparison layer.
