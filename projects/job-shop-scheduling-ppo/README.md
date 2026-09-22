# Job Shop Scheduling with PPO

A compact industrial-engineering benchmark for comparing classical dispatching rules with reinforcement learning on a job-shop scheduling problem (JSSP).

## Why this project matters

Job-shop scheduling is a core operations-research problem: each job consists of an ordered sequence of operations, each operation requires a specific machine, and machine capacity is limited to one operation at a time. The scheduling objective is not only feasibility, but also operational performance such as makespan, tardiness, and utilization.

This repository frames JSSP as an event-driven Markov decision process and provides PPO as the learning-based policy, together with classical dispatching-rule baselines.

## MDP formulation

### State
The observation includes:

- normalized progress of every job,
- normalized job ready times,
- normalized machine ready times,
- current simulation time,
- normalized due dates.

### Action
The agent selects one job. If that job's next operation is feasible at the current decision epoch, it is scheduled. Invalid selections receive a penalty.

### Transition dynamics

An operation is feasible when:

1. all previous operations of the selected job are complete, and
2. the required machine is available.

The environment is event-driven. When no operation can start immediately, simulation time advances to the earliest next feasible event.

### Reward
The reward combines:

- small scheduling-time penalties,
- explicit invalid-action penalties,
- job-completion rewards,
- tardiness penalties,
- terminal makespan and total-tardiness penalties.

The exact weights are intentionally simple and readable so they can be changed for experiments.

## Baselines

The repository includes four dispatching policies:

- `SPT`: shortest processing time,
- `EDD`: earliest due date,
- `MWR`: most work remaining,
- random valid action.

These are important because an RL scheduling policy should be compared against credible scheduling heuristics rather than against a random policy only.

## Main KPIs

Evaluation reports:

- mean makespan,
- mean total tardiness,
- mean machine utilization,
- completed jobs.

For industrial use, additional KPIs can be added easily: weighted tardiness, setup cost, WIP, energy use, overtime, sequence-dependent setup time, or service level.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── jssp_ppo/
│       ├── __init__.py
│       ├── environment.py
│       ├── baselines.py
│       ├── evaluate.py
│       └── train_ppo.py
├── tests/
│   ├── test_environment.py
│   └── test_baselines.py
└── .github/workflows/ci.yml
```

## Installation

Core environment and baseline evaluation:

```bash
pip install -e .
```

With tests:

```bash
pip install -e '.[test]'
```

With PPO support:

```bash
pip install -e '.[rl]'
```

## Run classical scheduling baselines

```bash
python -m jssp_ppo.evaluate --policy spt --episodes 20
python -m jssp_ppo.evaluate --policy edd --episodes 20
python -m jssp_ppo.evaluate --policy mwr --episodes 20
```

## Train PPO

```bash
python -m jssp_ppo.train_ppo --timesteps 50000 --model-path ppo_jssp
```

The training entry point uses Stable-Baselines3 PPO with an MLP policy.

## Research extensions

Useful extensions for a thesis or paper include:

1. **Action masking** so the policy assigns probability only to feasible jobs.
2. **Dynamic arrivals** where jobs enter the shop over time.
3. **Machine breakdowns** and stochastic processing times.
4. **Sequence-dependent setup times**.
5. **Flexible JSSP**, where an operation can be processed by alternative machines.
6. **Multi-objective RL** for makespan, tardiness, energy, and WIP.
7. **Graph neural networks** for permutation- and topology-aware state encoding.
8. **Generalization tests** across unseen job/machine counts.
9. **MILP or CP-SAT lower bounds** for small instances.
10. **Benchmark datasets** such as Taillard or OR-Library instances.

## Industrial interpretation

A production scheduler should not replace a deterministic optimization model merely because RL is available. RL becomes more compelling when the shop is dynamic, stochastic, repeatedly rescheduled, or too large for exact re-optimization at every event. This repository is therefore designed as a comparison framework rather than a claim that PPO dominates classical OR methods.

## CI

GitHub Actions runs the package and tests on Python 3.10, 3.11, and 3.12. It also executes a quick SPT evaluation smoke test. Full PPO training is deliberately excluded from CI because it is computationally expensive and nondeterministic relative to unit testing.
