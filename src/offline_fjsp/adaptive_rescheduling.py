# ruff: noqa: I001
"""Adaptive job-shop rescheduling with learned constraint shielding and DQN.

The environment is intentionally compact: two job families follow opposite
two-machine routes, jobs arrive dynamically, machines can become unavailable,
and some operations experience processing-time slowdowns. Hard feasibility is
enforced through action masking. A learned constraint model represents synthetic
expert acceptability and can shield a DQN policy at inference time.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools

import numpy as np


FAMILIES = (0, 1)
IDLE = -1
ACTIONS = tuple(itertools.product((IDLE, 0, 1), repeat=2))
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
CONSTRAINT_FEATURES = (
    "idle_with_ready",
    "urgent_misses",
    "setup_changes",
    "simultaneous_setup_changes",
    "backlog",
    "urgent_backlog",
)


@dataclasses.dataclass(frozen=True)
class DynamicJob:
    job_id: int
    family: int
    release: int
    due: int
    weight: float


@dataclasses.dataclass(frozen=True)
class DisruptionScenario:
    jobs: tuple[DynamicJob, ...]
    availability: np.ndarray
    slowdown: np.ndarray
    planning_horizon: int

    @property
    def max_time(self) -> int:
        return int(self.availability.shape[0])


def route_for_family(family: int) -> tuple[int, int]:
    if family == 0:
        return (0, 1)
    if family == 1:
        return (1, 0)
    raise ValueError("family must be 0 or 1")


def generate_disruption_scenario(
    seed: int,
    *,
    planning_horizon: int = 14,
) -> DisruptionScenario:
    if planning_horizon < 8:
        raise ValueError("planning_horizon must be at least 8")
    rng = np.random.default_rng(seed)

    jobs: list[DynamicJob] = []
    job_id = 0
    for family in (0, 1, 0, 1):
        jobs.append(
            DynamicJob(
                job_id=job_id,
                family=family,
                release=0,
                due=int(rng.integers(4, 7)),
                weight=1.0,
            )
        )
        job_id += 1

    for release in range(1, planning_horizon - 2):
        if rng.random() < 0.42:
            family = int(rng.integers(0, 2))
            rush = bool(rng.random() < 0.30)
            jobs.append(
                DynamicJob(
                    job_id=job_id,
                    family=family,
                    release=release,
                    due=release + (3 if rush else int(rng.integers(4, 7))),
                    weight=3.0 if rush else 1.0,
                )
            )
            job_id += 1

    max_time = planning_horizon + 8
    availability = rng.random((max_time, 2)) >= 0.08
    availability[0, :] = True
    slowdown = rng.random((max_time, 2)) < 0.12
    return DisruptionScenario(
        jobs=tuple(jobs),
        availability=availability.astype(bool),
        slowdown=slowdown.astype(bool),
        planning_horizon=int(planning_horizon),
    )


class AdaptiveJobShopEnv:
    """Discrete-time two-machine job shop with dynamic disruptions."""

    def __init__(self, scenario: DisruptionScenario):
        self.scenario = scenario
        self.jobs = {job.job_id: job for job in scenario.jobs}
        self.reset()

    def reset(self) -> np.ndarray:
        self.time = 0
        self.stage = {job_id: 0 for job_id in self.jobs}
        self.completion: dict[int, int] = {}
        self.machine_setup = [IDLE, IDLE]
        self.busy_remaining = [0, 0]
        self.busy_job: list[int | None] = [None, None]
        self.total_cost = 0.0
        self.total_setup_changes = 0
        self.total_idle_with_ready = 0
        return self.state_vector()

    def _availability(self, machine: int) -> bool:
        index = min(self.time, self.scenario.max_time - 1)
        return bool(self.scenario.availability[index, machine])

    def _slowdown(self, machine: int) -> bool:
        index = min(self.time, self.scenario.max_time - 1)
        return bool(self.scenario.slowdown[index, machine])

    def _is_busy_job(self, job_id: int) -> bool:
        return job_id in self.busy_job

    def released_incomplete_jobs(self) -> list[DynamicJob]:
        return [
            job
            for job in self.scenario.jobs
            if job.release <= self.time and self.stage[job.job_id] < 2
        ]

    def ready_jobs(self, machine: int, family: int) -> list[DynamicJob]:
        route = route_for_family(family)
        rows = []
        for job in self.scenario.jobs:
            if job.family != family or job.release > self.time:
                continue
            if self._is_busy_job(job.job_id):
                continue
            stage = self.stage[job.job_id]
            if stage < 2 and route[stage] == machine:
                rows.append(job)
        return sorted(rows, key=lambda job: (job.due, -job.weight, job.job_id))

    def feasible_actions(self) -> list[tuple[int, int]]:
        choices: list[list[int]] = []
        for machine in range(2):
            if self.busy_remaining[machine] > 0 or not self._availability(machine):
                choices.append([IDLE])
                continue
            machine_choices = [IDLE]
            for family in FAMILIES:
                if self.ready_jobs(machine, family):
                    machine_choices.append(family)
            choices.append(machine_choices)
        return [tuple(action) for action in itertools.product(*choices)]

    def _complete_operation(self, job_id: int) -> None:
        self.stage[job_id] += 1
        if self.stage[job_id] == 2:
            self.completion[job_id] = self.time + 1

    def _selected_job(self, machine: int, family: int) -> DynamicJob:
        jobs = self.ready_jobs(machine, family)
        if not jobs:
            raise RuntimeError("selected family has no ready job")
        return jobs[0]

    def step(
        self,
        action: tuple[int, int],
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = tuple(int(x) for x in action)
        if action not in self.feasible_actions():
            raise ValueError(f"infeasible action: {action}")

        features = action_features(self, action)
        existing_busy = [self.busy_remaining[machine] > 0 for machine in range(2)]
        immediate_completions: list[int] = []
        setup_changes = 0

        for machine, family in enumerate(action):
            if family == IDLE:
                continue
            job = self._selected_job(machine, family)
            if self.machine_setup[machine] not in (IDLE, family):
                setup_changes += 1
            self.machine_setup[machine] = family

            if self._slowdown(machine):
                self.busy_remaining[machine] = 1
                self.busy_job[machine] = job.job_id
            else:
                immediate_completions.append(job.job_id)

        for machine, was_busy in enumerate(existing_busy):
            if not was_busy:
                continue
            self.busy_remaining[machine] -= 1
            if self.busy_remaining[machine] == 0:
                job_id = self.busy_job[machine]
                if job_id is None:
                    raise RuntimeError("busy machine is missing its job")
                immediate_completions.append(job_id)
                self.busy_job[machine] = None

        for job_id in immediate_completions:
            self._complete_operation(job_id)

        self.total_setup_changes += setup_changes
        self.total_idle_with_ready += int(features["idle_with_ready"])
        self.time += 1

        released = self.released_incomplete_jobs()
        backlog = len(released)
        tardy_weight = sum(job.weight for job in released if job.due < self.time)
        period_cost = (
            0.25 * backlog
            + 1.00 * tardy_weight
            + 0.60 * setup_changes
            + 0.15 * float(features["idle_with_ready"])
        )
        self.total_cost += float(period_cost)

        all_released = self.time >= self.scenario.planning_horizon
        all_complete = len(self.completion) == len(self.jobs)
        timed_out = self.time >= self.scenario.max_time
        done = bool((all_released and all_complete) or timed_out)
        if timed_out and not all_complete:
            unfinished = len(self.jobs) - len(self.completion)
            terminal_penalty = 10.0 * unfinished
            self.total_cost += terminal_penalty
            period_cost += terminal_penalty

        info = self.metrics()
        return self.state_vector(), -float(period_cost), done, info

    def state_vector(self) -> np.ndarray:
        values: list[float] = [
            min(self.time / max(self.scenario.planning_horizon, 1), 1.5)
        ]
        for machine in range(2):
            for family in FAMILIES:
                ready = self.ready_jobs(machine, family)
                values.append(min(len(ready), 4) / 4.0)
        for machine in range(2):
            for family in FAMILIES:
                ready = self.ready_jobs(machine, family)
                if ready:
                    slack = min(job.due - self.time for job in ready)
                    values.append(float(np.clip(slack, -3, 6)) / 6.0)
                else:
                    values.append(1.0)

        released = self.released_incomplete_jobs()
        urgent = sum(job.due - self.time <= 2 for job in released)
        values.append(min(len(released), 8) / 8.0)
        values.append(min(urgent, 8) / 8.0)
        values.extend(float(setup) for setup in self.machine_setup)
        values.extend(float(self._availability(machine)) for machine in range(2))
        values.extend(min(self.busy_remaining[machine], 2) / 2.0 for machine in range(2))
        return np.asarray(values, dtype=np.float32)

    def metrics(self) -> dict[str, float]:
        tardiness = 0.0
        for job in self.scenario.jobs:
            completion = self.completion.get(job.job_id, self.time)
            tardiness += job.weight * max(0, completion - job.due)
        return {
            "total_cost": float(self.total_cost),
            "weighted_tardiness": float(tardiness),
            "setup_changes": float(self.total_setup_changes),
            "completion_rate": float(len(self.completion) / max(len(self.jobs), 1)),
            "time": float(self.time),
        }


def minimum_slack_action(env: AdaptiveJobShopEnv) -> tuple[int, int]:
    action: list[int] = []
    for machine in range(2):
        if env.busy_remaining[machine] > 0 or not env._availability(machine):
            action.append(IDLE)
            continue

        choices = []
        for family in FAMILIES:
            ready = env.ready_jobs(machine, family)
            if not ready:
                continue
            job = ready[0]
            slack = job.due - env.time
            setup_penalty = int(env.machine_setup[machine] not in (IDLE, family))
            choices.append((slack, setup_penalty, -job.weight, family))
        action.append(min(choices)[-1] if choices else IDLE)
    result = tuple(action)
    if result not in env.feasible_actions():
        raise RuntimeError("minimum-slack policy produced an infeasible action")
    return result


def action_features(
    env: AdaptiveJobShopEnv,
    action: tuple[int, int],
) -> dict[str, float]:
    if action not in env.feasible_actions():
        raise ValueError("action features require a hard-feasible action")

    idle_with_ready = 0
    urgent_misses = 0
    setup_changes = 0

    for machine, choice in enumerate(action):
        ready_by_family = {
            family: env.ready_jobs(machine, family) for family in FAMILIES
        }
        has_ready = any(ready_by_family.values())
        if (
            choice == IDLE
            and has_ready
            and env._availability(machine)
            and env.busy_remaining[machine] == 0
        ):
            idle_with_ready += 1

        urgent_family = None
        urgent_key = None
        for family, jobs in ready_by_family.items():
            if not jobs:
                continue
            job = jobs[0]
            key = (job.due - env.time, -job.weight, family)
            if key[0] <= 2 and (urgent_key is None or key < urgent_key):
                urgent_key = key
                urgent_family = family
        if urgent_family is not None and choice != urgent_family:
            urgent_misses += 1

        if choice != IDLE and env.machine_setup[machine] not in (IDLE, choice):
            setup_changes += 1

    released = env.released_incomplete_jobs()
    urgent_backlog = sum(job.due - env.time <= 2 for job in released)
    return {
        "idle_with_ready": float(idle_with_ready),
        "urgent_misses": float(urgent_misses),
        "setup_changes": float(setup_changes),
        "simultaneous_setup_changes": float(setup_changes == 2),
        "backlog": float(len(released)),
        "urgent_backlog": float(urgent_backlog),
    }


def synthetic_expert_acceptable(features: dict[str, float]) -> int:
    score = (
        2.2
        - 1.35 * features["idle_with_ready"]
        - 1.75 * features["urgent_misses"]
        - 0.35 * features["setup_changes"]
        - 0.70
        * features["simultaneous_setup_changes"]
        * float(features["backlog"] >= 4.0)
        - 0.10 * max(0.0, features["urgent_backlog"] - 2.0)
    )
    return int(score >= 0.0)


def feature_array(features: dict[str, float]) -> np.ndarray:
    return np.asarray([features[name] for name in CONSTRAINT_FEATURES], dtype=float)


def build_constraint_dataset(
    *,
    n_scenarios: int = 60,
    seed: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if n_scenarios < 2:
        raise ValueError("n_scenarios must be at least 2")

    X: list[np.ndarray] = []
    y: list[int] = []
    for scenario_seed in range(seed, seed + n_scenarios):
        env = AdaptiveJobShopEnv(generate_disruption_scenario(scenario_seed))
        done = False
        while not done:
            for action in env.feasible_actions():
                features = action_features(env, action)
                X.append(feature_array(features))
                y.append(synthetic_expert_acceptable(features))
            action = minimum_slack_action(env)
            _, _, done, _ = env.step(action)

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("constraint dataset must contain acceptable and rejected actions")
    return features, labels


class OperationalConstraintModel:
    """Small dependency-free logistic model for expert acceptability."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        steps: int = 900,
        learning_rate: float = 0.08,
        l2: float = 0.01,
    ) -> OperationalConstraintModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(CONSTRAINT_FEATURES):
            raise ValueError("unexpected constraint feature matrix shape")
        if y.shape != (len(X),):
            raise ValueError("labels must be one-dimensional")
        if set(np.unique(y.astype(int))) != {0, 1}:
            raise ValueError("labels must contain both classes")

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        normalized = (X - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(normalized)), normalized])
        weights = np.zeros(design.shape[1], dtype=float)

        for _ in range(int(steps)):
            logits = np.clip(design @ weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (probabilities - y) / len(y)
            gradient[1:] += l2 * weights[1:]
            weights -= learning_rate * gradient

        self.weights_ = weights
        return self

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("fit the constraint model before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        normalized = (X - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(normalized)), normalized])
        logits = np.clip(design @ self.weights_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def action_probability(
        self,
        env: AdaptiveJobShopEnv,
        action: tuple[int, int],
    ) -> float:
        features = feature_array(action_features(env, action))
        return float(self.predict_probability(features)[0])


class ConstraintShield:
    def __init__(
        self,
        model: OperationalConstraintModel,
        *,
        threshold: float = 0.55,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.model = model
        self.threshold = float(threshold)

    def accepted_actions(
        self,
        env: AdaptiveJobShopEnv,
    ) -> list[tuple[int, int]]:
        feasible = env.feasible_actions()
        probabilities = [self.model.action_probability(env, action) for action in feasible]
        accepted = [
            action
            for action, probability in zip(feasible, probabilities)
            if probability >= self.threshold
        ]
        if accepted:
            return accepted
        return [feasible[int(np.argmax(probabilities))]]


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Install the neural extra with: pip install -e '.[neural]'"
        ) from exc
    return torch, nn


class DQNRescheduler:
    """Masked DQN over the fixed nine-action dispatch space."""

    def __init__(
        self,
        state_dim: int,
        *,
        seed: int = 0,
        gamma: float = 0.97,
        learning_rate: float = 1e-3,
        memory_size: int = 6000,
    ) -> None:
        torch, nn = _torch_modules()
        torch.manual_seed(seed)
        self.torch = torch
        self.rng = np.random.default_rng(seed)
        self.gamma = float(gamma)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        self.memory = collections.deque(maxlen=memory_size)
        self.steps = 0

        def network():
            return nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(ACTIONS)),
            )

        self.q_network = network()
        self.target_network = network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

    @staticmethod
    def action_indices(actions) -> list[int]:
        return [ACTION_TO_INDEX[tuple(action)] for action in actions]

    def _greedy_index(
        self,
        state: np.ndarray,
        valid_indices: list[int],
    ) -> int:
        torch = self.torch
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(tensor)[0].cpu().numpy()
        return int(max(valid_indices, key=lambda idx: q_values[idx]))

    def choose_action(
        self,
        env: AdaptiveJobShopEnv,
        *,
        explore: bool,
        shield: ConstraintShield | None = None,
    ) -> tuple[tuple[int, int], bool]:
        feasible = env.feasible_actions()
        valid_indices = self.action_indices(feasible)
        raw_index = self._greedy_index(env.state_vector(), valid_indices)

        allowed = feasible
        if shield is not None:
            allowed = shield.accepted_actions(env)
        allowed_indices = self.action_indices(allowed)

        if explore and self.rng.random() < self.epsilon:
            chosen_index = int(self.rng.choice(allowed_indices))
        else:
            chosen_index = self._greedy_index(env.state_vector(), allowed_indices)

        intervention = shield is not None and chosen_index != raw_index
        return ACTIONS[chosen_index], bool(intervention)

    def remember(
        self,
        state: np.ndarray,
        action: tuple[int, int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_actions,
    ) -> None:
        self.memory.append(
            (
                np.asarray(state, dtype=np.float32),
                ACTION_TO_INDEX[action],
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                tuple(self.action_indices(next_actions)),
            )
        )

    def replay(self, *, batch_size: int = 32) -> float | None:
        if len(self.memory) < batch_size:
            return None

        torch = self.torch
        indices = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[int(index)] for index in indices]
        states = torch.tensor(
            np.asarray([row[0] for row in batch]),
            dtype=torch.float32,
        )
        actions = torch.tensor([row[1] for row in batch], dtype=torch.int64)
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        next_states = torch.tensor(
            np.asarray([row[3] for row in batch]),
            dtype=torch.float32,
        )
        dones = torch.tensor([row[4] for row in batch], dtype=torch.bool)

        current = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_rows = self.target_network(next_states)
            next_values = []
            for row, transition in zip(target_rows, batch):
                valid = transition[5]
                if not valid:
                    next_values.append(torch.tensor(0.0))
                else:
                    next_values.append(torch.max(row[list(valid)]).cpu())
            next_values_tensor = torch.stack(next_values).to(rewards.device)
            target = rewards + self.gamma * next_values_tensor * (~dones)

        loss = torch.mean((current - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % 40 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.detach().cpu().item())

    def train(
        self,
        *,
        episodes: int = 200,
        scenario_seed: int = 1000,
        batch_size: int = 32,
    ) -> list[float]:
        returns = []
        for episode in range(episodes):
            env = AdaptiveJobShopEnv(
                generate_disruption_scenario(scenario_seed + episode)
            )
            state = env.reset()
            done = False
            episode_return = 0.0
            while not done:
                action, _ = self.choose_action(env, explore=True)
                next_state, reward, done, _ = env.step(action)
                next_actions = [] if done else env.feasible_actions()
                self.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_actions,
                )
                self.replay(batch_size=batch_size)
                state = next_state
                episode_return += reward
            returns.append(float(episode_return))
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )
        return returns


def rollout_heuristic(seed: int) -> dict[str, float]:
    env = AdaptiveJobShopEnv(generate_disruption_scenario(seed))
    done = False
    while not done:
        action = minimum_slack_action(env)
        _, _, done, _ = env.step(action)
    result = env.metrics()
    result["shield_intervention_rate"] = 0.0
    return result


def rollout_dqn(
    agent: DQNRescheduler,
    seed: int,
    *,
    shield: ConstraintShield | None = None,
) -> dict[str, float]:
    env = AdaptiveJobShopEnv(generate_disruption_scenario(seed))
    done = False
    interventions = 0
    decisions = 0
    while not done:
        action, intervention = agent.choose_action(
            env,
            explore=False,
            shield=shield,
        )
        interventions += int(intervention)
        decisions += 1
        _, _, done, _ = env.step(action)
    result = env.metrics()
    result["shield_intervention_rate"] = float(
        interventions / max(decisions, 1)
    )
    return result


def aggregate_results(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "total_cost",
        "weighted_tardiness",
        "setup_changes",
        "completion_rate",
        "shield_intervention_rate",
    )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }


def run_adaptive_benchmark(
    *,
    train_episodes: int = 200,
    eval_seeds=range(3000, 3020),
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    X, y = build_constraint_dataset(n_scenarios=50, seed=200)
    constraint_model = OperationalConstraintModel().fit(X, y)
    shield = ConstraintShield(constraint_model, threshold=0.55)

    sample_env = AdaptiveJobShopEnv(generate_disruption_scenario(1000))
    agent = DQNRescheduler(len(sample_env.state_vector()), seed=seed)
    agent.train(episodes=train_episodes, scenario_seed=1000)

    eval_seeds = list(eval_seeds)
    heuristic = aggregate_results([rollout_heuristic(s) for s in eval_seeds])
    dqn = aggregate_results([rollout_dqn(agent, s) for s in eval_seeds])
    shielded = aggregate_results(
        [rollout_dqn(agent, s, shield=shield) for s in eval_seeds]
    )
    return {
        "minimum_slack": heuristic,
        "dqn": dqn,
        "dqn_with_constraint_shield": shielded,
    }
