"""Tests for adaptive job-shop rescheduling with learned constraint shielding."""

import numpy as np
import pytest

from offline_fjsp.adaptive_rescheduling import (
    ACTIONS,
    AdaptiveJobShopEnv,
    ConstraintShield,
    DQNRescheduler,
    DisruptionScenario,
    DynamicJob,
    OperationalConstraintModel,
    build_constraint_dataset,
    generate_disruption_scenario,
    minimum_slack_action,
)


def test_disruption_scenario_is_reproducible() -> None:
    a = generate_disruption_scenario(77)
    b = generate_disruption_scenario(77)
    assert a.jobs == b.jobs
    assert np.array_equal(a.availability, b.availability)
    assert np.array_equal(a.slowdown, b.slowdown)


def test_hard_action_mask_forces_idle_when_both_machines_are_down() -> None:
    jobs = (
        DynamicJob(0, 0, 0, 3, 1.0),
        DynamicJob(1, 1, 0, 3, 1.0),
    )
    availability = np.ones((12, 2), dtype=bool)
    availability[0, :] = False
    slowdown = np.zeros((12, 2), dtype=bool)
    env = AdaptiveJobShopEnv(
        DisruptionScenario(jobs, availability, slowdown, planning_horizon=8)
    )
    assert env.feasible_actions() == [(-1, -1)]


def test_minimum_slack_policy_always_respects_hard_feasibility() -> None:
    env = AdaptiveJobShopEnv(generate_disruption_scenario(78))
    for _ in range(8):
        action = minimum_slack_action(env)
        assert action in env.feasible_actions()
        _, _, done, _ = env.step(action)
        if done:
            break


def test_learned_constraint_model_recovers_synthetic_expert_labels() -> None:
    X, y = build_constraint_dataset(n_scenarios=24, seed=400)
    split = int(0.75 * len(X))
    model = OperationalConstraintModel().fit(X[:split], y[:split])
    probabilities = model.predict_probability(X[split:])
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = np.mean(predictions == y[split:])
    assert 0.10 < y.mean() < 0.90
    assert accuracy >= 0.82


def test_constraint_shield_returns_only_hard_feasible_actions() -> None:
    X, y = build_constraint_dataset(n_scenarios=16, seed=500)
    model = OperationalConstraintModel().fit(X, y)
    shield = ConstraintShield(model, threshold=0.55)
    env = AdaptiveJobShopEnv(generate_disruption_scenario(501))
    accepted = shield.accepted_actions(env)
    assert accepted
    assert all(action in env.feasible_actions() for action in accepted)


def test_dqn_smoke_respects_action_mask_and_runs_updates() -> None:
    pytest.importorskip("torch")
    env = AdaptiveJobShopEnv(generate_disruption_scenario(600))
    agent = DQNRescheduler(len(env.state_vector()), seed=3)
    returns = agent.train(episodes=6, scenario_seed=600, batch_size=8)
    assert len(returns) == 6
    assert agent.steps > 0

    eval_env = AdaptiveJobShopEnv(generate_disruption_scenario(900))
    action, _ = agent.choose_action(eval_env, explore=False)
    assert action in eval_env.feasible_actions()
    assert action in ACTIONS
