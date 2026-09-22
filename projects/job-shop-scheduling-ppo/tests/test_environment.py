import numpy as np

from jssp_ppo.environment import JobShopEnv


def test_reset_observation_shape_and_bounds():
    env = JobShopEnv()
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert info["completed_jobs"] == 0


def test_valid_action_schedules_one_operation():
    env = JobShopEnv()
    env.reset(seed=1)
    action = env.valid_actions()[0]
    _, reward, terminated, truncated, info = env.step(action)
    assert reward > -10
    assert not truncated
    assert len(env.schedule) == 1
    assert env.job_next_op[action] == 1
    assert not terminated
    assert info["completed_jobs"] == 0


def test_invalid_action_is_penalized_without_progress():
    env = JobShopEnv()
    env.reset(seed=2)
    valid = env.valid_actions()
    first = valid[0]
    env.step(first)
    invalid = first
    before = env.job_next_op.copy()
    _, reward, _, _, _ = env.step(invalid)
    assert reward <= -env.invalid_action_penalty
    assert np.array_equal(before, env.job_next_op)


def test_episode_completes_with_valid_actions():
    env = JobShopEnv()
    env.reset(seed=3)
    terminated = False
    info = {}
    guard = 0
    while not terminated and guard < 200:
        valid = env.valid_actions()
        if not valid:
            env._advance_time()
            valid = env.valid_actions()
        _, _, terminated, _, info = env.step(valid[0])
        guard += 1
    assert terminated
    assert info["completed_jobs"] == env.num_jobs
    assert info["makespan"] > 0
    assert 0.0 < info["machine_utilization"] <= 1.0
