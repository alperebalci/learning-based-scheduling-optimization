from job_shop_demo.rl_env import make_ft06_env, rollout_first_available


def test_environment_reset_exposes_legal_actions():
    env = make_ft06_env()
    observation, info = env.reset()

    assert observation
    assert "available_operations_with_ids" in info
    assert info["available_operations_with_ids"]


def test_first_available_rollout_completes_ft06():
    result = rollout_first_available()

    assert result["steps"] == 36
    assert result["makespan"] > 0
    assert isinstance(result["total_reward"], float)
