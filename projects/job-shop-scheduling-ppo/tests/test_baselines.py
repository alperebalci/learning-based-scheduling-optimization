from jssp_ppo.baselines import earliest_due_date, most_work_remaining, shortest_processing_time
from jssp_ppo.environment import JobShopEnv
from jssp_ppo.evaluate import run_policy


def test_dispatching_rules_return_valid_actions():
    env = JobShopEnv()
    env.reset(seed=0)
    for policy in (shortest_processing_time, earliest_due_date, most_work_remaining):
        action = policy(env)
        assert action in env.valid_actions()


def test_spt_smoke_evaluation():
    metrics = run_policy("spt", episodes=2)
    assert metrics["mean_makespan"] > 0
    assert metrics["mean_total_tardiness"] >= 0
    assert 0 < metrics["mean_machine_utilization"] <= 1
