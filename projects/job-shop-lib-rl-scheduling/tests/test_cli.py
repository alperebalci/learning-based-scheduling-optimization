from job_shop_demo.cli import main


def test_cli_runs(capsys):
    main()
    output = capsys.readouterr().out

    assert "Dispatching baselines" in output
    assert "RL-ready Gymnasium rollout" in output
    assert "makespan:" in output
