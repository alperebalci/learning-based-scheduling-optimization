from __future__ import annotations

import argparse
import numpy as np

from la_scheduling import (
    ProcessingTimePredictor,
    controlled_error_sweep,
    evaluate_predictions,
    flatten_instances,
    generate_instance_batch,
    lpt_consistency_bound,
    list_scheduling_robustness_bound,
    prediction_rmse,
)


def self_test():
    from la_scheduling import (
        exact_makespan_branch_and_bound,
        exhaustive_assignment_oracle,
        predicted_lpt,
    )
    p = np.array([3, 3, 2, 2, 2], dtype=np.int64)
    exact = exact_makespan_branch_and_bound(p, 2)
    brute = exhaustive_assignment_oracle(p, 2)
    assert exact.makespan == brute.makespan == 6
    lpt = predicted_lpt(p, p.astype(float), 2)
    assert lpt.makespan == 7
    assert abs(lpt.makespan/exact.makespan - 7/6) < 1e-12
    print("Learning-augmented scheduling self-test: OK")


def run_experiment(args):
    train = generate_instance_batch(
        args.train_instances,
        seed=args.seed,
        n_jobs=args.jobs,
    )
    test = generate_instance_batch(
        args.test_instances,
        seed=args.seed + 2_000_000,
        n_jobs=args.jobs,
    )

    x_train, y_train = flatten_instances(train)
    predictor = ProcessingTimePredictor.fit(
        x_train, y_train, seed=args.seed, trees=args.trees
    )
    predictions = [predictor.predict(x.features) for x in test]
    actual_flat = np.concatenate([x.actual for x in test])
    pred_flat = np.concatenate(predictions)

    print("="*104)
    print("LEARNING-AUGMENTED NON-CLAIRVOYANT PARALLEL-MACHINE SCHEDULING")
    print("="*104)
    print(f"machines                              : {args.machines}")
    print(f"jobs / instance                       : {args.jobs}")
    print(f"ML test RMSE                          : {prediction_rmse(actual_flat, pred_flat):.3f}")
    print(f"robust list-scheduling bound          : {list_scheduling_robustness_bound(args.machines):.6f}")
    print(f"exact-prediction LPT consistency bound: {lpt_consistency_bound(args.machines):.6f}")

    metrics = evaluate_predictions(
        test,
        predictions,
        n_machines=args.machines,
    )
    print()
    print(f"{'method':<38}{'mean Cmax':>12}{'mean / OPT':>14}{'max / OPT':>13}{'feasible':>11}")
    for row in metrics:
        print(
            f"{row.name:<38}{row.mean_makespan:12.3f}"
            f"{row.mean_ratio_to_opt:14.4f}{row.max_ratio_to_opt:13.4f}"
            f"{row.feasible_rate:11.3f}"
        )

    if args.sweep_instances > 0:
        sweep_instances = generate_instance_batch(
            args.sweep_instances,
            seed=args.seed + 4_000_000,
            n_jobs=args.sweep_jobs,
        )
        sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
        sweep = controlled_error_sweep(
            sweep_instances,
            n_machines=args.machines,
            sigmas=sigmas,
            seed=args.seed + 5_000_000,
        )
        print()
        print("="*104)
        print("CONTROLLED PREDICTION-ERROR SWEEP")
        print("="*104)
        print(
            f"{'sigma':>7}{'mean |log err|':>16}{'mean eta':>12}"
            f"{'LA ratio':>12}{'static ratio':>14}{'fixed ratio':>13}"
            f"{'true LPT':>11}{'max LA':>10}"
        )
        for row in sweep:
            print(
                f"{row.sigma:7.2f}{row.mean_log_error:16.4f}{row.mean_eta:12.3f}"
                f"{row.predicted_lpt_ratio:12.4f}{row.static_prediction_ratio:14.4f}"
                f"{row.fixed_order_ratio:13.4f}{row.true_lpt_ratio:11.4f}"
                f"{row.max_predicted_lpt_ratio:10.4f}"
            )

        bound = list_scheduling_robustness_bound(args.machines)
        if max(x.max_predicted_lpt_ratio for x in sweep) > bound + 1e-10:
            raise AssertionError("learning-augmented list schedule exceeded proven robustness bound")

    print()
    print(
        "The learning-augmented policy uses predictions only for priority order. "
        "Its actual machine assignment remains dynamic list scheduling, so bad "
        "predictions can hurt consistency but do not remove the classical "
        "list-scheduling worst-case bound."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--machines", type=int, default=3)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--train-instances", type=int, default=220)
    p.add_argument("--test-instances", type=int, default=50)
    p.add_argument("--trees", type=int, default=180)
    p.add_argument("--sweep-instances", type=int, default=40)
    p.add_argument("--sweep-jobs", type=int, default=9)
    p.add_argument("--sigmas", default="0,0.1,0.25,0.5,0.8,1.2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(args)
