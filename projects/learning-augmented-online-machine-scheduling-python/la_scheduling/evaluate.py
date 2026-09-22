from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .exact import exact_makespan_branch_and_bound
from .predictor import (
    mean_absolute_log_error,
    multiplicative_error,
    corrupt_predictions_multiplicative,
)
from .schedule import (
    audit_schedule,
    fixed_order_list,
    predicted_lpt,
    static_predicted_partition,
    true_lpt,
)


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    mean_makespan: float
    mean_ratio_to_opt: float
    max_ratio_to_opt: float
    feasible_rate: float


@dataclass(frozen=True)
class ErrorSweepPoint:
    sigma: float
    mean_log_error: float
    mean_eta: float
    predicted_lpt_ratio: float
    static_prediction_ratio: float
    fixed_order_ratio: float
    true_lpt_ratio: float
    max_predicted_lpt_ratio: float


def evaluate_predictions(instances, predictions, *, n_machines: int):
    if len(instances) != len(predictions):
        raise ValueError("instances/predictions length mismatch")

    records = {
        "Learning-augmented predicted LPT": [],
        "Prediction-only static partition": [],
        "Prediction-free fixed-order list": [],
        "Clairvoyant true LPT": [],
    }
    feasible = {name: [] for name in records}

    for instance, pred in zip(instances, predictions):
        opt = exact_makespan_branch_and_bound(instance.actual, n_machines).makespan
        methods = {
            "Learning-augmented predicted LPT": predicted_lpt(
                instance.actual, pred, n_machines
            ),
            "Prediction-only static partition": static_predicted_partition(
                instance.actual, pred, n_machines
            ),
            "Prediction-free fixed-order list": fixed_order_list(
                instance.actual, n_machines
            ),
            "Clairvoyant true LPT": true_lpt(instance.actual, n_machines),
        }
        for name, result in methods.items():
            records[name].append((result.makespan, result.makespan/opt))
            feasible[name].append(
                audit_schedule(instance.actual, result, n_machines) <= 1e-12
            )

    out = []
    for name, rows in records.items():
        a = np.asarray(rows, dtype=float)
        out.append(MethodMetrics(
            name=name,
            mean_makespan=float(a[:,0].mean()),
            mean_ratio_to_opt=float(a[:,1].mean()),
            max_ratio_to_opt=float(a[:,1].max()),
            feasible_rate=float(np.mean(feasible[name])),
        ))
    return tuple(out)


def controlled_error_sweep(
    instances,
    *,
    n_machines: int,
    sigmas,
    seed: int = 42,
):
    optima = [
        exact_makespan_branch_and_bound(x.actual, n_machines).makespan
        for x in instances
    ]
    rows = []
    for sigma in sigmas:
        pred_ratios = []
        static_ratios = []
        fixed_ratios = []
        true_ratios = []
        log_errors = []
        etas = []
        for i, (instance, opt) in enumerate(zip(instances, optima)):
            pred = corrupt_predictions_multiplicative(
                instance.actual,
                seed=seed + 1_000_003*i + int(round(10000*sigma)),
                sigma=float(sigma),
            )
            log_errors.append(mean_absolute_log_error(instance.actual, pred))
            etas.append(multiplicative_error(instance.actual, pred))
            pred_ratios.append(
                predicted_lpt(instance.actual, pred, n_machines).makespan/opt
            )
            static_ratios.append(
                static_predicted_partition(
                    instance.actual, pred, n_machines
                ).makespan/opt
            )
            fixed_ratios.append(
                fixed_order_list(instance.actual, n_machines).makespan/opt
            )
            true_ratios.append(
                true_lpt(instance.actual, n_machines).makespan/opt
            )

        rows.append(ErrorSweepPoint(
            sigma=float(sigma),
            mean_log_error=float(np.mean(log_errors)),
            mean_eta=float(np.mean(etas)),
            predicted_lpt_ratio=float(np.mean(pred_ratios)),
            static_prediction_ratio=float(np.mean(static_ratios)),
            fixed_order_ratio=float(np.mean(fixed_ratios)),
            true_lpt_ratio=float(np.mean(true_ratios)),
            max_predicted_lpt_ratio=float(np.max(pred_ratios)),
        ))
    return tuple(rows)
