from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class ProcessingTimePredictor:
    model: ExtraTreesRegressor

    @classmethod
    def fit(
        cls,
        feature_rows: np.ndarray,
        processing_times: np.ndarray,
        *,
        seed: int = 42,
        trees: int = 240,
    ):
        model = ExtraTreesRegressor(
            n_estimators=int(trees),
            random_state=int(seed),
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=1,
        )
        model.fit(
            np.asarray(feature_rows, dtype=float),
            np.asarray(processing_times, dtype=float),
        )
        return cls(model)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.asarray(self.model.predict(features), dtype=float),
            1e-6,
        )


def flatten_instances(instances):
    x = np.concatenate([z.features for z in instances], axis=0)
    y = np.concatenate([z.actual for z in instances], axis=0)
    return x, y


def prediction_rmse(actual, predicted) -> float:
    p = np.asarray(actual, dtype=float)
    q = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((p-q)**2)))


def multiplicative_error(actual, predicted) -> float:
    """Worst multiplicative distortion eta >= 1."""
    p = np.asarray(actual, dtype=float)
    q = np.asarray(predicted, dtype=float)
    if p.shape != q.shape or np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("positive equal-shape arrays required")
    ratio = np.maximum(p/q, q/p)
    return float(np.max(ratio))


def mean_absolute_log_error(actual, predicted) -> float:
    p = np.asarray(actual, dtype=float)
    q = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(np.log(q/p))))


def corrupt_predictions_multiplicative(
    actual,
    *,
    seed: int,
    sigma: float,
):
    """Controlled multiplicative log-noise for smoothness experiments."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    p = np.asarray(actual, dtype=float)
    rng = np.random.default_rng(seed)
    if sigma == 0:
        return p.copy()
    eps = rng.normal(0.0, sigma, size=len(p))
    return np.maximum(p*np.exp(eps), 1e-6)
