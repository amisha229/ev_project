"""
Adaptive Random Forest regressor for streaming demand (online learning).

River 0.22+ implements this as ``ARFRegressor`` in ``river.forest`` (Adaptive
Random Forest for regression). Public API: ``predict_one``, ``learn_one``.
"""

from __future__ import annotations

from typing import Any

from river import preprocessing
from river.forest import ARFRegressor


def build_model(
    seed: int = 42,
    n_models: int = 10,
    grace_period: int = 45,
    with_scaler: bool = True,
) -> Any:
    """Scaler + ARF pipeline for numeric stability on heterogeneous features."""
    arf = ARFRegressor(
        seed=seed,
        n_models=n_models,
        grace_period=grace_period,
        max_features="sqrt",
        aggregation_method="median",
        disable_weighted_vote=True,
    )
    if with_scaler:
        return preprocessing.StandardScaler() | arf
    return arf


def predict_safe(model: Any, x: dict[str, float]) -> float | None:
    pred = model.predict_one(x)
    if pred is None:
        return None
    return float(pred)


def learn_step(model: Any, x: dict[str, float], y: float) -> Any:
    return model.learn_one(x, y)
