"""Streaming MAE using River."""

from __future__ import annotations

from river import metrics


def new_mae() -> metrics.MAE:
    return metrics.MAE()


def update_mae(mae: metrics.MAE, y_true: float, y_pred: float) -> metrics.MAE:
    mae.update(y_true, y_pred)
    return mae


def mae_value(mae: metrics.MAE) -> float:
    return float(mae.get())
