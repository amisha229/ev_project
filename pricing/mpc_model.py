"""
MPC-style pricing: prediction-driven demand curve + grid search.

    q(p) = predicted_demand - beta * p

    maximize_p  p*q(p) - lambda_penalty * (q(p) - capacity)^2
"""

from __future__ import annotations

from collections import deque

import numpy as np


def demand_q(price: np.ndarray, predicted_demand: float, beta: float) -> np.ndarray:
    """Demand response anchored on the current ML prediction."""
    return np.maximum(predicted_demand - beta * price, 0.0)


def objective(
    price: np.ndarray,
    predicted_demand: float,
    beta: float,
    capacity: float,
    lambda_penalty: float,
) -> np.ndarray:
    q = demand_q(price, predicted_demand, beta)
    return price * q - lambda_penalty * (q - capacity) ** 2


def optimal_price_grid(
    predicted_demand: float,
    beta: float,
    capacity: float,
    lambda_penalty: float = 0.15,
    p_min: float = 0.5,
    p_max: float = 40.0,
    n_steps: int = 400,
) -> tuple[float, np.ndarray, np.ndarray]:
    prices = np.linspace(p_min, p_max, n_steps)
    scores = objective(prices, predicted_demand, beta, capacity, lambda_penalty)
    idx = int(np.nanargmax(scores))
    return float(prices[idx]), prices, scores


class OnlineDemandCurve:
    """Sliding-window least squares to estimate price sensitivity ``beta``."""

    def __init__(self, maxlen: int = 400, min_samples: int = 35) -> None:
        self.maxlen = maxlen
        self.min_samples = min_samples
        self._p = deque(maxlen=maxlen)
        self._q = deque(maxlen=maxlen)
        self.beta = 0.35

    def update(self, price: float, demand: float) -> None:
        self._p.append(float(price))
        self._q.append(float(demand))
        if len(self._p) < self.min_samples:
            return
        p = np.array(self._p, dtype=float)
        q = np.array(self._q, dtype=float)
        a = np.column_stack([np.ones_like(p), -p])
        coef, _, rank, _ = np.linalg.lstsq(a, q, rcond=None)
        if rank < 2:
            return
        _, beta_hat = float(coef[0]), float(coef[1])
        if beta_hat <= 1e-6:
            return
        self.beta = min(beta_hat, 10.0)
