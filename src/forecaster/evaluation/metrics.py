from __future__ import annotations

import numpy as np


def _observed(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only positions where the actual value was observed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("actuals and forecasts must have the same shape")
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual, pred = _observed(y_true, y_pred)
    return float(np.mean(np.abs(actual - pred))) if actual.size else float("nan")


def pinball(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Quantile (pinball) loss at level tau."""
    actual, pred = _observed(y_true, y_pred)
    if not actual.size:
        return float("nan")
    diff = actual - pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of observed actuals inside the closed interval [lo, hi]."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return float("nan")
    return float(np.mean((y_true[keep] >= lo[keep]) & (y_true[keep] <= hi[keep])))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    width = width[np.isfinite(width)]
    return float(width.mean()) if width.size else float("nan")


def seasonal_scale(history: np.ndarray, period: int = 24) -> float:
    """Mean absolute seasonal-naive error on observed pairs of the history."""
    history = np.asarray(history, dtype=float)
    diffs = np.abs(history[period:] - history[:-period])
    diffs = diffs[np.isfinite(diffs)]
    return float(diffs.mean()) if diffs.size else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, period: int = 24) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE of the history. NaN when the scale is zero."""
    scale = seasonal_scale(history, period)
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale
