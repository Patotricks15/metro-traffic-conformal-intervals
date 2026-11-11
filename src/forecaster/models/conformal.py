from __future__ import annotations

import numpy as np


def cqr_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Conformity score of conformalized quantile regression: negative when inside the band."""
    y = np.asarray(y, dtype=float)
    low, high = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.maximum(low - y, y - high)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile at coverage 1 - alpha.

    Returns +inf when there are too few scores for the requested coverage.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    values = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    n = values.size
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(values[rank - 1])


def cqr_interval(lo: np.ndarray, hi: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Widen (or shrink, for negative q) the base quantile band by the conformal correction."""
    low, high = np.minimum(lo, hi), np.maximum(lo, hi)
    return low - q, high + q


class AdaptiveConformal:
    """Adaptive conformal inference (Gibbs and Candes): miscoverage level tracks the target.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t), where err_t = 1 on a miss. Scores from past
    outcomes join the calibration pool, so the threshold always uses only observed outcomes.
    """

    def __init__(
        self,
        target_alpha: float,
        calibration_scores: np.ndarray,
        gamma: float = 0.01,
        alpha_bounds: tuple[float, float] = (0.001, 0.5),
    ) -> None:
        if not 0 < target_alpha < 1:
            raise ValueError("target_alpha must lie in (0, 1)")
        self.target_alpha = target_alpha
        self.alpha = target_alpha
        self.gamma = gamma
        self.alpha_bounds = alpha_bounds
        self.scores = list(np.asarray(calibration_scores, dtype=float)[np.isfinite(calibration_scores)])

    def threshold(self) -> float:
        return conformal_quantile(np.asarray(self.scores), self.alpha)

    def update(self, score: float, threshold_used: float) -> None:
        """Record the realised score and the threshold that was applied to it."""
        missed = bool(score > threshold_used)
        low, high = self.alpha_bounds
        self.alpha = float(np.clip(self.alpha + self.gamma * (self.target_alpha - float(missed)), low, high))
        self.scores.append(float(score))
