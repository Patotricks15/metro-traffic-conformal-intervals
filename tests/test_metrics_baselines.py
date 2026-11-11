from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecaster.evaluation.metrics import coverage, mae, mase, seasonal_scale
from forecaster.models.baselines import seasonal_naive_ffill


def test_mae_ignores_missing_actuals():
    assert mae(np.array([1.0, np.nan, 3.0]), np.array([2.0, 5.0, 3.0])) == pytest.approx(0.5)


def test_coverage_counts_closed_interval_hits():
    assert coverage(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.full(3, 2.5)) == pytest.approx(2 / 3)


def test_seasonal_scale_and_mase_on_a_linear_history():
    history = np.arange(72, dtype=float)
    assert seasonal_scale(history, 24) == pytest.approx(24.0)
    assert mase(history[24:], history[24:] + 12.0, history, 24) == pytest.approx(0.5)


def test_seasonal_naive_is_causal():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(size=300))
    series.iloc[[50, 51, 120]] = np.nan
    cut = 150
    full = seasonal_naive_ffill(series, 24)
    truncated = seasonal_naive_ffill(series.iloc[:cut], 24)
    pd.testing.assert_series_equal(full.iloc[:cut], truncated)


def test_seasonal_naive_fills_from_same_hour_of_day():
    series = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    filled = seasonal_naive_ffill(series, 3)
    assert filled.iloc[3] == 1.0
