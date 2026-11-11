from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_ffill(traffic: pd.Series, period: int = 24) -> pd.Series:
    """Causal daily-naive forecast for origin t: the latest observed value at the same hour of day.

    For a horizon equal to the period, the target at t + period shares its hour of day with t.
    Missing hours are carried forward within each hour-of-day group, so no future value is used.
    """
    position = np.arange(len(traffic))
    return traffic.astype(float).groupby(position % period).ffill()
