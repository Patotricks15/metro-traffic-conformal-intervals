from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(n_hours: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n_hours, freq="h", name="timestamp")
    hour = index.hour.to_numpy()
    weekend_drop = -300.0 * (index.day_of_week.to_numpy() >= 5)
    base = 2000.0 + 1500.0 * np.sin(2 * np.pi * hour / 24) + weekend_drop
    traffic = np.clip(base + rng.normal(0.0, 150.0, n_hours), 0.0, None)
    return pd.DataFrame(
        {
            "traffic_volume": traffic,
            "temp": 280.0 + 10.0 * rng.standard_normal(n_hours),
            "rain_1h": np.clip(rng.exponential(0.2, n_hours), 0.0, None),
            "snow_1h": np.zeros(n_hours),
            "clouds_all": rng.integers(0, 100, n_hours).astype(float),
        },
        index=index,
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def holidays() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(["2015-01-05", "2015-01-19"])


@pytest.fixture
def fast_params() -> dict:
    return {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}
