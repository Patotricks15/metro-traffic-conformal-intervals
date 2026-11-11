from __future__ import annotations

import pandas as pd

LAGS = (0, 1, 2, 23, 144)
ROLLING_WINDOWS = (24, 168)
WEATHER = ("temp", "rain_1h", "snow_1h", "clouds_all")
FEATURE_COLUMNS = (
    "traffic_lag_0",
    "traffic_lag_1",
    "traffic_lag_2",
    "traffic_lag_23",
    "traffic_lag_144",
    "traffic_mean_24",
    "traffic_std_24",
    "traffic_mean_168",
    "traffic_std_168",
    "origin_temp",
    "origin_rain_1h",
    "origin_snow_1h",
    "origin_clouds_all",
    "target_hour",
    "target_dow",
    "target_month",
    "target_is_weekend",
    "target_is_holiday",
)


def build_features(frame: pd.DataFrame, horizon: int, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per origin t, using only values at or before t, for the target at t + horizon.

    Lag and rolling features read the traffic history ending at the origin. Weather is the
    observation at the origin, since no weather forecast is used. Calendar features describe the
    target time, which is known in advance.
    """
    traffic = frame["traffic_volume"].astype(float)
    features = pd.DataFrame(index=frame.index)
    for lag in LAGS:
        features[f"traffic_lag_{lag}"] = traffic.shift(lag)
    for window in ROLLING_WINDOWS:
        features[f"traffic_mean_{window}"] = traffic.rolling(window, min_periods=window // 2).mean()
        features[f"traffic_std_{window}"] = traffic.rolling(window, min_periods=window // 2).std()
    for column in WEATHER:
        features[f"origin_{column}"] = frame[column].astype(float)

    target_index = frame.index + pd.Timedelta(hours=horizon)
    features["target_hour"] = target_index.hour
    features["target_dow"] = target_index.day_of_week
    features["target_month"] = target_index.month
    features["target_is_weekend"] = (target_index.day_of_week >= 5).astype(int)
    features["target_is_holiday"] = target_index.normalize().isin(holidays).astype(int)
    return features[list(FEATURE_COLUMNS)]


def build_targets(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Traffic at t + horizon, aligned to origin t. NaN where the target hour is missing."""
    return frame["traffic_volume"].astype(float).shift(-horizon).rename("target")
