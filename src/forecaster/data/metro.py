from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_TIME_FORMAT = "%d-%m-%Y %H:%M"
MAX_RAIN_MM_PER_HOUR = 100.0
MISSING_TEMP_KELVIN = 0.0
COLUMNS = ("traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle export and parse the day-first timestamps."""
    raw = pd.read_csv(path)
    raw["timestamp"] = pd.to_datetime(raw.pop("date_time"), format=RAW_TIME_FORMAT)
    return raw


def holiday_dates(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Calendar dates carrying a holiday label. Known in advance, so usable for target-time features."""
    labelled = raw.loc[raw["holiday"].notna(), "timestamp"]
    return pd.DatetimeIndex(labelled.dt.normalize().drop_duplicates().sort_values())


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a complete hourly grid. Hours with no record keep NaN traffic and are never imputed.

    Duplicate timestamps keep the first record; traffic values are identical across duplicates in
    this export, so the choice only affects weather fields and is deterministic.
    """
    unique = raw.drop_duplicates(subset="timestamp", keep="first").set_index("timestamp").sort_index()
    grid = pd.date_range(unique.index.min(), unique.index.max(), freq="h", name="timestamp")
    hourly = unique.reindex(grid)[list(COLUMNS)].astype(float)
    hourly["temp"] = hourly["temp"].mask(hourly["temp"] == MISSING_TEMP_KELVIN)
    hourly["rain_1h"] = hourly["rain_1h"].mask(hourly["rain_1h"] > MAX_RAIN_MM_PER_HOUR)
    return hourly
