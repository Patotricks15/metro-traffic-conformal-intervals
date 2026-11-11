from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.data.metro import holiday_dates, load_raw, to_hourly


def write_raw(tmp_path):
    rows = [
        ("02-10-2012 09:00", None, 288.0, 0.0, 0.0, 40, 5000),
        ("02-10-2012 09:00", None, 288.5, 0.0, 0.0, 40, 5000),
        ("02-10-2012 11:00", "Columbus Day", 0.0, 9831.0, 0.0, 90, 4000),
        ("02-10-2012 12:00", None, 290.0, 0.5, 0.0, 10, 4200),
    ]
    columns = ["date_time", "holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_load_raw_parses_day_first_timestamps(tmp_path):
    raw = load_raw(write_raw(tmp_path))
    assert raw["timestamp"].iloc[0] == pd.Timestamp("2012-10-02 09:00")
    assert "date_time" not in raw.columns


def test_to_hourly_builds_complete_grid_with_keep_first_and_no_target_imputation(tmp_path):
    hourly = to_hourly(load_raw(write_raw(tmp_path)))
    assert list(hourly.index) == list(pd.date_range("2012-10-02 09:00", periods=4, freq="h"))
    assert hourly.loc["2012-10-02 09:00", "temp"] == 288.0
    assert np.isnan(hourly.loc["2012-10-02 10:00", "traffic_volume"])


def test_to_hourly_masks_sentinel_temperature_and_implausible_rain(tmp_path):
    hourly = to_hourly(load_raw(write_raw(tmp_path)))
    assert np.isnan(hourly.loc["2012-10-02 11:00", "temp"])
    assert np.isnan(hourly.loc["2012-10-02 11:00", "rain_1h"])
    assert hourly.loc["2012-10-02 12:00", "rain_1h"] == 0.5


def test_holiday_dates_are_normalised_calendar_days(tmp_path):
    dates = holiday_dates(load_raw(write_raw(tmp_path)))
    assert list(dates) == [pd.Timestamp("2012-10-02")]
