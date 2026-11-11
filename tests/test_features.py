from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecaster.features.origin import FEATURE_COLUMNS, build_features, build_targets

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 24


def test_feature_registry_matches_code(frame, holidays):
    registry = yaml.safe_load((ROOT / "conf" / "features.yaml").read_text(encoding="utf-8"))
    declared = [entry["name"] for entry in registry["features"]]
    assert declared == list(FEATURE_COLUMNS)
    features = build_features(frame, HORIZON, holidays)
    assert list(features.columns) == declared


def test_truncated_history_gives_identical_features(frame, holidays):
    cut = 800
    full = build_features(frame, HORIZON, holidays)
    truncated = build_features(frame.iloc[:cut], HORIZON, holidays)
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_future_values_do_not_change_features_at_earlier_origins(frame, holidays):
    k = 700
    changed = frame.copy()
    changed.iloc[k, changed.columns.get_loc("traffic_volume")] += 1000.0
    changed.iloc[k, changed.columns.get_loc("temp")] += 5.0
    base = build_features(frame, HORIZON, holidays)
    alt = build_features(changed, HORIZON, holidays)
    pd.testing.assert_frame_equal(base.iloc[:k], alt.iloc[:k])
    assert not np.allclose(base.iloc[k].to_numpy(float), alt.iloc[k].to_numpy(float), equal_nan=True)


def test_target_is_traffic_at_origin_plus_horizon(frame):
    targets = build_targets(frame, HORIZON)
    assert targets.iloc[10] == frame["traffic_volume"].iloc[10 + HORIZON]
    assert targets.iloc[-HORIZON:].isna().all()


def test_changing_one_hour_only_moves_one_target(frame):
    k = 700
    changed = frame.copy()
    changed.iloc[k, changed.columns.get_loc("traffic_volume")] += 1000.0
    base = build_targets(frame, HORIZON).to_numpy()
    alt = build_targets(changed, HORIZON).to_numpy()
    assert list(np.flatnonzero(~np.isclose(base, alt, equal_nan=True))) == [k - HORIZON]
