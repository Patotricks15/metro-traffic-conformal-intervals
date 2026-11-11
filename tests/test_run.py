from __future__ import annotations

import numpy as np
import pytest

from forecaster.evaluation.run import (
    FEATURE_GROUPS,
    evaluate_backtest,
    evaluate_feature_groups,
    summarize_intervals,
    summarize_points,
)

HORIZON = 24
N_FOLDS = 2
TEST_LEN = 48


def run_backtest(frame, holidays, params):
    return evaluate_backtest(frame, HORIZON, holidays, N_FOLDS, TEST_LEN, params=params)


def test_backtest_produces_ordered_intervals(frame, holidays, fast_params):
    results = run_backtest(frame, holidays, fast_params)
    assert set(results["fold"]) == {0, 1}
    assert len(results) == N_FOLDS * TEST_LEN
    assert (results["cqr_lo_0.9"] <= results["cqr_hi_0.9"]).all()
    assert (results["aci_lo"] <= results["aci_hi"]).all()
    assert np.isfinite(results[["median", "aci_lo", "aci_hi"]]).all().all()


def test_backtest_tolerates_missing_targets(frame, holidays, fast_params):
    frame = frame.copy()
    frame.iloc[300, frame.columns.get_loc("traffic_volume")] = np.nan
    results = run_backtest(frame, holidays, fast_params)
    points = summarize_points(results)
    assert set(points["model"]) == {"lightgbm_median", "seasonal_naive_ffill"}
    assert np.isfinite(points["mae"]).all()


def test_fold_forecasts_do_not_depend_on_later_traffic(frame, holidays, fast_params):
    base = run_backtest(frame, holidays, fast_params)
    fold0_end = frame.index.get_loc(base.loc[base["fold"] == 0].index.max()) + 1
    changed = frame.copy()
    changed.iloc[fold0_end:, changed.columns.get_loc("traffic_volume")] += 5000.0
    alt = run_backtest(changed, holidays, fast_params)
    columns = [c for c in base.columns if c != "y"]
    base0 = base.loc[base["fold"] == 0, columns]
    alt0 = alt.loc[alt["fold"] == 0, columns]
    np.testing.assert_allclose(base0.drop(columns=["target_time"]).to_numpy(float),
                               alt0.drop(columns=["target_time"]).to_numpy(float), equal_nan=True)


def test_summaries_report_expected_methods(frame, holidays, fast_params):
    results = run_backtest(frame, holidays, fast_params)
    intervals = summarize_intervals(results)
    assert {"quantile_raw", "cqr_split", "aci"} == set(intervals["method"])
    assert intervals["coverage"].between(0, 1).all()


def test_feature_group_ablation_covers_every_group(frame, holidays, fast_params):
    table = evaluate_feature_groups(frame, HORIZON, holidays, N_FOLDS, TEST_LEN, params=fast_params)
    assert list(table["group"]) == list(FEATURE_GROUPS)
    assert (table["n_features"] == [len(FEATURE_GROUPS[g]) for g in table["group"]]).all()


def test_non_daily_horizon_is_rejected(frame, holidays, fast_params):
    with pytest.raises(ValueError):
        evaluate_backtest(frame, 12, holidays, N_FOLDS, TEST_LEN, params=fast_params)
