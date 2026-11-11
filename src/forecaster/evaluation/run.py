from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.evaluation.backtest import Fold, rolling_origin_folds
from forecaster.evaluation.metrics import coverage, mae, mean_width, pinball, seasonal_scale
from forecaster.features.origin import FEATURE_COLUMNS, build_features, build_targets
from forecaster.models.baselines import seasonal_naive_ffill
from forecaster.models.conformal import AdaptiveConformal, conformal_quantile, cqr_interval, cqr_scores
from forecaster.models.quantile import fit_quantile_models, predict_quantiles

MEDIAN = 0.5
DAILY = 24
ACI_LEVEL = 0.9
LEVEL_TAUS: dict[float, tuple[float, float]] = {
    0.5: (0.25, 0.75),
    0.8: (0.1, 0.9),
    0.9: (0.05, 0.95),
    0.95: (0.025, 0.975),
}
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": FEATURE_COLUMNS,
    "no_weather": tuple(c for c in FEATURE_COLUMNS if not c.startswith("origin_")),
    "no_traffic_history": tuple(c for c in FEATURE_COLUMNS if not c.startswith("traffic_")),
    "calendar_only": tuple(c for c in FEATURE_COLUMNS if c.startswith("target_")),
}


def _adaptive_intervals(
    lo: np.ndarray,
    hi: np.ndarray,
    outcomes: np.ndarray,
    horizon: int,
    calibration_scores: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ACI over test origins in time order.

    At origin k, the outcome of origin k - horizon (traffic at k) is already observed, so it is the
    only realised outcome available for updating the miscoverage level.
    """
    aci = AdaptiveConformal(1 - ACI_LEVEL, calibration_scores, gamma=gamma)
    count = len(lo)
    out_lo = np.empty(count)
    out_hi = np.empty(count)
    threshold_used = np.empty(count)
    for k in range(count):
        if k >= horizon and np.isfinite(outcomes[k]):
            prev = k - horizon
            score = float(cqr_scores(outcomes[k : k + 1], lo[prev : prev + 1], hi[prev : prev + 1])[0])
            aci.update(score, threshold_used[prev])
        threshold_used[k] = aci.threshold()
        out_lo[k], out_hi[k] = cqr_interval(lo[k], hi[k], threshold_used[k])
    return out_lo, out_hi


def _run_fold(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    Y: pd.Series,
    fold: Fold,
    horizon: int,
    cal_fraction: float,
    params: dict | None,
    gamma: float,
) -> pd.DataFrame:
    """Forecast one fold. Training is split into a proper-training block and a later calibration block."""
    cal_start = int(fold.train_end * (1 - cal_fraction))
    proper_end = cal_start - horizon
    if proper_end < 1:
        raise ValueError("training window too short for a proper/calibration split")

    taus = sorted({MEDIAN, *(tau for pair in LEVEL_TAUS.values() for tau in pair)})
    models = fit_quantile_models(X.iloc[:proper_end], Y.iloc[:proper_end], taus, params)

    cal = slice(cal_start, fold.train_end)
    cal_q = predict_quantiles(models, X.iloc[cal])
    y_cal = Y.iloc[cal].to_numpy()
    observed = np.isfinite(y_cal)

    thresholds: dict[float, float] = {}
    scores_by_level: dict[float, np.ndarray] = {}
    for level, (lo_tau, hi_tau) in LEVEL_TAUS.items():
        scores = cqr_scores(
            y_cal[observed], cal_q[lo_tau].to_numpy()[observed], cal_q[hi_tau].to_numpy()[observed]
        )
        scores_by_level[level] = scores
        thresholds[level] = conformal_quantile(scores, 1 - level)

    test = slice(fold.test_start, fold.test_end)
    test_q = predict_quantiles(models, X.iloc[test])
    origins = frame.index[test]
    out = pd.DataFrame(index=origins)
    out["target_time"] = origins + pd.Timedelta(hours=horizon)
    out["y"] = Y.iloc[test].to_numpy()
    out["median"] = test_q[MEDIAN].to_numpy()
    out["seasonal"] = seasonal_naive_ffill(frame["traffic_volume"], DAILY).iloc[test].to_numpy()
    out["scale"] = seasonal_scale(frame["traffic_volume"].iloc[: fold.test_start].to_numpy(), DAILY)

    for level, (lo_tau, hi_tau) in LEVEL_TAUS.items():
        lo = np.minimum(test_q[lo_tau], test_q[hi_tau]).to_numpy()
        hi = np.maximum(test_q[lo_tau], test_q[hi_tau]).to_numpy()
        out[f"raw_lo_{level}"] = lo
        out[f"raw_hi_{level}"] = hi
        cqr_lo, cqr_hi = cqr_interval(lo, hi, thresholds[level])
        out[f"cqr_lo_{level}"] = cqr_lo
        out[f"cqr_hi_{level}"] = cqr_hi

    aci_lo, aci_hi = _adaptive_intervals(
        out[f"raw_lo_{ACI_LEVEL}"].to_numpy(),
        out[f"raw_hi_{ACI_LEVEL}"].to_numpy(),
        frame["traffic_volume"].iloc[test].to_numpy(float),
        horizon,
        scores_by_level[ACI_LEVEL],
        gamma,
    )
    out["aci_lo"] = aci_lo
    out["aci_hi"] = aci_hi
    return out


def evaluate_backtest(
    frame: pd.DataFrame,
    horizon: int,
    holidays: pd.DatetimeIndex,
    n_folds: int,
    test_len: int,
    cal_fraction: float = 0.2,
    params: dict | None = None,
    gamma: float = 0.01,
) -> pd.DataFrame:
    """Rolling-origin forecasts with intervals. Returns one row per test origin per fold."""
    if horizon != DAILY:
        raise ValueError("the seasonal baseline and MASE scale are defined for a 24-hour horizon")
    X = build_features(frame, horizon, holidays)
    Y = build_targets(frame, horizon)
    n_origins = len(frame) - horizon
    parts = []
    for index, fold in enumerate(rolling_origin_folds(n_origins, test_len, n_folds, gap=horizon)):
        part = _run_fold(frame, X.iloc[:n_origins], Y.iloc[:n_origins], fold, horizon, cal_fraction, params, gamma)
        part["fold"] = index
        parts.append(part)
    return pd.concat(parts)


def summarize_points(results: pd.DataFrame) -> pd.DataFrame:
    observed = results[results["y"].notna()]
    rows = []
    for name, column in (("lightgbm_median", "median"), ("seasonal_naive_ffill", "seasonal")):
        fold_mase = []
        for _, part in observed.groupby("fold"):
            fold_mase.append(mae(part["y"], part[column]) / part["scale"].iloc[0])
        rows.append(
            {
                "model": name,
                "mae": mae(observed["y"], observed[column]),
                "mase": float(np.nanmean(fold_mase)),
                "n_observed": int(observed["y"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_intervals(results: pd.DataFrame) -> pd.DataFrame:
    observed = results[results["y"].notna()]
    specs = []
    for level, (lo_tau, hi_tau) in LEVEL_TAUS.items():
        specs.append(("quantile_raw", level, f"raw_lo_{level}", f"raw_hi_{level}", lo_tau, hi_tau))
        specs.append(("cqr_split", level, f"cqr_lo_{level}", f"cqr_hi_{level}", lo_tau, hi_tau))
    lo_tau, hi_tau = LEVEL_TAUS[ACI_LEVEL]
    specs.append(("aci", ACI_LEVEL, "aci_lo", "aci_hi", lo_tau, hi_tau))

    rows = []
    for method, level, lo_col, hi_col, lo_tau, hi_tau in specs:
        fold_cov = [
            coverage(part["y"], part[lo_col], part[hi_col]) for _, part in observed.groupby("fold")
        ]
        rows.append(
            {
                "method": method,
                "nominal": level,
                "coverage": coverage(observed["y"], observed[lo_col], observed[hi_col]),
                "coverage_fold_min": float(min(fold_cov)),
                "coverage_fold_max": float(max(fold_cov)),
                "mean_width": mean_width(observed[lo_col], observed[hi_col]),
                "pinball_pair": 0.5
                * (
                    pinball(observed["y"], observed[lo_col], lo_tau)
                    + pinball(observed["y"], observed[hi_col], hi_tau)
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_groups(
    frame: pd.DataFrame,
    horizon: int,
    holidays: pd.DatetimeIndex,
    n_folds: int,
    test_len: int,
    params: dict | None = None,
    groups: dict[str, tuple[str, ...]] = FEATURE_GROUPS,
) -> pd.DataFrame:
    """Median-model ablation. Each group trains on all observed targets before its fold's test window."""
    if horizon != DAILY:
        raise ValueError("the MASE scale is defined for a 24-hour horizon")
    n_origins = len(frame) - horizon
    X = build_features(frame, horizon, holidays).iloc[:n_origins]
    Y = build_targets(frame, horizon).iloc[:n_origins]
    traffic = frame["traffic_volume"].to_numpy(float)
    folds = list(rolling_origin_folds(n_origins, test_len, n_folds, gap=horizon))

    rows = []
    for name, columns in groups.items():
        cols = list(columns)
        fold_errors, fold_scales, fold_abs = [], [], []
        for fold in folds:
            models = fit_quantile_models(X.iloc[: fold.train_end][cols], Y.iloc[: fold.train_end], [MEDIAN], params)
            test = slice(fold.test_start, fold.test_end)
            pred = predict_quantiles(models, X.iloc[test][cols])[MEDIAN].to_numpy()
            actual = Y.iloc[test].to_numpy()
            fold_abs.append(mae(actual, pred))
            fold_scales.append(seasonal_scale(traffic[: fold.test_start], DAILY))
        rows.append(
            {
                "group": name,
                "n_features": len(cols),
                "mae": float(np.nanmean(fold_abs)),
                "mase": float(np.nanmean(np.array(fold_abs) / np.array(fold_scales))),
            }
        )
    return pd.DataFrame(rows)
