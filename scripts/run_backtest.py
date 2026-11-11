"""Run the TS05 rolling-origin backtest and write result tables to outputs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from forecaster.data.metro import holiday_dates, load_raw, to_hourly
from forecaster.evaluation.run import (
    evaluate_backtest,
    evaluate_feature_groups,
    summarize_intervals,
    summarize_points,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 24
N_FOLDS = 6
TEST_LEN = 672


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "Metro_Interstate_Traffic_Volume.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.csv)
    frame = to_hourly(raw)
    holidays = holiday_dates(raw)

    results = evaluate_backtest(frame, HORIZON, holidays, N_FOLDS, TEST_LEN)
    points = summarize_points(results)
    intervals = summarize_intervals(results)
    groups = evaluate_feature_groups(frame, HORIZON, holidays, N_FOLDS, TEST_LEN)

    results.to_csv(args.out / "forecasts.csv")
    points.to_csv(args.out / "point_metrics.csv", index=False)
    intervals.to_csv(args.out / "interval_metrics.csv", index=False)
    groups.to_csv(args.out / "feature_group_ablation.csv", index=False)

    print("Point forecasts\n", points.to_string(index=False))
    print("\nIntervals\n", intervals.to_string(index=False))
    print("\nFeature-group ablation\n", groups.to_string(index=False))


if __name__ == "__main__":
    main()
