# Metro Traffic: Quantile Forecasts with Split and Adaptive Conformal Intervals

Day-ahead hourly forecasts of interstate traffic volume with calibrated prediction
intervals. Quantile LightGBM models produce raw bands, which are then corrected with
split conformalized quantile regression (CQR) and adaptive conformal inference (ACI).
All forecasts are produced in a strictly causal rolling-origin backtest.

## Dataset

- **Metro Interstate Traffic Volume** (UCI, mirrored on Kaggle as
  [`anshtanwar/metro-interstate-traffic-volume`](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)).
- Westbound I-94 at Minneapolis-St Paul, hourly, 2012-10-02 to 2018-09-30.
- License: **CC BY 4.0** (as stated on the Kaggle page). Attribution is required.
- Target: `traffic_volume`. Weather fields and holiday labels are used as covariates.

The file is not committed. Download it with the Kaggle API, passing the token inline
so it is never written to disk:

```bash
KAGGLE_API_TOKEN=<your-token> .venv/bin/kaggle datasets download -d anshtanwar/metro-interstate-traffic-volume -p data --unzip
```

## Cleaning policy

Implemented in [`metro.py`](src/forecaster/data/metro.py).

- Timestamps are day-first (`%d-%m-%Y %H:%M`) and parsed explicitly.
- Duplicate timestamps (about 7.6k rows) keep the **first** record. Traffic and
  holiday values are identical across duplicates in this export. Weather fields can
  differ, so the weather for those hours is a deterministic first-record choice.
- The series is reindexed to a complete hourly grid. About 23% of hours are absent
  in the export, concentrated in 2014 and 2015.
- **Targets are never imputed.** A missing target is dropped from training and
  from evaluation. Forecasts are still produced for those origins.
- `temp == 0` K is a missing-value code and is masked. `rain_1h > 100 mm/h` is a
  sensor error and is masked.

## Causal design (no leakage)

Implemented in [`origin.py`](src/forecaster/features/origin.py), with the feature
registry in [`features.yaml`](conf/features.yaml).

- Horizon `H = 24`. The origin `t` uses only information at or before `t` and
  predicts traffic at `t + 24`.
- Traffic lags at the origin (`0, 1, 2, 23, 144` hours) and trailing rolling mean and
  standard deviation over 24 and 168 hours.
- Weather is the observation at the origin. No weather forecast is used.
- Calendar features (hour, day of week, month, weekend, holiday) describe the target
  time. They are known in advance. Holiday dates come from the dataset's own labels,
  which are published ahead of time.

Tests enforce these rules:

- Truncated history yields identical features (parity).
- Changing a value at hour `k` alters features only at origins `>= k` (leakage).
- Changing a target moves exactly one training target.
- Fold forecasts do not change when traffic after the fold's test window changes.

### Validation scheme

- Rolling-origin folds with test windows of 672 origins (28 days) and `gap = H`, so
  training targets never overlap the test window.
- Within each fold's training data, the last 20% of origins form a calibration block.
  Proper training ends `H` origins before it, so no training target leaks into
  calibration.
- Conformal corrections use only calibration scores. ACI updates use only outcomes
  already observed at each origin.

## Models

- **Quantile LightGBM** for taus `0.025 ... 0.975` (one model per tau, 400 trees).
- **CQR split conformal** at nominal levels `0.5, 0.8, 0.9, 0.95`.
- **ACI** at level `0.9`, with `gamma = 0.01` and alpha clipped to `[0.001, 0.5]`.
- **Baselines:** seasonal naive (previous value at the same hour of day, causal
  forward-fill) and the LightGBM median.
- **MASE scale:** in-sample seasonal-naive MAE over history before each fold's test
  window.

## Results

Six folds, 4,026 observed test targets. Run with `.venv/bin/python scripts/run_backtest.py`
(about 1.5 minutes on a laptop CPU). Tables are written to `outputs/`, which is
git-ignored.

### Point forecasts

| model                | MAE    | MASE  |
|----------------------|--------|-------|
| lightgbm_median      | 186.1  | 0.324 |
| seasonal_naive_ffill | 523.6  | 0.910 |

### Interval calibration (test coverage, nominal in parentheses)

| method        | nominal | coverage | fold min | fold max | mean width | pinball (pair) |
|---------------|---------|----------|----------|----------|------------|----------------|
| quantile_raw  | 0.50    | 0.503    | 0.431    | 0.583    | 301.9      | 74.84          |
| cqr_split     | 0.50    | 0.544    | 0.474    | 0.624    | 318.5      | 74.93          |
| quantile_raw  | 0.80    | 0.782    | 0.700    | 0.871    | 611.0      | 45.11          |
| cqr_split     | 0.80    | 0.846    | 0.785    | 0.920    | 677.1      | 45.40          |
| quantile_raw  | 0.90    | 0.876    | 0.827    | 0.927    | 872.7      | 29.70          |
| cqr_split     | 0.90    | 0.930    | 0.894    | 0.967    | 985.9      | 29.93          |
| aci           | 0.90    | 0.906    | 0.891    | 0.920    | 951.8      | 30.13          |
| quantile_raw  | 0.95    | 0.923    | 0.884    | 0.967    | 1172.0     | 19.35          |
| cqr_split     | 0.95    | 0.970    | 0.951    | 0.987    | 1351.8     | 19.48          |

Raw quantile bands are slightly under-covered. CQR moves coverage towards nominal
and slightly over-covers at lower levels, at the cost of wider bands. ACI keeps
coverage close to 0.90 over the whole test period, with the narrowest fold-to-fold
spread among the 0.90 methods.

### Feature-group ablation (median model, MAE)

| group              | features | MAE   | MASE  |
|--------------------|----------|-------|-------|
| all                | 18       | 182.0 | 0.317 |
| no_weather         | 14       | 184.3 | 0.320 |
| no_traffic_history | 9        | 181.7 | 0.316 |
| calendar_only      | 5        | 173.8 | 0.302 |

The ablation trains on every observed target before each fold's test window. The
main run uses the proper-training split, so its MAE differs slightly. The ablation
is median-only, uses a single seed, and reports MAE only. Its differences are small.
The calendar-only model beats the full feature set here. Traffic history adds little
at a 24-hour horizon once the diurnal profile is known. This finding should be
checked with repeated seeds before drawing stronger conclusions.

## Scope

This project is a local Python package with tests and a GitHub Actions workflow
([`ci.yml`](.github/workflows/ci.yml)). It has no Terraform, no Floci or other cloud
infrastructure, no architecture diagrams, and no LaTeX report, as requested in the
portfolio plan.

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/check_neural_deps.py
.venv/bin/pytest
.venv/bin/python scripts/run_backtest.py
```
