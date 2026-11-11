from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.features.origin import FEATURE_COLUMNS, build_features, build_targets
from forecaster.models.quantile import fit_quantile_models, predict_quantiles


def test_quantile_predictions_are_ordered_on_a_learnable_signal(fast_params):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(800, 3)), columns=["a", "b", "c"])
    y = pd.Series(3 * X["a"] + rng.normal(scale=0.5, size=800))
    models = fit_quantile_models(X, y, [0.05, 0.5, 0.95], fast_params)
    pred = predict_quantiles(models, X)
    assert list(pred.columns) == [0.05, 0.5, 0.95]
    assert (pred[0.05].mean() < pred[0.5].mean()) and (pred[0.5].mean() < pred[0.95].mean())


def test_rows_with_missing_targets_are_dropped_during_fit(fast_params, frame, holidays):
    X = build_features(frame, 24, holidays)
    y = build_targets(frame, 24)
    models = fit_quantile_models(X.iloc[:600], y.iloc[:600], [0.5], fast_params)
    assert set(models) == {0.5}
    assert list(X.columns) == list(FEATURE_COLUMNS)
