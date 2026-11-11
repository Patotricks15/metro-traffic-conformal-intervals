from __future__ import annotations

from collections.abc import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}


def fit_quantile_models(
    X: pd.DataFrame, y: pd.Series, taus: Iterable[float], params: dict | None = None
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM quantile regressor per tau. Rows with a missing target are dropped."""
    keep = y.notna().to_numpy()
    settings = {**DEFAULT_PARAMS, **(params or {})}
    models: dict[float, lgb.LGBMRegressor] = {}
    for tau in taus:
        model = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), **settings)
        model.fit(X.loc[keep], y.loc[keep])
        models[float(tau)] = model
    return models


def predict_quantiles(models: dict[float, lgb.LGBMRegressor], X: pd.DataFrame) -> pd.DataFrame:
    """Column per tau. Crossings are left as predicted; the conformal layer orders each pair."""
    return pd.DataFrame({tau: model.predict(X) for tau, model in models.items()}, index=X.index)
