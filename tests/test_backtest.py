from __future__ import annotations

import pytest

from forecaster.evaluation.backtest import rolling_origin_folds


def test_folds_keep_gap_and_cover_the_tail():
    folds = list(rolling_origin_folds(n=500, horizon=50, n_folds=3, gap=24))
    assert folds[-1].test_end == 500
    for fold in folds:
        assert fold.test_end - fold.test_start == 50
        assert fold.train_end == fold.test_start - 24
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end <= later.test_start


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        list(rolling_origin_folds(n=100, horizon=50, n_folds=3))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        list(rolling_origin_folds(n=100, horizon=0, n_folds=1))
