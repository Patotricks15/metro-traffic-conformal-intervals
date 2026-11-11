from __future__ import annotations

import numpy as np
import pytest

from forecaster.models.conformal import AdaptiveConformal, conformal_quantile, cqr_interval, cqr_scores


def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.arange(1, 11, dtype=float)
    assert conformal_quantile(scores, 0.1) == 10.0
    assert conformal_quantile(scores, 0.5) == 6.0


def test_conformal_quantile_is_infinite_when_too_few_scores():
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.1) == float("inf")


def test_conformal_quantile_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([1.0]), 0.0)


def test_cqr_score_is_negative_inside_band_and_positive_outside():
    assert cqr_scores(np.array([5.0]), np.array([2.0]), np.array([8.0]))[0] == -3.0
    assert cqr_scores(np.array([10.0]), np.array([2.0]), np.array([8.0]))[0] == 2.0


def test_crossed_quantiles_keep_the_full_band():
    lo, hi = cqr_interval(np.array([8.0]), np.array([2.0]), 0.0)
    assert (lo[0], hi[0]) == (2.0, 8.0)
    score = cqr_scores(np.array([5.0]), np.array([8.0]), np.array([2.0]))[0]
    assert score == -3.0


def test_aci_lowers_alpha_after_a_miss_and_raises_it_after_a_hit():
    aci = AdaptiveConformal(0.1, np.arange(100, dtype=float) / 100, gamma=0.1)
    aci.update(10.0, aci.threshold())
    assert aci.alpha == pytest.approx(0.01)
    aci.update(-1.0, aci.threshold())
    assert aci.alpha == pytest.approx(0.02)


def test_aci_alpha_stays_within_bounds():
    aci = AdaptiveConformal(0.1, np.arange(100, dtype=float), gamma=1.0, alpha_bounds=(0.01, 0.5))
    for _ in range(20):
        aci.update(1e9, aci.threshold())
    assert aci.alpha == pytest.approx(0.01)
