"""Leakage tests.

These are first-class, not incidental. Every failure mode here produces a *plausible*
result rather than a crash -- an inflated IC that looks like signal and is not. That makes
them the only thing standing between a leaky pipeline and a model that backtests well and
loses money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.quant.labels import (
    LabelSpec,
    cross_sectional_rank,
    equal_weight_benchmark,
    forward_return,
    relative_labels,
)
from argus.quant.validation.splitter import PurgedWalkForward


@pytest.fixture
def panel() -> pd.DataFrame:
    """Deterministic 4-year daily panel, 20 names."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=1000, name="date")
    tickers = [f"T{i:02d}" for i in range(20)]
    rets = rng.normal(0.0004, 0.02, size=(len(dates), len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)


# --------------------------------------------------------------------------- splitter

def test_purge_and_embargo_prevent_label_overlap(panel):
    """The core guarantee: no training label window touches its test window."""
    for horizon in (5, 10, 15):
        cv = PurgedWalkForward(n_splits=5, horizon=horizon, min_train_dates=252)
        cv.assert_no_leakage(panel.index)  # raises on violation


def test_embargo_shorter_than_horizon_is_rejected():
    """A too-short embargo is the failure mode that makes leakage invisible."""
    with pytest.raises(ValueError, match="embargo"):
        PurgedWalkForward(n_splits=5, horizon=10, embargo=3)


def test_splitter_inherits_horizon_from_label_spec():
    """Splitter and labels must not be independently configurable."""
    spec = LabelSpec(horizon_days=15)
    cv = PurgedWalkForward.from_label_spec(spec, n_splits=5)
    assert cv.horizon == 15
    assert cv.embargo >= 15


def test_train_and_test_never_share_dates(panel):
    cv = PurgedWalkForward(n_splits=5, horizon=10, min_train_dates=252)
    for fold in cv.split(panel.index):
        assert not set(fold.train_idx) & set(fold.test_idx)


def test_purge_actually_removes_samples(panel):
    """A purge that silently removes nothing would pass the overlap test vacuously."""
    cv = PurgedWalkForward(n_splits=5, horizon=10, min_train_dates=252)
    folds = list(cv.split(panel.index))
    # Every fold except the first has training data before its test block to purge.
    assert all(f.n_purged > 0 for f in folds)
    # All but the last fold have data after the test block to embargo.
    assert sum(f.n_embargoed for f in folds) > 0


def test_folds_cover_the_tail_without_overlap(panel):
    cv = PurgedWalkForward(n_splits=5, horizon=10, min_train_dates=252)
    seen: set[int] = set()
    for fold in cv.split(panel.index):
        assert not seen & set(fold.test_idx), "test blocks must be disjoint"
        seen |= set(fold.test_idx)


# ----------------------------------------------------------------------------- labels

def test_forward_return_uses_future_prices_only(panel):
    """Sanity: label at t must be computable from t and t+h, nothing earlier."""
    h = 10
    fwd = forward_return(panel, h)
    t = 100
    expected = panel.iloc[t + h, 0] / panel.iloc[t, 0] - 1.0
    assert fwd.iloc[t, 0] == pytest.approx(expected)
    # The last h rows cannot have labels.
    assert fwd.iloc[-h:].isna().all().all()


def test_benchmark_excludes_own_name(panel):
    """Leave-one-out: a name must not appear in the benchmark it is measured against."""
    fwd = forward_return(panel, 10)
    bench = equal_weight_benchmark(fwd, leave_one_out=True)

    t = 200
    row = fwd.iloc[t]
    for col in fwd.columns[:5]:
        others = row.drop(col).dropna()
        assert bench.iloc[t][col] == pytest.approx(others.mean())


def test_leave_one_out_differs_from_naive_mean(panel):
    """Guards against leave_one_out silently degrading to the naive mean."""
    fwd = forward_return(panel, 10)
    loo = equal_weight_benchmark(fwd, leave_one_out=True)
    naive = equal_weight_benchmark(fwd, leave_one_out=False)
    assert not np.allclose(loo.iloc[200].to_numpy(), naive.iloc[200].to_numpy())


def test_relative_labels_sum_to_approximately_zero(panel):
    """With leave-one-out the cross-section is centred by construction."""
    rel, _ = relative_labels(panel, LabelSpec(horizon_days=10))
    row = rel.iloc[300].dropna()
    assert abs(row.mean()) < 1e-9


def test_thin_cross_sections_are_dropped():
    """Ranking a handful of names yields a number, not a signal."""
    dates = pd.bdate_range("2020-01-01", periods=100, name="date")
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B", "C"])
    prices += np.arange(len(dates))[:, None] * 0.1
    rel, _ = relative_labels(prices, LabelSpec(horizon_days=5, min_names_per_date=10))
    assert rel.isna().all().all()


# ---------------------------------------------------------------- cross-sectional norm

def test_rank_is_computed_within_date_not_pooled(panel):
    """Pooling ranks across time leaks the future into every row."""
    ranks = cross_sectional_rank(panel)
    for i in (0, 500, 999):
        row = ranks.iloc[i].dropna()
        assert row.min() > 0.0 and row.max() == pytest.approx(1.0)
        # A per-date percentile rank averages ~0.5 regardless of that date's level.
        assert row.mean() == pytest.approx(0.5, abs=0.05)


def test_rank_is_invariant_to_date_level_shifts(panel):
    """The decisive property: a rising market must not change within-date ranks.

    If normalisation were fitted globally, a regime shift would alter every row's values
    and the model could infer *when* it was -- leaking regime information.
    """
    ranks_a = cross_sectional_rank(panel)
    shifted = panel * np.linspace(1.0, 3.0, len(panel))[:, None]  # strong upward drift
    ranks_b = cross_sectional_rank(shifted)
    pd.testing.assert_frame_equal(ranks_a, ranks_b)
