"""Baseline predictors -- the control arms of the Phase 1 gate.

WHY THESE MATTER MORE THAN THE ABSOLUTE IC THRESHOLD
Cross-sectional momentum is a well-documented equity factor. A GBDT fed momentum features
will learn momentum, post a respectable IC, and clear any absolute threshold -- while
adding nothing whatsoever over three lines of pandas.

So the gate is not "IC >= 0.03". It is "IC >= 0.03 AND beats these". If the model cannot
beat a 20-day relative-strength sort, the honest conclusion is that the GBDT is
unnecessary and the sort is the model.

Each baseline is a pure function of the same price panel the model sees, so the comparison
is like-for-like: same universe, same dates, same labels, same purging.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from argus.quant.features.registry import cross_sectional_rank

BaselineFn = Callable[[pd.DataFrame], pd.DataFrame]


def momentum_12_1_baseline(close: pd.DataFrame) -> pd.DataFrame:
    """Classic 12-1 cross-sectional momentum, rank-normalised."""
    raw = close.shift(21) / close.shift(252) - 1.0
    return cross_sectional_rank(raw)


def relative_strength_20d_baseline(close: pd.DataFrame) -> pd.DataFrame:
    """20-day sector-relative return, rank-normalised.

    The most direct baseline: it is close to what the label measures, one horizon earlier.
    Beating it is the real test of whether the model has learned anything beyond
    persistence.
    """
    r = close / close.shift(20) - 1.0
    return cross_sectional_rank(r.sub(r.mean(axis=1), axis=0))


def short_term_reversal_baseline(close: pd.DataFrame) -> pd.DataFrame:
    """5-day reversal (negated recent return).

    Included because at swing horizons reversal and momentum compete; if reversal wins,
    the feature set is pointed the wrong way.
    """
    r = close / close.shift(5) - 1.0
    return cross_sectional_rank(-r.sub(r.mean(axis=1), axis=0))


def random_baseline(close: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Deterministic noise -- the floor.

    Should score ~0 IC. If it does not, the evaluation harness itself is broken, which is
    a different and worse problem than a weak model.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    vals = rng.random(close.shape)
    return pd.DataFrame(vals, index=close.index, columns=close.columns).where(close.notna())


BASELINES: dict[str, BaselineFn] = {
    "momentum_12_1": momentum_12_1_baseline,
    "relative_strength_20d": relative_strength_20d_baseline,
    "short_term_reversal": short_term_reversal_baseline,
    "random": random_baseline,
}

# Baselines the model must beat to clear the Phase 1 gate. `random` is a harness check,
# not a competitor, so it is excluded.
GATE_BASELINES: tuple[str, ...] = ("momentum_12_1", "relative_strength_20d")
