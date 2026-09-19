"""Feature registry and cross-sectional normalisation.

Two rules hold for every feature here, and both exist to prevent leakage rather than to
tidy the code:

  1. A feature at date t may use data up to and including t, never beyond. Features are
     computed with backward-looking windows only.

  2. Normalisation is ALWAYS within-date. Fitting a scaler over the full sample would let
     the model infer the regime -- a z-score computed against the whole history encodes
     where a date sits relative to periods that had not happened yet. This is the subtlest
     leak in the pipeline and the easiest to introduce by accident.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# name -> fn(close, volume, high) -> wide frame aligned to close
FeatureFn = Callable[..., pd.DataFrame]
_REGISTRY: dict[str, FeatureFn] = {}


def register(name: str) -> Callable[[FeatureFn], FeatureFn]:
    def deco(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> FeatureFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# ------------------------------------------------------------------ normalisation

def cross_sectional_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Z-score within each date, clipped to limit outlier leverage.

    Clipping matters for GBDTs less than for linear models, but an untrimmed 20-sigma
    move still lets one name dominate a split.
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank within each date.

    Preferred default: immune to outliers and to the level shifts that make raw values
    regime-dependent.
    """
    return df.rank(axis=1, pct=True, na_option="keep")


def normalize(df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    if method == "rank":
        return cross_sectional_rank(df)
    if method == "zscore":
        return cross_sectional_zscore(df)
    if method == "none":
        return df
    raise ValueError(f"unknown normalisation {method!r}")


# ---------------------------------------------------------------------- features

@register("momentum_12_1")
def momentum_12_1(close: pd.DataFrame, **_) -> pd.DataFrame:
    """12-month return skipping the most recent month.

    The classic cross-sectional momentum factor. The one-month skip avoids short-term
    reversal, which otherwise contaminates the signal.
    """
    return close.shift(21) / close.shift(252) - 1.0


@register("momentum_20d")
def momentum_20d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Recent one-month return -- the swing-horizon momentum term."""
    return close / close.shift(20) - 1.0


@register("momentum_60d")
def momentum_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    return close / close.shift(60) - 1.0


@register("relative_strength_20d")
def relative_strength_20d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """20-day return minus the equal-weight sector return over the same window.

    Sector-relative by construction, so it isolates idiosyncratic strength from a rising
    tide -- which is what a relative-return label rewards.
    """
    r = close / close.shift(20) - 1.0
    return r.sub(r.mean(axis=1), axis=0)


@register("relative_strength_60d")
def relative_strength_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    r = close / close.shift(60) - 1.0
    return r.sub(r.mean(axis=1), axis=0)


@register("volatility_20d")
def volatility_20d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Annualised realised volatility."""
    return close.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)


@register("volatility_regime")
def volatility_regime(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Short-window vol relative to its own long-window average.

    Above 1 means the name is currently more volatile than its own norm -- a compression/
    expansion signal that a raw vol level cannot express, since it is self-referential
    rather than cross-sectional.
    """
    ret = close.pct_change()
    short = ret.rolling(20, min_periods=10).std()
    long = ret.rolling(120, min_periods=60).std()
    return short / long.replace(0.0, np.nan)


@register("volume_trend")
def volume_trend(close: pd.DataFrame, volume: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """20-day average volume over its 60-day average.

    Rising participation alongside price strength is the confirmation a swing setup wants.
    """
    if volume is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    short = volume.rolling(20, min_periods=10).mean()
    long = volume.rolling(60, min_periods=30).mean()
    return short / long.replace(0.0, np.nan)


@register("dollar_volume_trend")
def dollar_volume_trend(close: pd.DataFrame, volume: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    if volume is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    dv = close * volume
    return dv.rolling(20, min_periods=10).mean() / dv.rolling(60, min_periods=30).mean().replace(0.0, np.nan)


@register("distance_from_high_252d")
def distance_from_high_252d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Drawdown from the trailing 52-week high (negative = below).

    Rolling, not expanding: an expanding max would use the whole history including periods
    after t once the frame is reused, and the 52-week high is the level traders actually
    watch.
    """
    high = close.rolling(252, min_periods=60).max()
    return close / high - 1.0


@register("distance_from_high_60d")
def distance_from_high_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    high = close.rolling(60, min_periods=20).max()
    return close / high - 1.0


DEFAULT_FEATURES: tuple[str, ...] = (
    "momentum_12_1",
    "momentum_20d",
    "momentum_60d",
    "relative_strength_20d",
    "relative_strength_60d",
    "volatility_20d",
    "volatility_regime",
    "volume_trend",
    "dollar_volume_trend",
    "distance_from_high_252d",
    "distance_from_high_60d",
)
