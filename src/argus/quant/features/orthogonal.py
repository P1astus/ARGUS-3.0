"""Features chosen to be economically distinct from the momentum block.

MOTIVATION (measured, not assumed)
The v1 feature set had an effective rank of 5.4 across 11 features, with two pairs at
rho = 1.00 exactly:

    momentum_20d == relative_strength_20d
    momentum_60d == relative_strength_60d

Not a coincidence -- `relative_strength` subtracted the cross-sectional mean, and rank
normalisation is invariant to subtracting a constant. Those two features were literal
duplicates. The model had far less independent information than the feature count implied.

Each family below is included because it captures a DIFFERENT mechanism, decided before
seeing any result. They are not a menu to try and filter -- adding candidates and keeping
whichever happen to score is how a pre-registered gate gets quietly gamed.

  sub-segment relative    memory and equipment do not cycle together; strength within
                          one's own sub-segment is not the same as strength within semis
  residual momentum       trend after removing sector beta -- who is moving on their own
  trend quality           how *smooth* a trend is, independent of how large it is
  risk-adjusted momentum  return per unit of volatility
  short-term reversal     opposite sign to momentum at short horizons
  intraday structure      uses open/high/low, which v1 fetched and then ignored entirely
  illiquidity             price impact per unit of volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from argus.quant.features.registry import register


# ------------------------------------------------------- sub-segment relative strength

def _group_demean(values: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Subtract each date's within-group mean.

    Unlike subtracting the whole cross-section's mean, this survives rank normalisation:
    the adjustment differs per name depending on its group, so it changes the ordering.
    """
    if not groups:
        return values
    grp = pd.Series({c: groups.get(c, "other") for c in values.columns})
    out = values.copy()
    for g in grp.unique():
        cols = grp.index[grp == g]
        cols = [c for c in cols if c in values.columns]
        if len(cols) < 2:
            continue
        block = values[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


@register("subsegment_rel_20d")
def subsegment_rel_20d(close: pd.DataFrame, groups: dict[str, str] | None = None, **_) -> pd.DataFrame:
    """20-day return minus the name's own sub-segment average.

    Isolates idiosyncratic strength from a sub-segment-wide move -- a memory name up 8%
    while all memory is up 8% is not strong, it is carried.
    """
    r = close / close.shift(20) - 1.0
    return _group_demean(r, groups or {})


@register("subsegment_rel_60d")
def subsegment_rel_60d(close: pd.DataFrame, groups: dict[str, str] | None = None, **_) -> pd.DataFrame:
    r = close / close.shift(60) - 1.0
    return _group_demean(r, groups or {})


# ------------------------------------------------------------------ residual momentum

@register("residual_momentum_60d")
def residual_momentum_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Momentum after removing each name's beta to the equal-weight sector.

    Rolling-window OLS of daily returns on the sector return; the feature is the sum of
    residuals. Distinguishes "up because the sector is up" from "up on its own", which
    raw momentum cannot.
    """
    ret = close.pct_change()
    sector = ret.mean(axis=1)

    win = 60
    cov = ret.rolling(win, min_periods=40).cov(sector)
    var = sector.rolling(win, min_periods=40).var()
    beta = cov.div(var, axis=0)

    resid = ret.sub(beta.mul(sector, axis=0))
    return resid.rolling(win, min_periods=40).sum()


@register("beta_60d")
def beta_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Sensitivity to the sector. High-beta names behave differently in drawdowns."""
    ret = close.pct_change()
    sector = ret.mean(axis=1)
    cov = ret.rolling(60, min_periods=40).cov(sector)
    var = sector.rolling(60, min_periods=40).var()
    return cov.div(var, axis=0)


# --------------------------------------------------------------------- trend quality

@register("trend_quality_60d")
def trend_quality_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """R-squared of a linear fit to log price over 60 days, signed by direction.

    Magnitude-independent: a stock grinding steadily higher and one that jumped once on
    an acquisition rumour can share a 60-day return but differ completely here. Swing
    setups favour the former.
    """
    logp = np.log(close.where(close > 0))
    win = 60
    x = np.arange(win, dtype="float64")
    x_dm = x - x.mean()
    denom_x = (x_dm ** 2).sum()

    def _r2_signed(col: np.ndarray) -> float:
        if np.isnan(col).any():
            return np.nan
        y_dm = col - col.mean()
        slope = (x_dm * y_dm).sum() / denom_x
        pred = slope * x_dm
        ss_res = ((y_dm - pred) ** 2).sum()
        ss_tot = (y_dm ** 2).sum()
        if ss_tot <= 0:
            return np.nan
        r2 = 1.0 - ss_res / ss_tot
        return float(np.sign(slope) * r2)

    return logp.rolling(win, min_periods=win).apply(_r2_signed, raw=True)


@register("risk_adjusted_momentum_60d")
def risk_adjusted_momentum_60d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """60-day return divided by realised volatility -- trend per unit of risk."""
    r = close / close.shift(60) - 1.0
    vol = close.pct_change().rolling(60, min_periods=40).std() * np.sqrt(252)
    return r.div(vol.replace(0.0, np.nan))


# ------------------------------------------------------------------ short-term reversal

@register("reversal_5d")
def reversal_5d(close: pd.DataFrame, **_) -> pd.DataFrame:
    """Negated 5-day return.

    At swing horizons reversal and momentum compete. Included with the opposite sign so
    the model can express "recently oversold" as distinct from "weak".
    """
    return -(close / close.shift(5) - 1.0)


# ----------------------------------------------------------------- intraday structure
# v1 fetched open/high/low and used none of them.

@register("close_location_20d")
def close_location_20d(close: pd.DataFrame, high: pd.DataFrame | None = None,
                       low: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """Where the close sits within the day's range, averaged over 20 days.

    Near 1 means buyers consistently held into the close; near 0 means sellers did.
    A participation signal that price change alone does not capture.
    """
    if high is None or low is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    rng = (high - low).replace(0.0, np.nan)
    loc = (close - low) / rng
    return loc.rolling(20, min_periods=10).mean()


@register("intraday_range_20d")
def intraday_range_20d(close: pd.DataFrame, high: pd.DataFrame | None = None,
                       low: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """Average daily high-low range as a fraction of close.

    A volatility measure built from intraday extremes rather than close-to-close, so it
    captures churn that close-based volatility misses.
    """
    if high is None or low is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    return ((high - low) / close.replace(0.0, np.nan)).rolling(20, min_periods=10).mean()


@register("overnight_gap_20d")
def overnight_gap_20d(close: pd.DataFrame, open_: pd.DataFrame | None = None, **_) -> pd.DataFrame:
    """Mean overnight return (prior close -> open) over 20 days.

    Overnight moves carry news and earnings reactions; intraday moves carry flow. The
    split is informative and invisible to close-to-close features.
    """
    if open_ is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    gap = open_ / close.shift(1) - 1.0
    return gap.rolling(20, min_periods=10).mean()


# ----------------------------------------------------------------------- illiquidity

@register("amihud_illiquidity_60d")
def amihud_illiquidity_60d(close: pd.DataFrame, volume: pd.DataFrame | None = None,
                           **_) -> pd.DataFrame:
    """Amihud measure: |daily return| per dollar traded.

    How much price moves per unit of volume. Distinct from the liquidity *screen*, which
    is a yes/no filter; this is a continuous measure of price impact, and impact predicts
    how a swing entry will actually fill.
    """
    if volume is None:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    dv = (close * volume).replace(0.0, np.nan)
    illiq = (close.pct_change().abs() / dv) * 1e9
    return illiq.rolling(60, min_periods=30).mean()


# v2 feature set: the non-duplicated momentum block plus the families above.
# relative_strength_{20,60}d are DROPPED -- they were identical to momentum_{20,60}d
# after rank normalisation. dollar_volume_trend is dropped as ~0.94 with volume_trend.
FEATURES_V2: tuple[str, ...] = (
    # momentum block (deduplicated)
    "momentum_12_1",
    "momentum_20d",
    "momentum_60d",
    # volatility / participation
    "volatility_20d",
    "volatility_regime",
    "volume_trend",
    "distance_from_high_252d",
    # orthogonal additions
    "subsegment_rel_20d",
    "subsegment_rel_60d",
    "residual_momentum_60d",
    "beta_60d",
    "trend_quality_60d",
    "risk_adjusted_momentum_60d",
    "reversal_5d",
    "close_location_20d",
    "intraday_range_20d",
    "overnight_gap_20d",
    "amihud_illiquidity_60d",
)
