"""Validation metrics for the Phase 1 gate.

The original gate was "mean IC consistently positive (0.03-0.05+) across all folds".
Directionally right, incomplete in ways that would let a useless model pass:

  * A mean IC of 0.04 with a t-statistic of 0.8 is noise wearing a number's clothes.
    IC_T_STAT is the criterion with actual teeth.
  * A GBDT that has merely rediscovered momentum would clear an absolute IC threshold
    while adding nothing over three lines of pandas. Hence baselines.py.

Everything here is computed per-date on the cross-section, then aggregated -- never pooled
across dates, which would let a few large-cross-section days dominate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICResult:
    """Information-coefficient statistics for one fold or period."""

    mean_ic: float
    std_ic: float
    t_stat: float
    ir: float                      # information ratio = mean/std
    n_dates: int
    hit_rate: float                # share of dates with positive IC
    ic_series: pd.Series = field(repr=False, default_factory=pd.Series)

    @property
    def passes_t_stat(self) -> bool:
        return self.t_stat >= 2.0

    def summary(self) -> str:
        return (f"IC {self.mean_ic:+.4f}  t={self.t_stat:+.2f}  IR={self.ir:+.3f}  "
                f"hit={self.hit_rate:.1%}  n={self.n_dates}")


def rank_ic(predictions: pd.DataFrame, labels: pd.DataFrame,
            min_names: int = 10) -> pd.Series:
    """Per-date Spearman rank IC between predictions and forward relative returns.

    Spearman rather than Pearson: we care about ordering, and fat-tailed return
    distributions make Pearson unstable. Dates with fewer than `min_names` valid pairs are
    dropped -- an IC computed on 4 names is noise.
    """
    common_dates = predictions.index.intersection(labels.index)
    out: dict[pd.Timestamp, float] = {}

    for dt in common_dates:
        p = predictions.loc[dt]
        y = labels.loc[dt]
        mask = p.notna() & y.notna()
        if mask.sum() < min_names:
            continue
        pv, yv = p[mask], y[mask]
        # A constant prediction has no ordering; spearmanr would return NaN.
        if pv.nunique() < 2 or yv.nunique() < 2:
            continue
        rho, _ = stats.spearmanr(pv, yv)
        if not np.isnan(rho):
            out[dt] = float(rho)

    return pd.Series(out, name="ic").sort_index()


def summarize_ic(ic: pd.Series) -> ICResult:
    """Aggregate a per-date IC series.

    The t-statistic assumes independent daily ICs. With overlapping 10-day labels that is
    optimistic, which is one more reason the gate demands t >= 2.0 rather than 1.65 --
    the threshold absorbs some of the autocorrelation the test ignores.
    """
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return ICResult(np.nan, np.nan, np.nan, np.nan, 0, np.nan, ic)

    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else np.nan
    t = mean / (std / np.sqrt(n)) if std and std > 0 else np.nan
    return ICResult(
        mean_ic=mean,
        std_ic=std,
        t_stat=float(t) if t == t else np.nan,
        ir=float(mean / std) if std and std > 0 else np.nan,
        n_dates=n,
        hit_rate=float((ic > 0).mean()),
        ic_series=ic,
    )


def quantile_spread(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    n_quantiles: int = 5,
    cost_bps: float = 10.0,
    min_names: int = 10,
) -> dict:
    """Top-minus-bottom quantile forward relative return, net of costs.

    This is the economic reading of the signal: the IC can be positive while the tradeable
    spread is not, if the relationship is concentrated in the middle of the distribution.

    `cost_bps` is a round-trip assumption applied to the long-short spread. It is recorded
    in the result so a report can never present a gross number as if it were net.
    """
    common = predictions.index.intersection(labels.index)
    rows = []

    for dt in common:
        p, y = predictions.loc[dt], labels.loc[dt]
        mask = p.notna() & y.notna()
        if mask.sum() < max(min_names, n_quantiles * 2):
            continue
        pv, yv = p[mask], y[mask]
        try:
            q = pd.qcut(pv.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        top = yv[q == n_quantiles - 1].mean()
        bot = yv[q == 0].mean()
        rows.append({"date": dt, "top": top, "bottom": bot, "spread": top - bot})

    if not rows:
        return {"n_dates": 0, "mean_spread_gross": np.nan, "mean_spread_net": np.nan,
                "t_stat": np.nan, "cost_bps": cost_bps}

    df = pd.DataFrame(rows).set_index("date")
    gross = float(df["spread"].mean())
    net = gross - cost_bps / 10_000.0
    sd = float(df["spread"].std(ddof=1))
    t = gross / (sd / np.sqrt(len(df))) if sd > 0 else np.nan

    return {
        "n_dates": len(df),
        "mean_spread_gross": gross,
        "mean_spread_net": net,
        "mean_top": float(df["top"].mean()),
        "mean_bottom": float(df["bottom"].mean()),
        "t_stat": float(t) if t == t else np.nan,
        "cost_bps": cost_bps,
        "spread_series": df["spread"],
    }


def shuffle_test(predictions: pd.DataFrame, labels: pd.DataFrame,
                 n_trials: int = 20, seed: int = 0) -> dict:
    """Permute labels WITHIN each date and re-measure IC.

    The pipeline's smoke alarm. Shuffling within a date destroys the prediction-label
    relationship while preserving every other structural property. If IC does not collapse
    to ~0, the signal is coming from the plumbing rather than from the market -- there is
    leakage somewhere upstream.
    """
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_trials):
        # pandas 3.0 returns a read-only view from to_numpy(); copy before shuffling.
        vals = np.array(labels.to_numpy(), dtype="float64", copy=True)
        for i in range(vals.shape[0]):
            row = vals[i]
            idx = np.flatnonzero(~np.isnan(row))
            if idx.size > 1:
                row[idx] = rng.permutation(row[idx])
        shuffled = pd.DataFrame(vals, index=labels.index, columns=labels.columns)
        means.append(summarize_ic(rank_ic(predictions, shuffled)).mean_ic)

    arr = np.array([m for m in means if m == m])
    return {
        "n_trials": int(arr.size),
        "mean_shuffled_ic": float(arr.mean()) if arr.size else np.nan,
        "std_shuffled_ic": float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
        "max_abs_shuffled_ic": float(np.abs(arr).max()) if arr.size else np.nan,
    }
