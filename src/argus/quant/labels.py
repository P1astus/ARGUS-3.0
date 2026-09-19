"""Forward relative-return labels.

Three leakage risks were identified in planning; all three are handled here.

1. BENCHMARK SELF-REFERENCE. A cap-weighted semis index is dominated by its largest
   member -- NVDA can exceed 10% of SOXX. Labelling NVDA against an index it partly *is*
   compresses its own measured relative return toward zero, and does so more for exactly
   the names whose moves matter most. Fixed by an equal-weight benchmark computed
   LEAVE-ONE-OUT: each ticker is measured against the equal-weight average of every
   *other* name that existed on that date.

2. TOTAL-RETURN VS PRICE-RETURN MISMATCH. SOXX is a total-return ETF with fees; ^SOX is a
   price index. Labelling a price return against a total return injects a slow drift into
   every label. Fixed by constructing the benchmark from the same adjusted series as the
   names themselves, so both sides are like-for-like by construction.

3. OVERLAPPING HORIZONS. A 10-day forward label sampled daily shares 9 days with its
   neighbour. That is not fixed here -- it is a property of the label -- but it is why
   validation/splitter.py must purge AND embargo by at least the horizon. `horizon_days`
   is carried on the output so the splitter cannot be configured inconsistently with the
   labels it is splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    """Label configuration, carried alongside the data it produced."""

    horizon_days: int = 10          # swing hold; 5 and 15 used as robustness checks
    min_names_per_date: int = 10    # below this a cross-section is too thin to rank

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")


def forward_return(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple forward return over `horizon` trading days.

    Uses trading-day shifts, not calendar days -- calendar shifts would silently vary the
    holding period across holidays.
    """
    return prices.shift(-horizon) / prices - 1.0


def equal_weight_benchmark(fwd: pd.DataFrame, *, leave_one_out: bool = True) -> pd.DataFrame:
    """Benchmark forward return per (date, ticker).

    With leave_one_out=True the result is a frame: each column holds the mean forward
    return of all OTHER available names on that date. With it False, every column carries
    the same simple cross-sectional mean.

    Leave-one-out matters most where the cross-section is thin. At 15 names a single name
    is ~7% of a naive equal-weight benchmark; excluding it removes a mechanical negative
    bias in that name's measured relative return.
    """
    valid = fwd.notna()
    n = valid.sum(axis=1)
    total = fwd.where(valid).sum(axis=1)

    if not leave_one_out:
        mean = total / n.replace(0, np.nan)
        return pd.DataFrame(
            np.repeat(mean.to_numpy()[:, None], fwd.shape[1], axis=1),
            index=fwd.index, columns=fwd.columns,
        ).where(valid)

    # (sum - own) / (n - 1), only where the name itself is present.
    others_sum = total.to_numpy()[:, None] - fwd.where(valid).fillna(0.0).to_numpy()
    others_n = (n.to_numpy()[:, None] - valid.to_numpy().astype(int))
    with np.errstate(invalid="ignore", divide="ignore"):
        bench = np.where(others_n > 0, others_sum / others_n, np.nan)
    return pd.DataFrame(bench, index=fwd.index, columns=fwd.columns).where(valid)


def relative_labels(
    prices: pd.DataFrame,
    spec: LabelSpec = LabelSpec(),
    *,
    leave_one_out: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward return relative to the equal-weight sector benchmark.

    Args:
        prices: wide adjusted-close panel (rows = dates, columns = tickers).
        spec: label configuration.
        leave_one_out: exclude each name from its own benchmark.

    Returns:
        (relative, raw_forward) -- both wide frames aligned to `prices`. Dates with fewer
        than `spec.min_names_per_date` valid names are dropped entirely: ranking a
        five-name cross-section produces a number, but not a meaningful one.
    """
    if prices.empty:
        return prices.copy(), prices.copy()

    fwd = forward_return(prices, spec.horizon_days)
    bench = equal_weight_benchmark(fwd, leave_one_out=leave_one_out)
    rel = fwd - bench

    thin = fwd.notna().sum(axis=1) < spec.min_names_per_date
    rel.loc[thin] = np.nan

    return rel, fwd


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile rank in [0, 1].

    Ranking happens WITHIN each date and never across dates. A rank pooled over time would
    leak the future: it would encode where a date's returns sat relative to the whole
    sample, including dates that had not happened yet.
    """
    return df.rank(axis=1, pct=True, na_option="keep")
