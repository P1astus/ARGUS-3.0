"""Panel assembly: prices -> normalised features + labels, ready for the model.

The one invariant worth restating: every feature is normalised WITHIN each date before it
reaches the model. Nothing here ever fits a transform across dates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from argus.data.prices import PriceStore, dollar_volume
from argus.quant.features import registry as feat
from argus.quant.labels import LabelSpec, relative_labels
from argus.quant.universe import UniverseSpec, coverage_report, liquidity_mask

log = logging.getLogger(__name__)


@dataclass
class Panel:
    """Everything the model and the validator need, aligned on one date axis."""

    features: pd.DataFrame          # long: MultiIndex (date, ticker) -> feature columns
    labels: pd.Series               # long: MultiIndex (date, ticker) -> forward rel return
    close: pd.DataFrame             # wide, for baselines
    coverage: pd.DataFrame
    label_spec: LabelSpec
    feature_names: list[str] = field(default_factory=list)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.features.index.get_level_values("date").unique()).sort_values()

    def wide_labels(self) -> pd.DataFrame:
        return self.labels.unstack("ticker")

    def slice_dates(self, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
        """Rows for the given dates, in the order the model expects."""
        mask = self.features.index.get_level_values("date").isin(dates)
        return self.features[mask], self.labels[mask]


def build_panel(
    store: PriceStore,
    spec: UniverseSpec,
    end: date,
    label_spec: LabelSpec = LabelSpec(),
    feature_names: tuple[str, ...] = feat.DEFAULT_FEATURES,
    normalization: str = "rank",
    rebalance: str = "W-FRI",
) -> Panel:
    """Assemble the modelling panel.

    Args:
        rebalance: resample frequency for the sampling dates. Weekly by default -- daily
            sampling with a 10-day label means 9/10 of consecutive rows share label
            information, which inflates apparent sample size without adding independent
            observations.
    """
    symbols = spec.symbols
    log.info("fetching %d tickers from %s", len(symbols), spec.start_date)

    close = store.panel(symbols, spec.start_date, end, field="adj_close")
    volume = store.panel(symbols, spec.start_date, end, field="volume")
    if close.empty:
        raise RuntimeError("no price data returned for any configured ticker")

    volume = volume.reindex(index=close.index, columns=close.columns)

    # OHLC carries intraday structure (where the close sat in the day's range, overnight
    # gaps) that close-to-close features cannot see. v1 fetched these and used none of
    # them. Adjusted to the same basis as adj_close so the ratios stay consistent across
    # splits -- an unadjusted high against an adjusted close would be meaningless.
    raw_close = store.panel(symbols, spec.start_date, end, field="close")
    raw_close = raw_close.reindex(index=close.index, columns=close.columns)
    adj_factor = (close / raw_close.replace(0.0, np.nan))

    high = store.panel(symbols, spec.start_date, end, field="high")
    low = store.panel(symbols, spec.start_date, end, field="low")
    open_ = store.panel(symbols, spec.start_date, end, field="open")
    high = high.reindex(index=close.index, columns=close.columns) * adj_factor
    low = low.reindex(index=close.index, columns=close.columns) * adj_factor
    open_ = open_.reindex(index=close.index, columns=close.columns) * adj_factor

    # Liquidity screen before features, so excluded names never enter a cross-section.
    dvol = dollar_volume(close, volume, window=60)
    tradeable = liquidity_mask(close, dvol, spec)
    close_ok = close.where(tradeable)

    cov = coverage_report(close_ok, spec)

    # Features on the screened panel. Sub-segment map is passed through so that
    # "relative" features can be relative to a name's OWN sub-segment -- demeaning against
    # the whole cross-section is a no-op under rank normalisation, which is why v1's
    # relative_strength features were identical to plain momentum.
    groups = {t: spec.tickers[t].value for t in close_ok.columns if t in spec.tickers}
    tradeable_vol = volume.where(tradeable)

    raw: dict[str, pd.DataFrame] = {}
    for name in feature_names:
        fn = feat.get(name)
        raw[name] = fn(
            close=close_ok,
            volume=tradeable_vol,
            high=high.where(tradeable),
            low=low.where(tradeable),
            open_=open_.where(tradeable),
            groups=groups,
        )

    normed = {n: feat.normalize(df, normalization) for n, df in raw.items()}

    rel, _ = relative_labels(close_ok, label_spec)

    # Restrict to rebalance dates AFTER features/labels are computed on the full daily
    # series -- subsampling first would corrupt every rolling window.
    rebal_dates = close_ok.resample(rebalance).last().index
    rebal_dates = pd.DatetimeIndex([d for d in rebal_dates if d in close_ok.index])

    long_feats = []
    for name, df in normed.items():
        s = df.loc[rebal_dates].stack(future_stack=True)
        s.name = name
        long_feats.append(s)

    X = pd.concat(long_feats, axis=1)
    X.index = X.index.set_names(["date", "ticker"])

    y = rel.loc[rebal_dates].stack(future_stack=True)
    y.index = y.index.set_names(["date", "ticker"])
    y.name = "fwd_rel_return"

    # Keep only rows with a label and at least one usable feature.
    both = X.join(y, how="inner").dropna(subset=["fwd_rel_return"])
    both = both[both[list(feature_names)].notna().any(axis=1)]

    X_final = both[list(feature_names)]
    y_final = both["fwd_rel_return"]

    log.info("panel: %d rows, %d dates, %d features",
             len(X_final), X_final.index.get_level_values("date").nunique(), len(feature_names))

    return Panel(
        features=X_final,
        labels=y_final,
        close=close_ok,
        coverage=cov,
        label_spec=label_spec,
        feature_names=list(feature_names),
    )
