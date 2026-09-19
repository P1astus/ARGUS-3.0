"""Point-in-time universe construction.

Two jobs, and the second is the one that keeps the Phase 1 result honest:

  1. Resolve the configured ticker list and sub-segment map.
  2. Report, per date, how many names *should* have existed versus how many the provider
     can actually serve. With yfinance the gap is entirely delisted names, and since
     semiconductor exits skew towards acquisitions, that gap biases results upward. The
     number belongs in the validation report, not in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from argus.contracts.quant import SubSegment
from argus.data.delisted import delisted_as_of, existed_on


@dataclass(frozen=True)
class UniverseSpec:
    tickers: dict[str, SubSegment]
    pilot_tickers: frozenset[str]
    start_date: date
    min_price: float
    min_dollar_volume_60d: float

    @property
    def symbols(self) -> list[str]:
        return sorted(self.tickers)

    def sub_segment(self, ticker: str) -> SubSegment:
        return self.tickers[ticker]


def load_universe(path: str | Path) -> UniverseSpec:
    cfg = yaml.safe_load(Path(path).read_text())

    tickers: dict[str, SubSegment] = {}
    pilot: set[str] = set()
    for seg_name, entries in cfg["tickers"].items():
        seg = SubSegment(seg_name)
        for e in entries:
            sym = e["symbol"]
            # YAML 1.1 parses bare ON/OFF/YES/NO/TRUE/FALSE as booleans, which silently
            # turns the ticker "ON" (ON Semiconductor) into True. Symbols must be quoted
            # in the config; fail loudly here rather than several frames deeper.
            if not isinstance(sym, str):
                raise ValueError(
                    f"ticker symbol {sym!r} in section {seg_name!r} parsed as "
                    f"{type(sym).__name__}, not str -- quote it in the YAML "
                    '(e.g. {symbol: "ON"})')
            tickers[sym] = seg
            if e.get("pilot"):
                pilot.add(sym)

    f = cfg.get("filters", {})
    return UniverseSpec(
        tickers=tickers,
        pilot_tickers=frozenset(pilot),
        start_date=date.fromisoformat(str(cfg["start_date"])),
        min_price=float(f.get("min_price", 5.0)),
        min_dollar_volume_60d=float(f.get("min_dollar_volume_60d", 10_000_000)),
    )


def liquidity_mask(
    close: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    spec: UniverseSpec,
) -> pd.DataFrame:
    """Boolean frame: True where a name is tradeable on that date.

    Applied at feature time so an illiquid name is excluded from the cross-section rather
    than ranked and then ignored -- the difference matters, because including it would
    distort every other name's percentile.
    """
    ok_price = close >= spec.min_price
    ok_liquidity = dollar_vol >= spec.min_dollar_volume_60d
    return ok_price & ok_liquidity & close.notna()


def coverage_report(
    available: pd.DataFrame,
    spec: UniverseSpec,
    sample_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Per-date accounting of universe completeness.

    Columns:
        n_configured  names in the config that existed on that date (incl. delisted)
        n_available   names the provider actually served
        n_missing_delisted  known delisted names the provider cannot serve
        coverage      n_available / n_configured

    A coverage of 0.7 in 2012 means roughly a third of that date's real cross-section is
    absent -- and absent non-randomly.
    """
    dates = sample_dates if sample_dates is not None else available.index

    # A name that had not yet IPO'd is genuinely absent, not a data gap. Counting it in
    # the denominator would conflate IPO timing with survivorship bias -- and survivorship
    # is the only one of the two that biases results. Infer each name's listing start from
    # its first observation; this is sound for the IPO side and, crucially, is not
    # circular for the delisted side, which is what we are actually measuring.
    first_seen: dict[str, pd.Timestamp] = {}
    for col in available.columns:
        s = available[col].dropna()
        if not s.empty:
            first_seen[col] = s.index[0]

    rows = []
    for dt in dates:
        d = dt.date() if hasattr(dt, "date") else dt

        listed_yet = [t for t in spec.symbols
                      if t in first_seen and first_seen[t] <= dt]
        # Delisted names have no data at all, so they never appear in first_seen -- and
        # they are deliberately absent from the YAML ticker list, which holds only names
        # a provider can serve. They are nonetheless part of the true point-in-time
        # universe, so they are counted from the registry directly. Filtering these by
        # spec.tickers would silently zero out the entire survivorship disclosure.
        gone_but_alive = [x.ticker for x in delisted_as_of(d)]

        expected = len(listed_yet) + len(gone_but_alive)
        served = int(available.loc[dt].notna().sum()) if dt in available.index else 0

        rows.append({
            "date": dt,
            "n_configured": len(spec.symbols),
            "n_expected": expected,          # listed on this date, incl. since-delisted
            "n_available": served,
            "n_missing_delisted": len(gone_but_alive),
            # Coverage now isolates the survivorship gap: of the names that genuinely
            # traded on this date, what share can the provider serve?
            "coverage": served / max(expected, 1),
        })
    return pd.DataFrame(rows).set_index("date")


def summarize_coverage(cov: pd.DataFrame) -> dict:
    """Condense coverage into the few numbers a report should lead with."""
    by_year = cov.groupby(cov.index.year)["coverage"].mean()
    return {
        "mean_coverage": float(cov["coverage"].mean()),
        "min_coverage": float(cov["coverage"].min()),
        "worst_year": int(by_year.idxmin()) if len(by_year) else None,
        "worst_year_coverage": float(by_year.min()) if len(by_year) else float("nan"),
        "coverage_by_year": {int(y): float(v) for y, v in by_year.items()},
    }
