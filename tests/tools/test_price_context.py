"""Price-snapshot construction: leakage boundary, return math, thin-history handling.
No real PriceStore/yfinance involved -- `build_price_snapshot` only calls `.get(ticker,
start, end)`, so a minimal fake satisfying that is enough to test the logic on its own.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from argus.tools.price_context import build_price_snapshot, render_price_snapshot


class FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def get(self, ticker, start, end, refresh=False):
        return self.df.loc[:pd.Timestamp(end)]


def _daily_frame(n: int, start="2024-01-01", base=100.0, drift=0.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = base + np.arange(n) * drift
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "adj_close": close, "volume": np.full(n, 1_000_000.0),
    }, index=idx)


class TestBuildPriceSnapshot:
    def test_none_when_no_data_at_all(self):
        store = FakeStore(pd.DataFrame(columns=["adj_close", "volume"]))
        assert build_price_snapshot(store, "NOPE", date(2024, 6, 1)) is None

    def test_basic_fields_populate_from_the_last_available_bar(self):
        df = _daily_frame(30, drift=1.0)
        store = FakeStore(df)
        snap = build_price_snapshot(store, "ACME", df.index[-1].date())
        assert snap.ticker == "ACME"
        assert snap.close == df["adj_close"].iloc[-1]
        assert snap.bars_used == 30

    def test_never_uses_a_bar_dated_after_as_of(self):
        # 60 trading days; as_of lands mid-series -- the snapshot's close must be the
        # as_of-day bar, not a later one the fake store's raw frame also contains.
        df = _daily_frame(60, drift=1.0)
        store = FakeStore(df)
        as_of = df.index[29].date()
        snap = build_price_snapshot(store, "ACME", as_of)
        assert snap.close == df["adj_close"].iloc[29]
        assert snap.bars_used == 30   # only bars up to and including as_of

    def test_returns_none_for_windows_longer_than_available_history(self):
        df = _daily_frame(10, drift=1.0)   # far short of the 20d window
        snap = build_price_snapshot(FakeStore(df), "ACME", date(2024, 1, 15))
        assert snap.return_20d is None
        assert snap.return_90d is None
        assert snap.return_252d is None

    def test_return_math_is_correct(self):
        df = _daily_frame(25, base=100.0, drift=0.0)
        df.iloc[-1, df.columns.get_loc("adj_close")] = 110.0   # +10% vs 20 bars back
        snap = build_price_snapshot(FakeStore(df), "ACME", df.index[-1].date())
        assert snap.return_20d == pytest.approx(0.10, abs=1e-6)

    def test_252d_high_low_reflect_only_available_bars(self):
        df = _daily_frame(30, base=100.0, drift=1.0)
        snap = build_price_snapshot(FakeStore(df), "ACME", df.index[-1].date())
        assert snap.low_252d == pytest.approx(df["adj_close"].min())
        assert snap.high_252d == pytest.approx(df["adj_close"].max())


class TestRenderPriceSnapshot:
    def test_flags_thin_history(self):
        df = _daily_frame(10, drift=1.0)
        snap = build_price_snapshot(FakeStore(df), "ACME", date(2024, 1, 15))
        text = render_price_snapshot(snap)
        assert "only 10 trading days" in text
        assert "n/a (insufficient history)" in text

    def test_well_covered_snapshot_has_no_thin_history_warning(self):
        df = _daily_frame(300, drift=0.1)
        snap = build_price_snapshot(FakeStore(df), "ACME", date(2025, 1, 1))
        assert "insufficient history" not in render_price_snapshot(snap)
