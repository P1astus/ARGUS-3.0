"""Price context for a briefing: recent return/volume snapshot as of a decision date.

Reuses `PriceStore`/`YFinanceProvider` as-is -- no new fetching code, just a read over an
existing cache and a leakage-safe window computation. `PriceStore.get` already caches to
parquet, so this costs one cheap read after the first call for a given ticker.
"""

from __future__ import annotations

from datetime import date, timedelta

from argus.contracts.context import PriceSnapshot
from argus.data.prices import PriceStore


def build_price_snapshot(store: PriceStore, ticker: str, as_of: date,
                         lookback_days: int = 400) -> PriceSnapshot | None:
    """Trailing price/volume context, using only bars on or before `as_of`.

    `lookback_days=400` gives headroom past the 252-trading-day window this pulls
    returns over (calendar days, not trading days, so ~365 would be cutting it close on a
    year with more holidays than usual). Returns None rather than a partially-populated
    snapshot when there is no data at all -- a name with zero bars (delisted, pre-IPO,
    bad ticker) should be absent from the briefing, not shown as a snapshot of nothing.
    """
    df = store.get(ticker, as_of - timedelta(days=lookback_days), as_of)
    df = df.loc[:str(as_of)]   # belt-and-suspenders: PriceStore's cache slice can return
                               # a few trailing days past `end` (see its own docstring)
    if df.empty:
        return None

    close = df["adj_close"].dropna()
    volume = df["volume"].dropna()
    if close.empty:
        return None

    def pct_return(window: int) -> float | None:
        if len(close) <= window:
            return None
        past = close.iloc[-(window + 1)]
        if past == 0:
            return None
        return float(close.iloc[-1] / past - 1.0)

    return PriceSnapshot(
        ticker=ticker, as_of=as_of,
        close=float(close.iloc[-1]),
        volume=float(volume.iloc[-1]) if not volume.empty else 0.0,
        return_20d=pct_return(20), return_90d=pct_return(90), return_252d=pct_return(252),
        avg_volume_20d=float(volume.tail(20).mean()) if len(volume) >= 1 else None,
        high_252d=float(close.tail(252).max()),
        low_252d=float(close.tail(252).min()),
        bars_used=len(close),
    )


def render_price_snapshot(p: PriceSnapshot) -> str:
    def fmt_pct(x: float | None) -> str:
        return f"{x:+.1%}" if x is not None else "n/a (insufficient history)"

    close_line = f"Close {p.close:.2f}  Volume {p.volume:,.0f}"
    if p.avg_volume_20d:
        close_line += f"  (20d avg {p.avg_volume_20d:,.0f})"

    lines = [
        close_line,
        f"Return: 20d {fmt_pct(p.return_20d)} · 90d {fmt_pct(p.return_90d)} · "
        f"252d {fmt_pct(p.return_252d)}",
    ]
    if p.high_252d is not None and p.low_252d is not None:
        lines.append(f"252d range: {p.low_252d:.2f} - {p.high_252d:.2f}")
    if p.bars_used < 60:
        lines.append(f"(only {p.bars_used} trading days of history available -- "
                     f"longer-window returns above are unreliable or absent)")
    return "\n".join(lines)
