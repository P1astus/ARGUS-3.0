"""Analyst price-target consensus, reconstructed point-in-time rather than read live.

WHY NOT Ticker.info['targetMeanPrice']
That field is real and free (yfinance, no signup), but it is a LIVE snapshot with no
date attached -- there is no way to tell whether it reflects today's consensus or six
months ago's. Attaching it to a historical `as_of` briefing would leak today's numbers
into a past decision date, the same class of bug this project's `as_of` discipline exists
everywhere else (search.py, live_edgar.py, the quant splitter, macro.py's revision_safe
flag) to prevent -- and a worse one, since it is not a minor revision, it is the whole
future.

WHAT THIS USES INSTEAD
yfinance also exposes `Ticker.upgrades_downgrades`: a dated history of every rating/target
action per firm, back to 2012 for names with long coverage (verified live for GOOGL: 992
rows, 2012-03-14 to present). Reconstructing "consensus as of `as_of`" from this -- each
firm's most recent action on or before that date -- is genuinely point-in-time-correct,
not a snapshot pretending to be one.

WHAT THIS IS NOT
Best-effort, not authoritative: this is whatever Yahoo Finance's own aggregator happened
to capture, with the same silent-gap caveat `yfinance_provider.py` already documents for
price data. Some actions carry no price target at all (a pure rating change) -- these are
kept in `ratings` for the rating tally but excluded from `mean_target`/`high_target`/
`low_target`, which only average actions that actually gave a number.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from argus.contracts.context import AnalystConsensus, AnalystRating

log = logging.getLogger(__name__)


def _fetch_raw(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).upgrades_downgrades
    except Exception as e:
        log.warning("%s: yfinance upgrades_downgrades fetch failed (%s)",
                   ticker, type(e).__name__)
        return None
    return df


def build_analyst_consensus(ticker: str, as_of: date, raw: pd.DataFrame | None = None,
                            max_staleness_days: int = 400) -> AnalystConsensus | None:
    """Most recent rating/target per firm, on or before `as_of`.

    `raw` is injectable for tests; production callers omit it and this fetches live.
    Returns None when there is no data at all (unknown ticker, no analyst coverage, or
    the fetch failed) -- same "absent, not hollow" convention as price_context.py.

    MEASURED, NOT ASSUMED: an early version of this used every firm's most recent action
    with no staleness check, and for GOOGL that pulled in pre-split (2022) targets of
    $1700-$3000 alongside post-split targets of $100-$500 -- a firm that has not updated
    in years is not "current coverage", it is a stale number that happens to still be the
    most recent row for that firm. `max_staleness_days=400` (same lookback convention
    `ExtractConfig`/`price_context.py` already use) drops firms whose last action is
    older than that relative to `as_of`, rather than reporting a consensus quietly
    distorted by names nobody has re-rated in years.
    """
    df = raw if raw is not None else _fetch_raw(ticker)
    if df is None or df.empty:
        return None

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)   # GradeDate carries a time; a
    # same-day action is "on as_of" even if timestamped later in that day than midnight.
    df = df[df.index < cutoff]
    stale_before = cutoff - pd.Timedelta(days=max_staleness_days)
    df = df[df.index >= stale_before]
    if df.empty:
        return None

    # Most recent action per firm as of the cutoff -- a firm's older actions are
    # superseded, not additional evidence.
    df = df.sort_index()
    latest_per_firm = df.groupby("Firm").tail(1)

    ratings = []
    for grade_date, row in latest_per_firm.iterrows():
        target = row.get("currentPriceTarget")
        # yfinance writes 0.0 (not NaN) for a pure rating action with no price target --
        # priceTargetAction is empty in exactly that case, the reliable signal to use.
        has_target = bool(row.get("priceTargetAction")) and target not in (None, 0, 0.0)
        ratings.append(AnalystRating(
            firm=str(row["Firm"]), grade_date=grade_date.date(),
            to_grade=str(row.get("ToGrade") or ""), action=str(row.get("Action") or ""),
            price_target=float(target) if has_target else None))

    return AnalystConsensus(ticker=ticker, as_of=as_of, ratings=ratings)


def render_analyst_consensus(a: AnalystConsensus) -> str:
    if not a.ratings:
        return "No analyst coverage found."

    lines = []
    if a.targets:
        lines.append(f"Price target: mean {a.mean_target:.2f}  "
                     f"(range {a.low_target:.2f} - {a.high_target:.2f}, "
                     f"{len(a.targets)} of {len(a.ratings)} firms gave a target)")
    else:
        lines.append(f"{len(a.ratings)} firm(s) covering, none with a numeric target "
                     f"as of this date.")

    grades: dict[str, int] = {}
    for r in a.ratings:
        grades[r.to_grade] = grades.get(r.to_grade, 0) + 1
    grade_summary = ", ".join(f"{n} {g}" for g, n in sorted(grades.items(),
                                                            key=lambda x: -x[1]))
    lines.append(f"Ratings: {grade_summary}")

    most_recent = max(a.ratings, key=lambda r: r.grade_date)
    lines.append(f"Most recent action: {most_recent.grade_date} {most_recent.firm} "
                 f"→ {most_recent.to_grade}"
                 + (f" (target {most_recent.price_target:.2f})"
                    if most_recent.price_target else ""))
    return "\n".join(lines)
