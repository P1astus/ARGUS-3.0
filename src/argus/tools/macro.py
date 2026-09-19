"""Macro context for a briefing: a small set of FRED series, no API key required.

WHY THE fredgraph.csv ENDPOINT, NOT THE REST API
FRED's REST API needs a free-but-registered API key -- another account this project's own
"ask before spending money / signing up for things" discipline would need to clear with the
user first, the same reasoning `live_edgar.py` gives for not using a paid news API. The
`fredgraph.csv` export endpoint is public, unauthenticated, and returns the same published
series as plain CSV. Verified live for FEDFUNDS, CPIAUCSL, INDPRO, DGS10, UNRATE before
trusting it (an FRED series id that has been renamed/discontinued -- e.g. the old NAPM PMI
code -- returns an HTML error page instead of CSV, not an HTTP error, so the parser below
treats "first row isn't a date" as a signal to drop the series rather than crash on it).

REVISION LEAKAGE IS REAL AND NOT HIDDEN
This endpoint serves the CURRENT vintage of each series, not the point-in-time value that
was actually known on a historical `as_of` date. Policy-rate series (FEDFUNDS, DGS10) are
final at publication -- no leakage risk. CPI, industrial production and employment series
get seasonal and benchmark revisions after initial release, so a value pulled "as of"
2019-03-01 today may differ from what was published in March 2019. `MacroSeries.
revision_safe` carries this distinction into the rendered output rather than presenting
every series as equally point-in-time-clean, which is exactly the class of silent leakage
this project's own `as_of` discipline (search.py, live_edgar.py, the quant splitter) exists
to catch elsewhere.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from datetime import date

from argus.contracts.context import MacroContext, MacroSeries

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# A small, fixed set -- broad macro backdrop, not a sub-segment-specific factor model.
# (series_id, label, revision_safe)
DEFAULT_SERIES: tuple[tuple[str, str, bool], ...] = (
    ("FEDFUNDS", "Federal funds rate", True),
    ("DGS10", "10-year Treasury yield", True),
    ("CPIAUCSL", "CPI (all urban consumers)", False),
    ("INDPRO", "Industrial production index", False),
    ("UNRATE", "Unemployment rate", False),
)


def _fetch_series(series_id: str, as_of: date, user_agent: str) -> tuple[float, date] | None:
    """Most recent (value, value_date) published on or before `as_of`."""
    req = urllib.request.Request(FRED_CSV.format(series_id=series_id),
                                 headers={"User-Agent": user_agent})
    try:
        raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")
    except Exception as e:
        log.warning("%s: FRED fetch failed (%s)", series_id, type(e).__name__)
        return None

    lines = raw.strip().splitlines()
    if len(lines) < 2 or not lines[1][:4].isdigit():
        # A renamed/discontinued series id returns an HTML error page here, not an HTTP
        # error -- this is the "measured, not assumed" guard for that.
        log.warning("%s: not a recognised FRED series (renamed or discontinued?)", series_id)
        return None

    best: tuple[float, date] | None = None
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        d_raw, v_raw = parts
        try:
            d = date.fromisoformat(d_raw)
        except ValueError:
            continue
        if d > as_of:
            break   # CSV is chronological; nothing after this can qualify either
        try:
            v = float(v_raw)
        except ValueError:
            continue   # FRED writes "." for a missing observation
        best = (v, d)
    return best


def build_macro_context(as_of: date, series: tuple[tuple[str, str, bool], ...] = DEFAULT_SERIES,
                        user_agent: str = os.environ.get("ARGUS_SEC_UA", "ARGUS Research contact@example.com")
                        ) -> MacroContext:
    """Best-effort: a series that fails to fetch or parse is dropped, not fatal to the
    whole briefing -- macro context is backdrop, not evidence the extraction gate depends
    on."""
    points = []
    for series_id, label, revision_safe in series:
        result = _fetch_series(series_id, as_of, user_agent)
        if result is None:
            continue
        value, value_date = result
        points.append(MacroSeries(series_id=series_id, label=label, value=value,
                                  value_date=value_date, revision_safe=revision_safe))
    return MacroContext(as_of=as_of, series=points)


def render_macro_context(m: MacroContext) -> str:
    if not m.series:
        return "No macro series available."
    lines = []
    for s in m.series:
        flag = "" if s.revision_safe else "  [subject to later revision]"
        lines.append(f"- {s.label} ({s.series_id}): {s.value:g} as of {s.value_date}{flag}")
    return "\n".join(lines)
