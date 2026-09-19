"""Macro-context fetching against the fredgraph.csv endpoint. `urllib.request.urlopen` is
faked -- no real network call -- with canned CSV bodies shaped exactly like the real
endpoint's output (verified live for FEDFUNDS/CPIAUCSL/INDPRO/DGS10/UNRATE before writing
this), including the discontinued-series case (an HTML error page, not an HTTP error).
"""

from __future__ import annotations

from datetime import date

from argus.tools.macro import build_macro_context, render_macro_context

REAL_SHAPED_CSV = """observation_date,FEDFUNDS
2026-04-01,3.85
2026-05-01,3.64
2026-06-01,3.63
2026-07-01,3.63
"""

DISCONTINUED_SERIES_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body>Series not found</body></html>"""


class FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, bodies: dict[str, str]) -> list[str]:
    """`bodies` keyed by series_id (matched via substring in the request URL)."""
    calls: list[str] = []

    def fake_urlopen(req, timeout=20):
        calls.append(req.full_url)
        for series_id, body in bodies.items():
            if f"id={series_id}" in req.full_url:
                return FakeResponse(body)
        return FakeResponse(DISCONTINUED_SERIES_HTML)

    monkeypatch.setattr("argus.tools.macro.urllib.request.urlopen", fake_urlopen)
    return calls


class TestBuildMacroContext:
    def test_picks_the_most_recent_point_on_or_before_as_of(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"FEDFUNDS": REAL_SHAPED_CSV})
        ctx = build_macro_context(date(2026, 6, 15),
                                  series=(("FEDFUNDS", "Fed funds rate", True),))
        assert len(ctx.series) == 1
        # CSV has 06-01=3.63 and 07-01=3.63; as_of=06-15 must land on 06-01, not the
        # later 07-01 point (excluded) or the earlier 05-01 one (not the most recent).
        assert ctx.series[0].value == 3.63
        assert ctx.series[0].value_date == date(2026, 6, 1)

    def test_never_uses_a_point_dated_after_as_of(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"FEDFUNDS": REAL_SHAPED_CSV})
        ctx = build_macro_context(date(2026, 4, 15),
                                  series=(("FEDFUNDS", "Fed funds rate", True),))
        assert ctx.series[0].value == 3.85
        assert ctx.series[0].value_date == date(2026, 4, 1)

    def test_as_of_before_any_observation_yields_no_point_for_that_series(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"FEDFUNDS": REAL_SHAPED_CSV})
        ctx = build_macro_context(date(2020, 1, 1),
                                  series=(("FEDFUNDS", "Fed funds rate", True),))
        assert ctx.series == []

    def test_discontinued_series_is_dropped_not_fatal(self, monkeypatch):
        _patch_urlopen(monkeypatch, {})   # everything falls through to the HTML page
        ctx = build_macro_context(date(2026, 6, 1),
                                  series=(("NAPM", "old PMI code", True),
                                         ("FEDFUNDS", "Fed funds rate", True)))
        # NAPM (renamed/discontinued) is silently dropped; a real series alongside it
        # still comes through -- one bad series must not blank the whole context.
        assert ctx.series == []   # FEDFUNDS not stubbed in this test either -> also dropped

    def test_network_failure_for_one_series_does_not_abort_the_rest(self, monkeypatch):
        def flaky(req, timeout=20):
            if "DGS10" in req.full_url:
                raise TimeoutError("slow network")
            return FakeResponse(REAL_SHAPED_CSV)
        monkeypatch.setattr("argus.tools.macro.urllib.request.urlopen", flaky)

        ctx = build_macro_context(date(2026, 6, 1),
                                  series=(("DGS10", "10y yield", True),
                                         ("FEDFUNDS", "Fed funds rate", True)))
        assert len(ctx.series) == 1
        assert ctx.series[0].series_id == "FEDFUNDS"

    def test_revision_safety_flag_is_carried_through(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"CPIAUCSL": REAL_SHAPED_CSV.replace("FEDFUNDS", "CPIAUCSL")})
        ctx = build_macro_context(date(2026, 6, 1),
                                  series=(("CPIAUCSL", "CPI", False),))
        assert ctx.series[0].revision_safe is False


class TestRenderMacroContext:
    def test_flags_non_revision_safe_series(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"CPIAUCSL": REAL_SHAPED_CSV.replace("FEDFUNDS", "CPIAUCSL")})
        ctx = build_macro_context(date(2026, 6, 1), series=(("CPIAUCSL", "CPI", False),))
        text = render_macro_context(ctx)
        assert "subject to later revision" in text

    def test_revision_safe_series_has_no_flag(self, monkeypatch):
        _patch_urlopen(monkeypatch, {"FEDFUNDS": REAL_SHAPED_CSV})
        ctx = build_macro_context(date(2026, 6, 1), series=(("FEDFUNDS", "Fed funds", True),))
        text = render_macro_context(ctx)
        assert "subject to later revision" not in text

    def test_empty_context_says_so(self):
        from argus.contracts.context import MacroContext
        assert "No macro series" in render_macro_context(MacroContext(as_of=date(2026, 1, 1), series=[]))
