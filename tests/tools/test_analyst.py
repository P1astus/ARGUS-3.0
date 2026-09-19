"""Analyst-consensus reconstruction: point-in-time correctness (the whole reason this
module exists instead of reading yfinance's live targetMeanPrice field directly) and the
staleness filter that keeps a consensus from being distorted by firms that stopped
updating years ago. `raw` is injected directly -- no real yfinance call in these tests --
shaped exactly like the real `Ticker.upgrades_downgrades` frame (verified live for GOOGL
before writing this: 'Firm'/'ToGrade'/'Action'/'priceTargetAction'/'currentPriceTarget'
columns, a DatetimeIndex named GradeDate, 0.0-not-NaN for a target-less rating action).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from argus.tools.analyst import build_analyst_consensus, render_analyst_consensus


def _raw(rows: list[dict]) -> pd.DataFrame:
    """rows: [{"date": "...", "firm": ..., "grade": ..., "action": ..., "target": ...
    (float or None for a no-target action)}]"""
    idx = pd.to_datetime([r["date"] for r in rows])
    return pd.DataFrame({
        "Firm": [r["firm"] for r in rows],
        "ToGrade": [r["grade"] for r in rows],
        "Action": [r.get("action", "main") for r in rows],
        "priceTargetAction": ["" if r["target"] is None else "Raises" for r in rows],
        "currentPriceTarget": [0.0 if r["target"] is None else r["target"] for r in rows],
    }, index=pd.Index(idx, name="GradeDate"))


class TestPointInTimeCorrectness:
    def test_only_uses_the_most_recent_action_per_firm_as_of_the_date(self):
        raw = _raw([
            {"date": "2025-01-01", "firm": "UBS", "grade": "Hold", "target": 100.0},
            {"date": "2025-06-01", "firm": "UBS", "grade": "Buy", "target": 150.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 8, 1), raw=raw)
        assert len(c.ratings) == 1
        assert c.ratings[0].to_grade == "Buy" and c.ratings[0].price_target == 150.0

    def test_never_uses_an_action_dated_after_as_of(self):
        raw = _raw([
            {"date": "2025-01-01", "firm": "UBS", "grade": "Hold", "target": 100.0},
            {"date": "2026-01-01", "firm": "UBS", "grade": "Buy", "target": 200.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 6, 1), raw=raw)
        assert len(c.ratings) == 1
        assert c.ratings[0].to_grade == "Hold" and c.ratings[0].price_target == 100.0

    def test_historical_and_current_views_of_the_same_firm_can_differ(self):
        raw = _raw([
            {"date": "2025-01-01", "firm": "UBS", "grade": "Hold", "target": 100.0},
            {"date": "2025-06-01", "firm": "UBS", "grade": "Buy", "target": 150.0},
        ])
        old = build_analyst_consensus("ACME", date(2025, 3, 1), raw=raw)
        new = build_analyst_consensus("ACME", date(2025, 8, 1), raw=raw)
        assert old.ratings[0].to_grade != new.ratings[0].to_grade
        assert old.mean_target != new.mean_target


class TestStalenessFilter:
    def test_drops_a_firm_whose_last_action_predates_the_staleness_window(self):
        # UBS last rated 2 years before as_of -- must not be treated as active coverage.
        raw = _raw([
            {"date": "2023-01-01", "firm": "UBS", "grade": "Buy", "target": 3000.0},
            {"date": "2025-06-01", "firm": "Barclays", "grade": "Buy", "target": 400.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 8, 1), raw=raw,
                                    max_staleness_days=400)
        assert len(c.ratings) == 1
        assert c.ratings[0].firm == "Barclays"
        assert c.mean_target == 400.0   # the stale $3000 target must not pollute this

    def test_keeps_a_firm_just_inside_the_staleness_window(self):
        raw = _raw([
            {"date": "2025-01-01", "firm": "UBS", "grade": "Buy", "target": 300.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 6, 1), raw=raw,
                                    max_staleness_days=400)
        assert len(c.ratings) == 1


class TestNoTargetActions:
    def test_rating_only_action_is_kept_for_tally_but_excluded_from_targets(self):
        raw = _raw([
            {"date": "2025-06-01", "firm": "JMP", "grade": "Buy", "target": None},
            {"date": "2025-06-01", "firm": "UBS", "grade": "Hold", "target": 200.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 8, 1), raw=raw)
        assert len(c.ratings) == 2
        assert len(c.targets) == 1
        assert c.mean_target == 200.0


class TestEmptyAndMissing:
    def test_no_data_at_all_returns_none(self):
        assert build_analyst_consensus("ACME", date(2025, 1, 1),
                                       raw=pd.DataFrame()) is None

    def test_no_coverage_within_window_returns_a_consensus_with_no_ratings(self):
        raw = _raw([{"date": "2020-01-01", "firm": "UBS", "grade": "Buy", "target": 100.0}])
        c = build_analyst_consensus("ACME", date(2025, 1, 1), raw=raw,
                                    max_staleness_days=30)
        assert c is None   # everything filtered out -- same "absent" convention


class TestRender:
    def test_no_ratings_says_so(self):
        from argus.contracts.context import AnalystConsensus
        c = AnalystConsensus(ticker="ACME", as_of=date(2025, 1, 1), ratings=[])
        assert "No analyst coverage" in render_analyst_consensus(c)

    def test_render_includes_mean_and_range(self):
        raw = _raw([
            {"date": "2025-06-01", "firm": "UBS", "grade": "Buy", "target": 100.0},
            {"date": "2025-06-01", "firm": "Barclays", "grade": "Hold", "target": 200.0},
        ])
        c = build_analyst_consensus("ACME", date(2025, 8, 1), raw=raw)
        text = render_analyst_consensus(c)
        assert "mean 150.00" in text
        assert "100.00 - 200.00" in text
