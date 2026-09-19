"""LiveEdgarRetriever -- no real network calls. `EdgarClient` is replaced with a fake
that returns canned submissions/document bytes, so these test the retriever's own logic
(as_of filtering, form-type dispatch, section selection, graceful failure) rather than
SEC's API being reachable.
"""

from __future__ import annotations

from datetime import date

import pytest

from argus.tools.retrievers.live_edgar import LiveEdgarRetriever

# extract_narrative's real default is min_chars=500 (matching what a genuine filing
# section looks like) and live_edgar.py calls it with that default -- so these fixtures
# must clear 500 chars of actual section content, or every test document gets silently
# dropped as "too short" regardless of whether the retriever logic is correct.
TENK_HTML = """<html><body>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased 15% year over year, driven by strong demand in the automotive and
industrial end markets during the reported quarter, partially offset by softness in
consumer electronics. Gross margin expanded on favorable product mix and lower input
costs, while operating expenses grew modestly as the company continued to invest in
research and development for next-generation product families. Management expects
these trends to continue into the following quarter, subject to ongoing macroeconomic
uncertainty and customer inventory normalization across the broader distribution
channel during the remainder of the fiscal year.</p>
</body></html>"""

TWENTYF_HTML = """<html><body>
<p>ITEM 5. \n OPERATING AND FINANCIAL REVIEWS AND PROSPECTS \n 40</p>
<p>ITEM 5. OPERATING AND FINANCIAL REVIEWS AND PROSPECTS</p>
<p>Net revenue increased in the period, driven by strong demand for advanced process
nodes across our foundry customer base during the fiscal year under review. Capacity
utilization remained elevated throughout the period as customers continued to place
orders well ahead of anticipated demand, and capital expenditures were directed
primarily toward expanding capacity for our most advanced technology nodes. Gross
margin improved as a result of a more favorable product mix and continued cost
discipline across our manufacturing operations during the fiscal year.</p>
</body></html>"""


class FakeClient:
    """Satisfies the EdgarClient methods LiveEdgarRetriever actually calls."""

    def __init__(self, submissions: dict, docs: dict[str, bytes], filing_index=None):
        self._submissions = submissions
        self._docs = docs
        self._filing_index = filing_index or {}
        self.get_calls: list[str] = []
        self.filing_index_calls: list[tuple[str, str]] = []

    def submissions(self, cik: str) -> dict:
        return self._submissions

    def filing_index(self, cik: str, accession: str) -> dict:
        self.filing_index_calls.append((cik, accession))
        return self._filing_index

    def _get(self, url: str) -> bytes:
        self.get_calls.append(url)
        return self._docs[url]


CIK = "1234567"


def _accn(i: int) -> str:
    return f"0000000000-{i:02d}-000001"


def _url(i: int, doc: str) -> str:
    # Exact mirror of LiveEdgarRetriever.fetch()'s own URL construction, so a test
    # fixture and the code under test can never silently diverge on format.
    acc = _accn(i).replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc}/{doc}"


def _submissions(forms, dates, docs, accns=None):
    accns = accns or [_accn(i) for i in range(len(forms))]
    return {"filings": {"recent": {
        "form": forms, "filingDate": dates, "primaryDocument": docs,
        "accessionNumber": accns,
    }}}


@pytest.fixture
def retriever(monkeypatch):
    r = LiveEdgarRetriever()
    monkeypatch.setattr("argus.tools.retrievers.live_edgar.resolve_ciks",
                        lambda tickers, ua: {"ACME": "1234567"})
    return r


class TestLicenseAndSourceType:
    def test_edgar_is_public_domain(self, retriever):
        assert retriever.license_flags() == (True, True)

    def test_source_type_is_sec_filing(self, retriever):
        from argus.contracts.briefing import SourceType
        assert retriever.source_type() == SourceType.SEC_FILING


class TestUnresolvedTicker:
    def test_no_cik_returns_empty_list_not_an_exception(self, monkeypatch):
        r = LiveEdgarRetriever()
        monkeypatch.setattr("argus.tools.retrievers.live_edgar.resolve_ciks",
                            lambda tickers, ua: {})
        assert r.fetch("NOPE", date(2024, 1, 1), 5) == []


class TestFetch:
    def test_returns_extracted_section_for_a_recent_10k(self, retriever):
        url = _url(0, "f.htm")
        retriever.client = FakeClient(
            _submissions(["10-K"], ["2024-01-15"], ["f.htm"]), {url: TENK_HTML.encode()})
        out = retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert len(out) == 1
        text, meta = out[0]
        assert "Revenue increased 15%" in text
        assert meta["published"] == date(2024, 1, 15)
        assert "10-K" in meta["title"]

    def test_respects_as_of_leakage_boundary(self, retriever):
        url_old, url_new = _url(0, "f.htm"), _url(1, "g.htm")
        retriever.client = FakeClient(
            _submissions(["10-K", "10-K"], ["2023-01-15", "2025-01-15"], ["f.htm", "g.htm"]),
            {url_old: TENK_HTML.encode(), url_new: TENK_HTML.encode()})
        out = retriever.fetch("ACME", date(2024, 1, 1), limit=5)
        # Only the 2023 filing is on or before as_of; the 2025 one must never be fetched.
        assert url_new not in retriever.client.get_calls
        assert len(out) == 1

    def test_unknown_form_type_is_skipped(self, retriever):
        retriever.client = FakeClient(
            _submissions(["SC 13G"], ["2024-01-15"], ["x.htm"]), {})
        out = retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert out == []

    def test_20f_form_uses_20f_section_patterns(self, retriever):
        url = _url(0, "f.htm")
        retriever.client = FakeClient(
            _submissions(["20-F"], ["2024-01-15"], ["f.htm"]), {url: TWENTYF_HTML.encode()})
        out = retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert len(out) == 1
        assert "Net revenue increased" in out[0][0]

    def test_limit_caps_the_number_of_filings_fetched(self, retriever):
        forms = ["10-K"] * 5
        dates = [f"202{i}-01-15" for i in range(5)]
        docs = [f"f{i}.htm" for i in range(5)]
        urls = [_url(i, f"f{i}.htm") for i in range(5)]
        retriever.client = FakeClient(_submissions(forms, dates, docs),
                                      {u: TENK_HTML.encode() for u in urls})
        out = retriever.fetch("ACME", date(2029, 1, 1), limit=2)
        assert len(out) == 2

    def test_fetch_failure_for_one_filing_does_not_abort_the_rest(self, retriever):
        good_url, bad_url = _url(0, "f.htm"), _url(1, "g.htm")
        client = FakeClient(
            _submissions(["10-K", "10-K"], ["2024-01-15", "2024-02-15"], ["f.htm", "g.htm"]),
            {good_url: TENK_HTML.encode()})   # bad_url deliberately missing
        retriever.client = client
        out = retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert len(out) == 1

    def test_submissions_lookup_failure_returns_empty_not_raises(self, retriever):
        class BrokenClient:
            def submissions(self, cik):
                raise RuntimeError("network down")
        retriever.client = BrokenClient()
        assert retriever.fetch("ACME", date(2024, 1, 1), 5) == []

    def test_8k_fetches_exhibits_in_addition_to_the_cover_page(self, retriever):
        # Handoff §6: an 8-K's primary document is a cover page; the substance is in
        # EX-99 exhibits, reachable only via the filing's own index.json.
        cover_url = _url(0, "cover.htm")
        exhibit_url = _url(0, "ex99-1pressrelease.htm")
        retriever.client = FakeClient(
            _submissions(["8-K"], ["2024-01-15"], ["cover.htm"]),
            {cover_url: TENK_HTML.encode(), exhibit_url: TENK_HTML.encode()},
            filing_index={"directory": {"item": [
                {"name": "cover.htm"},
                {"name": "ex99-1pressrelease.htm"},
                {"name": "R1.htm"},          # XBRL viewer noise -- must be skipped
            ]}})
        out = retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert exhibit_url in retriever.client.get_calls
        titles = [meta["title"] for _, meta in out]
        assert any("8-K/EX" in t for t in titles)
        assert any("8-K " in t and "8-K/EX" not in t for t in titles)

    def test_non_8k_filings_do_not_trigger_exhibit_lookup(self, retriever):
        url = _url(0, "f.htm")
        client = FakeClient(_submissions(["10-K"], ["2024-01-15"], ["f.htm"]),
                            {url: TENK_HTML.encode()})
        retriever.client = client
        retriever.fetch("ACME", date(2024, 6, 1), limit=5)
        assert client.filing_index_calls == []   # never even asked for

    def test_cik_resolution_is_cached_across_calls(self, monkeypatch):
        calls = []
        def fake_resolve(tickers, ua):
            calls.append(tickers)
            return {"ACME": "1234567"}
        monkeypatch.setattr("argus.tools.retrievers.live_edgar.resolve_ciks", fake_resolve)
        r = LiveEdgarRetriever()
        r.client = FakeClient(_submissions([], [], []), {})
        r.fetch("ACME", date(2024, 1, 1), 5)
        r.fetch("ACME", date(2024, 2, 1), 5)
        assert len(calls) == 1, "second fetch must reuse the cached CIK"
