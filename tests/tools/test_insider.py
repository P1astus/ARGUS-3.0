"""Insider-activity (Form 3/4/5) parsing. No real network calls -- `EdgarClient` is faked,
the same pattern `tests/retrieval/test_live_edgar.py` uses. XML fixtures below are
minimised, byte-real reproductions of two live filings pulled and inspected before writing
the parser: NVIDIA's filing agent encodes booleans as "1"/"0", Apple's as "true"/"false" --
both must parse to the same result.
"""

from __future__ import annotations

from datetime import date

import pytest

from argus.tools.insider import build_insider_activity, render_insider_activity

# Reproduces the real NVDA filing measured live: a routine RSU award to a director,
# boolean flags encoded as "1"/"0".
FORM4_NUMERIC_BOOLEANS = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer><issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerName>JANE DIRECTOR</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-08-10</value></transactionDate>
            <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1262</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>1262</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

# Reproduces the real AAPL filing measured live: an open-market sale by an officer,
# boolean flags encoded as "true"/"false", officerTitle as direct text (no <value> wrapper).
FORM4_STRING_BOOLEANS = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer><issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
    <reportingOwner>
        <reportingOwnerId><rptOwnerName>JOHN OFFICER</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship>
            <isOfficer>true</isOfficer>
            <officerTitle>SVP, GC and Secretary</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-08-11</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1439</value></transactionShares>
                <transactionPricePerShare><value>307.75</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>40107</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

CIK = "1234567"


class FakeClient:
    def __init__(self, submissions: dict, docs: dict[str, bytes]) -> None:
        self._submissions = submissions
        self._docs = docs
        self.get_calls: list[str] = []

    def submissions(self, cik: str) -> dict:
        return self._submissions

    def _get(self, url: str) -> bytes:
        self.get_calls.append(url)
        return self._docs[url]


def _submissions(forms, dates, docs, accns=None):
    accns = accns or [f"0000000000-{i:02d}-000001" for i in range(len(forms))]
    return {"filings": {"recent": {
        "form": forms, "filingDate": dates, "primaryDocument": docs,
        "accessionNumber": accns,
    }}}


def _url(i: int, doc: str) -> str:
    acc = f"0000000000-{i:02d}-000001".replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc}/{doc}"


@pytest.fixture
def resolved(monkeypatch):
    monkeypatch.setattr("argus.tools.insider.resolve_ciks",
                        lambda tickers, ua: {"ACME": CIK})


class TestBooleanEncodingVariants:
    def test_numeric_1_0_encoding_parses_correctly(self, resolved):
        url = _url(0, "form4.xml")
        client = FakeClient(_submissions(["4"], ["2026-08-10"], ["form4.xml"]),
                            {url: FORM4_NUMERIC_BOOLEANS.encode()})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        t = a.transactions[0]
        assert t.is_director is True and t.is_officer is False
        assert t.owner_name == "JANE DIRECTOR"

    def test_string_true_false_encoding_parses_correctly(self, resolved):
        url = _url(0, "form4.xml")
        client = FakeClient(_submissions(["4"], ["2026-08-11"], ["form4.xml"]),
                            {url: FORM4_STRING_BOOLEANS.encode()})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        t = a.transactions[0]
        assert t.is_officer is True
        assert t.officer_title == "SVP, GC and Secretary"


class TestOpenMarketVsRoutine:
    def test_award_code_a_is_not_open_market(self, resolved):
        url = _url(0, "form4.xml")
        client = FakeClient(_submissions(["4"], ["2026-08-10"], ["form4.xml"]),
                            {url: FORM4_NUMERIC_BOOLEANS.encode()})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        assert a.transactions[0].is_open_market is False
        assert a.open_market_buys == 0 and a.open_market_sells == 0

    def test_sale_code_s_is_open_market_and_disposed(self, resolved):
        url = _url(0, "form4.xml")
        client = FakeClient(_submissions(["4"], ["2026-08-11"], ["form4.xml"]),
                            {url: FORM4_STRING_BOOLEANS.encode()})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        t = a.transactions[0]
        assert t.is_open_market is True
        assert t.acquired is False
        assert a.open_market_sells == 1


class TestAsOfAndLookback:
    def test_filings_after_as_of_are_excluded(self, resolved):
        client = FakeClient(_submissions(["4"], ["2026-09-01"], ["form4.xml"]), {})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        assert a.transactions == []
        assert client.get_calls == []   # never even fetched -- excluded before the request

    def test_filings_before_the_lookback_window_are_excluded(self, resolved):
        client = FakeClient(_submissions(["4"], ["2025-01-01"], ["form4.xml"]), {})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client,
                                   lookback_days=30)
        assert a.transactions == []

    def test_non_ownership_forms_are_ignored(self, resolved):
        client = FakeClient(_submissions(["10-K"], ["2026-08-10"], ["f.htm"]), {})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        assert a.transactions == []


class TestUnresolvedTickerAndFailures:
    def test_no_cik_returns_none(self, monkeypatch):
        monkeypatch.setattr("argus.tools.insider.resolve_ciks", lambda tickers, ua: {})
        assert build_insider_activity("NOPE", date(2026, 8, 15)) is None

    def test_one_bad_filing_does_not_abort_the_rest(self, resolved):
        good_url, bad_url = _url(0, "good.xml"), _url(1, "bad.xml")
        client = FakeClient(
            _submissions(["4", "4"], ["2026-08-10", "2026-08-11"], ["good.xml", "bad.xml"]),
            {good_url: FORM4_NUMERIC_BOOLEANS.encode()})   # bad_url deliberately missing
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        assert len(a.transactions) == 1

    def test_submissions_failure_returns_none(self, resolved):
        class BrokenClient:
            def submissions(self, cik):
                raise RuntimeError("network down")
        assert build_insider_activity("ACME", date(2026, 8, 15),
                                      client=BrokenClient()) is None


class TestRender:
    def test_no_transactions_says_so(self, resolved):
        client = FakeClient(_submissions([], [], []), {})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        assert "No Form 3/4/5" in render_insider_activity(a)

    def test_render_tags_open_market_vs_routine(self, resolved):
        url = _url(0, "form4.xml")
        client = FakeClient(_submissions(["4"], ["2026-08-11"], ["form4.xml"]),
                            {url: FORM4_STRING_BOOLEANS.encode()})
        a = build_insider_activity("ACME", date(2026, 8, 15), client=client)
        text = render_insider_activity(a)
        assert "OPEN MARKET" in text
        assert "JOHN OFFICER" in text
