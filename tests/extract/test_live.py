"""`run_live` -- the extraction/quote-verification pipeline sourced from a retriever
instead of a pre-built index. `LiveEdgarRetriever` itself already has its own tests
(tests/retrieval/test_live_edgar.py); this tests that `run_live` correctly threads its
output into the SAME verification path `ExtractionLoop.run()` uses, via a fake retriever
so no real fetch is needed here either.
"""

from __future__ import annotations

from datetime import date

from argus.contracts.quant import SubSegment
from argus.engine.extract.live import run_live
from argus.engine.inference.local import ScriptedGenerator
from argus.tools.provenance import ProvenanceStore, audit_briefing


class FakeRetriever:
    def __init__(self, passages: list[tuple[str, dict]]) -> None:
        self.passages = passages
        self.calls: list[tuple[str, date, int]] = []

    def fetch(self, ticker, as_of, limit):
        self.calls.append((ticker, as_of, limit))
        return self.passages[:limit]


PASSAGE_TEXT = ("Revenue grew 12% year over year, driven by strong demand in the data "
                "center segment during the quarter.")


def test_produces_a_verifiable_briefing_from_live_passages(tmp_path):
    retriever = FakeRetriever([(PASSAGE_TEXT, {
        "url": "https://example.com/f.htm", "title": "ACME 8-K/EX 2026-08-10 full",
        "published": date(2026, 8, 10)})])
    gen = ScriptedGenerator(outputs=[
        '{"kind":"sourced","source":"S1","quote":"Revenue grew 12% year over year",'
        '"text":"Revenue grew 12% YoY."}'])
    prov = ProvenanceStore(tmp_path / "prov")

    res = run_live("ACME", SubSegment.FABLESS, date(2026, 8, 15), retriever, gen, prov)

    assert res.briefing.claims
    audit = audit_briefing(res.briefing, prov)
    assert audit["passed"]
    assert audit["n_unverified_quotes"] == 0


def test_fabricated_quote_is_dropped_same_as_the_indexed_path(tmp_path):
    retriever = FakeRetriever([(PASSAGE_TEXT, {
        "url": None, "title": "ACME 10-K", "published": date(2026, 8, 10)})])
    gen = ScriptedGenerator(outputs=[
        '{"kind":"sourced","source":"S1","quote":"Entirely invented figures here",'
        '"text":"x"}'])
    prov = ProvenanceStore(tmp_path / "prov")

    res = run_live("ACME", SubSegment.FABLESS, date(2026, 8, 15), retriever, gen, prov)

    assert res.briefing.claims == []
    assert res.stats.quote_fabricated == 1


def test_limit_is_passed_through_to_the_retriever(tmp_path):
    retriever = FakeRetriever([(PASSAGE_TEXT, {"published": date(2026, 8, 10)})] * 5)
    gen = ScriptedGenerator(outputs=[""])
    prov = ProvenanceStore(tmp_path / "prov")

    run_live("ACME", SubSegment.FABLESS, date(2026, 8, 15), retriever, gen, prov, limit=3)

    assert retriever.calls == [("ACME", date(2026, 8, 15), 3)]


def test_no_passages_yields_an_empty_but_valid_briefing(tmp_path):
    retriever = FakeRetriever([])
    gen = ScriptedGenerator(outputs=[""])
    prov = ProvenanceStore(tmp_path / "prov")

    res = run_live("ACME", SubSegment.FABLESS, date(2026, 8, 15), retriever, gen, prov)

    assert res.briefing.claims == []
    assert res.briefing.sources == []
