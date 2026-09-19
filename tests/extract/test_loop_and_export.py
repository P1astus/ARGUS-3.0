"""End-to-end extraction on a synthetic index, with a scripted model.

Uses `ScriptedGenerator` so the fabrication path can be exercised deliberately -- a real
model cannot be asked to hallucinate on cue, which would leave the most important branch
in the system untested.
"""

from __future__ import annotations

from datetime import date

import pytest

from argus.contracts.briefing import ClaimKind
from argus.contracts.quant import SubSegment
from argus.engine.extract.export import filename, to_markdown
from argus.engine.extract.loop import ExtractConfig, ExtractionLoop, segment_queries
from argus.engine.inference.local import ScriptedGenerator
from argus.engine.retrieval.chunk import ChunkSpec, chunk_document
from argus.engine.retrieval.search import Searcher
from argus.engine.retrieval.store import ChunkStore
from argus.tools.provenance import ProvenanceStore, audit_briefing

SPEC = ChunkSpec(target=500, overlap=100, min_chars=50, snap_window=60)

MU_TEXT = (
    "Item 2. Management's Discussion and Analysis\n"
    "Revenue decreased 39% as compared to the prior quarter, primarily due to lower\n"
    "average selling prices per bit and lower bit shipments.\n"
    "Days of inventory increased to 214 days at the end of the period.\n"
    "Capital expenditure for fiscal 2023 is expected to be approximately $7.0 billion.\n"
    "Contract pricing for DRAM declined in the high twenties percent range.\n"
)


@pytest.fixture
def rig(tmp_path):
    store = ChunkStore(tmp_path / "idx")
    path = tmp_path / "d_mu.txt"
    path.write_text(MU_TEXT)
    rows, texts, cid = [], [], 1
    for ch in chunk_document("d_mu", MU_TEXT, SPEC):
        rows.append((cid, "d_mu", ch.ordinal, ch.start, ch.end, len(ch.text),
                     "MU", "0001", "2023-01-05", "10-Q", "mdna", str(path)))
        texts.append(ch.text)
        cid += 1
    store.add_batch(rows, texts, 1)
    prov = ProvenanceStore(tmp_path / "prov")
    return store, prov


def _loop(rig, outputs):
    store, prov = rig
    return ExtractionLoop(Searcher(store, None), ScriptedGenerator(outputs), prov,
                          ExtractConfig(max_passages=4, batch_passages=4)), prov


def test_end_to_end_produces_verifiable_briefing(rig):
    out = "\n".join([
        '{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"Days of inventory rose to 214."}',
        '{"kind":"inference","text":"This is a pricing correction, not a demand collapse."}',
    ])
    loop, prov = _loop(rig, [out])
    res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))

    assert res.briefing.claims
    audit = audit_briefing(res.briefing, prov)
    assert audit["passed"], "every accepted claim must resolve against retained bytes"
    assert audit["n_unverified_quotes"] == 0


def test_fabricated_claim_never_reaches_the_briefing(rig):
    out = "\n".join([
        '{"kind":"sourced","source":"S1","quote":"Gross margin expanded to 71 percent on favourable mix","text":"Margins expanded."}',
        '{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"Inventory 214 days."}',
    ])
    loop, prov = _loop(rig, [out])
    res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))

    assert res.stats.quote_fabricated == 1
    assert all("71 percent" not in (c.text or "") for c in res.briefing.claims)
    assert audit_briefing(res.briefing, prov)["passed"]


def test_briefing_with_no_survivors_is_still_valid(rig):
    loop, prov = _loop(rig, ['{"kind":"sourced","source":"S1","quote":"Entirely invented sentence about margins","text":"x"}'])
    res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))
    assert res.briefing.claims == []
    assert res.briefing.sourced_fraction == 0.0
    assert audit_briefing(res.briefing, prov)["passed"]


def test_as_of_bounds_what_the_model_is_shown(rig):
    loop, _ = _loop(rig, [""])
    res = loop.run("MU", SubSegment.MEMORY, date(2022, 1, 1))
    assert res.briefing.sources == [], "no filing existed on or before this as_of"


def test_empty_generation_yields_empty_briefing(rig):
    loop, _ = _loop(rig, [""])
    res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))
    assert res.briefing.claims == []
    assert res.briefing.sources, "retrieval still ran and retained what it found"


def test_prompt_contains_the_passages_verbatim(rig):
    loop, _ = _loop(rig, [""])
    loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))
    prompt = loop.generator.calls[0]
    assert "Days of inventory increased to 214 days" in prompt
    assert "TICKER: MU" in prompt
    assert "Memory" in prompt, "sub-segment framing must reach the model"


def test_segment_queries_differ_by_segment():
    mem = set(segment_queries(SubSegment.MEMORY))
    eq = set(segment_queries(SubSegment.EQUIPMENT))
    assert mem != eq
    assert any("book-to-bill" in q for q in eq)
    assert any("pricing" in q for q in mem)


class TestExport:
    def _briefing(self, rig):
        out = "\n".join([
            '{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"Days of inventory rose to 214."}',
            '{"kind":"inference","text":"Pricing correction rather than demand collapse."}',
        ])
        loop, _ = _loop(rig, [out])
        return loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))

    def test_markdown_separates_sourced_from_inference(self, rig):
        res = self._briefing(rig)
        md = to_markdown(res.briefing, res.audit)
        assert "## Sourced" in md and "## Inference" in md
        # The inference line must not appear inside the sourced section, or a skimming
        # reader treats the model's reasoning as something a filing said.
        sourced_block = md.split("## Sourced")[1].split("## Inference")[0]
        assert "Pricing correction" not in sourced_block

    def test_markdown_shows_the_quote_not_just_a_pointer(self, rig):
        res = self._briefing(rig)
        md = to_markdown(res.briefing, res.audit)
        assert "> Days of inventory increased to 214 days" in md

    def test_markdown_reports_the_audit(self, rig):
        res = self._briefing(rig)
        md = to_markdown(res.briefing, {**res.audit, "model": "test-model"})
        assert "Extraction audit" in md and "test-model" in md

    def test_empty_briefing_says_so_explicitly(self, rig):
        loop, _ = _loop(rig, [""])
        res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))
        md = to_markdown(res.briefing, res.audit)
        assert "No claim survived" in md, "silence must not read as 'no news'"

    def test_filename(self, rig):
        res = self._briefing(rig)
        assert filename(res.briefing) == "MU_2023-03-01_briefing.md"

    def test_no_structured_appendix_sections_when_none_attached(self, rig):
        res = self._briefing(rig)
        md = to_markdown(res.briefing, res.audit)
        assert "## Price context" not in md
        assert "## Insider activity" not in md
        assert "## Macro context" not in md

    def test_price_section_renders_when_attached(self, rig):
        from argus.contracts.context import PriceSnapshot
        res = self._briefing(rig)
        snap = PriceSnapshot(ticker="MU", as_of=date(2023, 3, 1), close=55.5,
                             volume=1_000_000, bars_used=252)
        briefing = res.briefing.model_copy(update={"price": snap})
        md = to_markdown(briefing, res.audit)
        assert "## Price context" in md
        assert "55.5" in md

    def test_insider_and_macro_sections_render_when_attached(self, rig):
        from argus.contracts.context import (InsiderActivity, InsiderTransactionCode,
                                              MacroContext, MacroSeries)
        res = self._briefing(rig)
        activity = InsiderActivity(ticker="MU", as_of=date(2023, 3, 1), transactions=[
            InsiderTransactionCode(
                filing_date=date(2023, 2, 1), transaction_date=date(2023, 1, 30),
                owner_name="JANE CFO", is_officer=True, is_director=False,
                is_ten_pct_owner=False, officer_title="CFO", code="S",
                is_open_market=True, shares=1000, price_per_share=55.0,
                shares_owned_after=9000, acquired=False)])
        macro = MacroContext(as_of=date(2023, 3, 1), series=[
            MacroSeries(series_id="FEDFUNDS", label="Fed funds rate", value=4.5,
                       value_date=date(2023, 2, 1), revision_safe=True)])
        briefing = res.briefing.model_copy(update={"insider_activity": activity,
                                                    "macro": macro})
        md = to_markdown(briefing, res.audit)
        assert "## Insider activity" in md and "JANE CFO" in md
        assert "## Macro context" in md and "Fed funds rate" in md

    def test_analyst_section_renders_when_attached(self, rig):
        from argus.contracts.context import AnalystConsensus, AnalystRating
        res = self._briefing(rig)
        consensus = AnalystConsensus(ticker="MU", as_of=date(2023, 3, 1), ratings=[
            AnalystRating(firm="Barclays", grade_date=date(2023, 2, 1), to_grade="Buy",
                         action="main", price_target=80.0)])
        briefing = res.briefing.model_copy(update={"analyst": consensus})
        md = to_markdown(briefing, res.audit)
        assert "## Analyst consensus" in md and "Barclays" in md


class TestBriefingContextFields:
    """Backward compatibility for the price/insider/macro appendices added to Briefing:
    every existing Briefing (baseline artifacts, reconstructed SFT briefings) must keep
    validating unchanged since these fields are optional and default to None."""

    def _minimal_briefing(self, rig):
        loop, _ = _loop(rig, [""])
        res = loop.run("MU", SubSegment.MEMORY, date(2023, 3, 1))
        return res.briefing

    def test_defaults_to_none_when_not_provided(self, rig):
        b = self._minimal_briefing(rig)
        assert b.price is None
        assert b.insider_activity is None
        assert b.macro is None

    def test_round_trips_through_json_with_appendices_attached(self, rig):
        from argus.contracts.context import PriceSnapshot
        b = self._minimal_briefing(rig)
        snap = PriceSnapshot(ticker="MU", as_of=date(2023, 3, 1), close=55.5,
                             volume=1_000_000, bars_used=252)
        b2 = b.model_copy(update={"price": snap})
        from argus.contracts.briefing import Briefing
        restored = Briefing.model_validate_json(b2.model_dump_json())
        assert restored.price.close == 55.5
