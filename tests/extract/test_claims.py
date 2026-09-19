"""Claim parsing and rejection accounting.

The invariant under test throughout: a claim whose quote cannot be located is DROPPED,
never downgraded to an inference claim. Downgrading would turn a fabrication into
well-formed output and would make `sourced_fraction` look worse while making the briefing
look fine -- the exact opposite of what the audit needs.
"""

from __future__ import annotations

import json

from argus.contracts.briefing import Briefing, ClaimKind, Source, SourceType
from argus.engine.extract.claims import ExtractionStats, iter_json_lines, parse_claims

DOC_A = ("Revenue for the first quarter of 2023 decreased 39% as compared to the fourth\n"
         "quarter of 2022. Days of inventory increased to 214 days.\n")
DOC_B = "Backlog for the segment was $2.61 billion at the end of the quarter.\n"

DOCS = {"aaa": DOC_A, "bbb": DOC_B}
DISPLAY = {"S1": "aaa", "S2": "bbb"}


def _parse(text):
    return parse_claims(text, DOCS, DISPLAY)


def test_sourced_claim_accepted_and_quote_is_document_bytes():
    out = ('{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 '
           'days","text":"Days of inventory rose to 214."}')
    claims, st = _parse(out)
    assert len(claims) == 1
    c = claims[0]
    assert c.kind is ClaimKind.SOURCED and c.source_id == "aaa"
    assert c.quote in DOC_A
    assert st.quote_exact == 1 and st.sourced_claims == 1


def test_inference_claim_accepted_without_citation():
    claims, st = _parse('{"kind":"inference","text":"This looks like a pricing correction."}')
    assert len(claims) == 1 and claims[0].kind is ClaimKind.INFERENCE
    assert claims[0].source_id is None and claims[0].quote is None
    assert st.inference_claims == 1


def test_fabricated_quote_is_dropped_not_downgraded():
    out = ('{"kind":"sourced","source":"S1","quote":"Gross margin expanded to 71 percent '
           'on favourable product mix","text":"Gross margin expanded to 71%."}')
    claims, st = _parse(out)
    assert claims == [], "a fabricated quote must not survive in any form"
    assert st.quote_fabricated == 1
    assert st.inference_claims == 0, "must not be laundered into an inference claim"
    assert st.rejected and st.rejected[0]["reason"] == "quote_not_in_any_source"


def test_inference_tagged_but_citing_keeps_tag_and_strips_citation():
    out = ('{"kind":"inference","source":"S1","quote":"Days of inventory increased to 214 '
           'days","text":"Inventory is the tell here."}')
    claims, st = _parse(out)
    assert len(claims) == 1 and claims[0].kind is ClaimKind.INFERENCE
    assert claims[0].source_id is None
    assert st.inference_with_citation == 1


def test_miscited_but_real_quote_is_recovered_and_counted():
    out = ('{"kind":"sourced","source":"S1","quote":"Backlog for the segment was $2.61 '
           'billion","text":"Backlog was $2.61bn."}')
    claims, st = _parse(out)
    assert len(claims) == 1 and claims[0].source_id == "bbb"
    assert st.quote_recited == 1


def test_unknown_source_id_counted():
    out = ('{"kind":"sourced","source":"S9","quote":"Days of inventory increased to 214 '
           'days","text":"Inventory 214 days."}')
    claims, st = _parse(out)
    assert len(claims) == 1, "the evidence is real even though the pointer was invented"
    assert st.unknown_source == 1


def test_malformed_lines_counted_and_do_not_abort_the_rest():
    out = "\n".join([
        '{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"a"}',
        '{"kind":"sourced", "quote": broken json here}',
        'Here is some prose the base model emitted.',
        '{"kind":"inference","text":"Still parsed."}',
    ])
    claims, st = _parse(out)
    assert len(claims) == 2
    assert st.malformed_json == 1


def test_truncated_final_line_costs_one_claim_only():
    out = ('{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"a"}\n'
           '{"kind":"sourced","source":"S1","quote":"Revenue for the first')
    claims, st = _parse(out)
    assert len(claims) == 1, "JSON Lines must localise a truncation to one claim"


def test_missing_text_rejected():
    claims, st = _parse('{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days"}')
    assert claims == [] and st.missing_fields == 1


def test_short_quote_rejected_and_counted_separately():
    out = '{"kind":"sourced","source":"S1","quote":"Revenue","text":"Revenue fell."}'
    claims, st = _parse(out)
    assert claims == []
    assert st.quote_too_short == 1 and st.quote_fabricated == 0


def test_exact_rate_ignores_repairs():
    exact = ('{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 '
             'days","text":"a"}')
    rewrapped = ('{"kind":"sourced","source":"S1","quote":"Revenue for the first quarter '
                 'of 2023 decreased 39% as compared to the fourth quarter of 2022","text":"b"}')
    claims, st = _parse(exact + "\n" + rewrapped)
    assert len(claims) == 2
    assert st.quote_exact == 1 and st.quote_whitespace == 1
    assert st.exact_quote_rate == 0.5, "repairs must not flatter the honesty metric"


def test_accepted_claims_always_pass_the_briefing_audit():
    """End-to-end: whatever survives parsing must satisfy unverified_quotes()."""
    out = "\n".join([
        '{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"a"}',
        '{"kind":"sourced","source":"S1","quote":"Revenue for the first quarter of 2023 decreased 39% as compared to the fourth quarter of 2022","text":"b"}',
        '{"kind":"sourced","source":"S1","quote":"Gross margin expanded to 71 percent on mix","text":"fabricated"}',
        '{"kind":"inference","text":"c"}',
    ])
    claims, st = _parse(out)
    from datetime import date, datetime, timezone
    sources = [Source(source_id=sid, source_type=SourceType.SEC_FILING,
                      retrieved_at=datetime.now(timezone.utc), content_sha256="0" * 64)
               for sid in DOCS]
    from argus.contracts.quant import SubSegment
    b = Briefing(ticker="MU", sub_segment=SubSegment.MEMORY, as_of=date(2023, 3, 1),
                 claims=claims, sources=sources)
    assert b.unverified_quotes(DOCS) == []
    assert st.quote_fabricated == 1


def test_iter_json_lines_skips_prose_and_fences():
    text = "```json\n{\"kind\":\"inference\",\"text\":\"x\"}\n```\nsome prose\n"
    objs = iter_json_lines(text)
    assert objs == [{"kind": "inference", "text": "x"}]


def test_stats_merge_accumulates():
    a, b = ExtractionStats(), ExtractionStats()
    a.quote_exact, a.sourced_claims = 3, 3
    b.quote_exact, b.quote_fabricated, b.sourced_claims = 1, 2, 1
    a.merge(b)
    assert a.quote_exact == 4 and a.quote_fabricated == 2 and a.accepted == 4


class TestInferenceCopiedFromFewshot:
    """Measured live: a GOOGL extraction's only inference line was a character-for-
    character copy of prompts.py's own MU/memory demo sentence ("...pricing correction
    rather than a demand collapse") -- asserting a revenue DECLINE while that same
    briefing's sourced claims showed GOOGL revenue up 24%. The model echoed its prompt
    instead of reasoning about the ticker in front of it; this is the drop-and-count for
    that specific, now-confirmed failure mode."""

    def test_verbatim_copy_of_a_fewshot_demo_is_dropped_and_counted(self):
        from argus.engine.extract.prompts import FEWSHOT_INFERENCE_TEXTS
        copied = next(iter(FEWSHOT_INFERENCE_TEXTS))
        claims, st = _parse(json.dumps({"kind": "inference", "text": copied}))
        assert claims == []
        assert st.inference_copied_fewshot == 1
        assert st.inference_claims == 0
        assert st.rejected[0]["reason"] == "inference_copied_fewshot"

    def test_all_three_fewshot_demo_sentences_are_caught(self):
        from argus.engine.extract.prompts import FEWSHOT_INFERENCE_TEXTS
        assert len(FEWSHOT_INFERENCE_TEXTS) == 3
        for text in FEWSHOT_INFERENCE_TEXTS:
            claims, st = _parse(json.dumps({"kind": "inference", "text": text}))
            assert claims == [], text

    def test_a_genuinely_new_inference_is_not_flagged(self):
        claims, st = _parse('{"kind":"inference","text":"This is a novel observation '
                            'about this specific filing, not from the demo."}')
        assert len(claims) == 1
        assert st.inference_copied_fewshot == 0

    def test_paraphrased_not_verbatim_is_not_caught(self):
        # Documents the real limitation: this check is exact-match only, so a paraphrase
        # of a demo sentence (rather than a verbatim copy) is NOT caught here. Honest
        # about scope rather than implying a fuzzy-match guarantee that doesn't exist.
        claims, st = _parse('{"kind":"inference","text":"Mostly a pricing correction, '
                            'not really a demand problem."}')
        assert len(claims) == 1
        assert st.inference_copied_fewshot == 0
