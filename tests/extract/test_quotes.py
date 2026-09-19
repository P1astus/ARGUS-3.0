"""Quote resolution -- the guard the frontier model's trust depends on.

`test_paraphrase_is_fabricated` and `test_near_miss_number_is_fabricated` are the two
that matter. A resolver loose enough to accept a paraphrase would let the local model
invent evidence that reads as a citation, and nothing downstream could catch it.
"""

from __future__ import annotations

from argus.engine.extract.quotes import (MIN_QUOTE_CHARS, QuoteStatus, resolve,
                                         resolve_in_any)

DOC = (
    "Item 2. Management's Discussion and Analysis\n"
    "Revenue for the first quarter of 2023 decreased 39% as compared to the fourth\n"
    "quarter of 2022, primarily due to decreases of approximately 25% in average\n"
    "selling prices per bit. Days of inventory increased to 214 days.\n"
)


def test_exact_substring():
    q = "Days of inventory increased to 214 days"
    r = resolve(DOC, q)
    assert r.status is QuoteStatus.EXACT
    assert r.text == q
    assert DOC[r.start:r.end] == q


def test_whitespace_repair_returns_document_bytes():
    # The model re-wrapped the line: same words, different newline placement.
    q = "Revenue for the first quarter of 2023 decreased 39% as compared to the fourth quarter of 2022"
    r = resolve(DOC, q)
    assert r.status is QuoteStatus.WHITESPACE
    # The stored text must be the DOCUMENT's span, not the model's rendering -- otherwise
    # the repair would launder unverifiable text into a "verified" claim.
    assert r.text != q
    assert r.text in DOC
    assert "\n" in r.text


def test_paraphrase_is_fabricated():
    r = resolve(DOC, "Revenue declined thirty-nine percent quarter over quarter in Q1")
    assert r.status is QuoteStatus.FABRICATED
    assert r.text is None


def test_near_miss_number_is_fabricated():
    # One digit changed. This is the dangerous case: it reads as a citation and is wrong.
    r = resolve(DOC, "Days of inventory increased to 241 days")
    assert r.status is QuoteStatus.FABRICATED


def test_too_short_quote_rejected():
    r = resolve(DOC, "Revenue")
    assert r.status is QuoteStatus.TOO_SHORT
    assert len("Revenue") < MIN_QUOTE_CHARS


def test_empty_quote_rejected():
    assert resolve(DOC, "").status is QuoteStatus.TOO_SHORT
    assert resolve(DOC, None).status is QuoteStatus.TOO_SHORT


def test_whitespace_only_quote_rejected():
    assert resolve(DOC, "   \n  \t ").status is QuoteStatus.TOO_SHORT


def test_resolve_in_any_prefers_cited_source():
    docs = {"a": DOC, "b": DOC}
    sid, r = resolve_in_any(docs, "Days of inventory increased to 214 days", preferred="b")
    assert sid == "b" and r.status is QuoteStatus.EXACT


def test_resolve_in_any_recovers_miscited_quote():
    other = "Backlog for the segment was $2.61 billion at the end of the quarter."
    docs = {"a": DOC, "b": other}
    sid, r = resolve_in_any(docs, "Backlog for the segment was $2.61 billion", preferred="a")
    assert sid == "b", "a real quote under the wrong id is a pointer error, not a lie"
    assert r.status is QuoteStatus.EXACT


def test_resolve_in_any_reports_fabrication():
    docs = {"a": DOC}
    sid, r = resolve_in_any(docs, "Gross margin expanded to 71% on favourable mix")
    assert sid is None and r.status is QuoteStatus.FABRICATED


def test_offsets_round_trip_after_whitespace_repair():
    doc = "alpha   beta\n\n\tgamma delta epsilon zeta eta theta"
    q = "beta gamma delta epsilon"
    r = resolve(doc, q, min_chars=8)
    assert r.status is QuoteStatus.WHITESPACE
    assert doc[r.start:r.end] == r.text
