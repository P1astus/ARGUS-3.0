"""Chunking invariants.

The one that matters is `test_chunks_are_literal_substrings`: it is the property the
entire Claim/quote audit rests on. If chunking ever normalises text, a faithful quote
from the model becomes unverifiable and the Phase 4 gate silently starts failing honest
extractions.
"""

from __future__ import annotations

import pytest

from argus.engine.retrieval.chunk import Chunk, ChunkSpec, chunk_document, expand

SPEC = ChunkSpec(target=400, overlap=80, min_chars=50, snap_window=60)

DOC = "\n".join(
    f"Paragraph {i}. Book-to-bill was {1.0 + i / 100:.2f} in the quarter, and "
    f"management noted that distributor inventory remained elevated at {i} weeks."
    for i in range(40)
)


def test_chunks_are_literal_substrings():
    for ch in chunk_document("d1", DOC, SPEC):
        assert DOC[ch.start:ch.end] == ch.text
        assert ch.text in DOC


def test_chunks_cover_the_document():
    chunks = chunk_document("d1", DOC, SPEC)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(DOC)
    # Consecutive chunks must not leave a hole, or content becomes unretrievable.
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end


def test_chunks_overlap_and_advance():
    chunks = chunk_document("d1", DOC, SPEC)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert b.start > a.start, "chunking must advance or it never terminates"
        assert b.ordinal == a.ordinal + 1


def test_short_document_yields_nothing():
    assert chunk_document("d1", "too short", SPEC) == []


def test_single_chunk_document():
    text = "x" * 300
    chunks = chunk_document("d1", text, SPEC)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_no_whitespace_only_chunks():
    text = "Real content here.\n" + "\n" * 500 + "More real content follows here.\n"
    for ch in chunk_document("d1", text, SPEC):
        assert ch.text.strip()


def test_pathological_no_breaks_still_terminates():
    text = "a" * 5000        # no newline, no sentence end: _snap can find nothing
    chunks = chunk_document("d1", text, SPEC)
    assert chunks
    assert chunks[-1].end == len(text)
    assert "".join(c.text for c in chunks[:1]) in text


def test_empty_span_rejected():
    with pytest.raises(ValueError):
        Chunk("d", 0, 10, 10, "")


def test_expand_stays_within_document_and_verbatim():
    lo, hi = expand(DOC, 1000, 1200, 600, 900)
    assert 0 <= lo <= 1000 < 1200 <= hi <= len(DOC)
    assert DOC[1000:1200] in DOC[lo:hi]


def test_expand_clamps_at_edges():
    lo, hi = expand(DOC, 0, 50, 10_000, 10_000)
    assert (lo, hi) == (0, len(DOC))


class TestExpandSentenceBoundary:
    """Traced fabrication: a passage cut off mid-sentence ("...had a valuation
    allowance of $") was completed by the model with figures present nowhere in the
    shown text. `expand()` must not hand out a passage that ends mid-sentence when a
    complete sentence boundary exists within reach, without ever cutting into the hit
    itself or below it."""

    # Four sentences. A hit on the first, with `after` reaching partway into the third
    # (past the second sentence's clean end), is the real-world shape: the raw window
    # lands mid-sentence while a complete boundary sits just behind it.
    TEXT = ("First sentence hit here. Second sentence ends cleanly right here. "
           "Third sentence keeps going for quite a long stretch without stopping. "
           "Fourth sentence.")

    def test_trims_a_dangling_tail_to_the_last_sentence_end(self):
        start = 0
        end = self.TEXT.index("First sentence hit here.") + len("First sentence hit here.")
        third_start = self.TEXT.index("Third sentence")
        after = (third_start + 20) - end   # land 20 chars into the third sentence
        lo, hi = expand(self.TEXT, start, end, before=0, after=after)

        assert self.TEXT[hi - 1] in ".!?\""
        assert hi <= self.TEXT.index("Third sentence") + 20, "must have trimmed back"
        assert self.TEXT[start:end] in self.TEXT[lo:hi], "must not cut into the hit itself"
        assert self.TEXT[:hi].endswith("ends cleanly right here."), (
            "must land on the SECOND sentence's end, the last complete one in reach")

    def test_never_trims_below_the_hit_end(self):
        # The hit itself ends mid-sentence (partway into "Third sentence..."); expand()
        # must not shrink hi below that, even though the result still ends mid-sentence.
        start = 0
        end = self.TEXT.index("Third sentence") + 10   # end is itself mid-sentence
        lo, hi = expand(self.TEXT, start, end, before=0, after=0)
        assert hi >= end

    def test_leaves_a_clean_ending_untouched(self):
        end = self.TEXT.index("Second sentence") + len("Second sentence ends cleanly right here.")
        lo, hi = expand(self.TEXT, 0, end, before=0, after=0)
        assert hi == end
        assert self.TEXT[hi - 1] == "."

    def test_no_sentence_end_in_window_leaves_hi_unchanged(self):
        # Dense text with no terminal punctuation anywhere nearby (e.g. a table) --
        # expand() must not shrink the passage to nothing chasing a boundary that isn't
        # there.
        text = "no punctuation anywhere in this entire passage at all " * 20
        start, end = 100, 150
        lo, hi = expand(text, start, end, before=20, after=300)
        assert hi >= end
        assert text[start:end] in text[lo:hi]
