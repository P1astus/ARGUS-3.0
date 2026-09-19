"""Document -> passage chunking, with exact character offsets.

WHY OFFSETS ARE LOAD-BEARING
The Claim schema requires a verbatim quote that resolves against the retained document.
That only holds if the text shown to the model is a *literal substring* of that document.
So a chunk is not "cleaned text" -- it is `doc[start:end]` and nothing else. No stripping,
no whitespace collapsing, no re-joining of paragraphs. Anything that rewrites the span
would make a faithful quote fail verification and an unfaithful one indistinguishable
from it.

TWO SIZES, ON PURPOSE
Retrieval wants small passages (precision, and bge-small truncates at 512 tokens);
the extracting model wants a wide enough window to see what a sentence refers to. So a
chunk is ~1800 chars for scoring, and `expand()` widens a hit to its neighbours before
the text is handed to the LLM. The expansion is still pure slicing, so quotes still hold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Paragraph boundary. Filings clean out to single-\n-separated paragraphs, so a blank
# line is rare; a bare newline is the real structural unit here.
_BREAK = re.compile(r"\n")

# Sentence-ish boundary, used only as a fallback inside a very long paragraph.
_SENT = re.compile(r"(?<=[.!?])\s")


@dataclass(frozen=True)
class ChunkSpec:
    """Chunking geometry.

    `target` is in characters rather than tokens because offsets are the contract and
    tokens are not addressable in the source text. Measured on this corpus, bge-small's
    wordpiece runs ~5.0 chars/token on filing prose, so 1800 chars lands around 360
    tokens -- comfortably inside the 512 limit at p95 (see docs/retrieval.md).
    """

    target: int = 1800
    overlap: int = 300
    min_chars: int = 200        # below this a chunk is boilerplate residue, not content
    snap_window: int = 250      # how far to search for a clean break near the target


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    ordinal: int
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("empty chunk span")


def _snap(text: str, pos: int, window: int) -> int:
    """Move `pos` to the nearest structural break within `window` chars.

    Searches forward first: ending a chunk *after* a paragraph break keeps the trailing
    sentence whole, which matters because a truncated sentence is exactly the kind of
    thing a model paraphrases rather than quotes.
    """
    if pos >= len(text):
        return len(text)

    ahead = text[pos:pos + window]
    m = _BREAK.search(ahead)
    if m:
        return pos + m.end()

    behind_start = max(0, pos - window)
    behind = text[behind_start:pos]
    hits = list(_BREAK.finditer(behind))
    if hits:
        return behind_start + hits[-1].end()

    m = _SENT.search(text[pos:pos + window])
    if m:
        return pos + m.end()
    return pos


def chunk_document(doc_id: str, text: str, spec: ChunkSpec | None = None) -> list[Chunk]:
    """Split into overlapping passages that are exact substrings of `text`."""
    spec = spec or ChunkSpec()
    if len(text) < spec.min_chars:
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    n = len(text)

    while start < n:
        raw_end = min(start + spec.target, n)
        end = raw_end if raw_end >= n else _snap(text, raw_end, spec.snap_window)
        end = max(end, start + spec.min_chars)
        end = min(end, n)

        span = text[start:end]
        if span.strip():
            chunks.append(Chunk(doc_id, ordinal, start, end, span))
            ordinal += 1

        if end >= n:
            break
        # Step from the *end* so overlap is measured against what was actually emitted;
        # stepping from `start + target` drifts whenever _snap moves the boundary.
        nxt = end - spec.overlap
        start = nxt if nxt > start else end

    return chunks


# Sentence-terminal punctuation, optionally followed by a closing quote. Used only to
# find a place to STOP a passage -- narrower than a full sentence-splitter needs to be.
_SENT_END = re.compile(r'[.!?]"?')


def expand(text: str, start: int, end: int, before: int, after: int) -> tuple[int, int]:
    """Widen a hit's span for reading, still by pure slicing.

    Returned bounds are snapped outward to paragraph breaks so the model is not handed a
    fragment beginning mid-sentence -- but they are bounds into the same string, so
    `text[start:end]` remains verbatim.

    MEASURED: before the sentence-boundary trim below existed, 20 of 30 sampled passages
    (67%) ended mid-sentence -- the paragraph-snap only fires when a newline happens to
    fall in the last 400 chars, which dense running prose (accounting notes, in
    particular) often does not have. A passage that stops mid-clause invites exactly the
    failure this architecture is built to prevent: the model "finishing" the sentence from
    whatever it already knows rather than from what it was shown. One traced case did
    exactly that -- a passage cut off after "...had a valuation allowance of $" and the
    extractor completed it with two specific dollar figures present nowhere in the shown
    text. The quote resolver correctly rejected it as unverifiable, but the invitation to
    fabricate should not be there in the first place.
    """
    lo = max(0, start - before)
    hi = min(len(text), end + after)
    if lo > 0:
        nl = text.find("\n", lo, min(lo + 400, start))
        if nl != -1:
            lo = nl + 1
    if hi < len(text):
        nl = text.rfind("\n", max(end, hi - 400), hi)
        if nl != -1:
            hi = nl

    # If the end still doesn't land on a sentence boundary, trim back to the last one
    # within a short lookback -- never below `end`, so the hit itself is never cut. If no
    # sentence end is found in the window (dense text, e.g. a table), leave `hi` as-is
    # rather than shrink the passage to nothing.
    if hi > 0 and not _SENT_END.match(text, hi - 1):
        last_end = None
        for m in _SENT_END.finditer(text, max(end, hi - 300), hi):
            last_end = m.end()
        if last_end is not None:
            hi = last_end

    return lo, hi
