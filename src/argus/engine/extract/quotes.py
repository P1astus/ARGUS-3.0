"""Verbatim quote resolution -- the mechanical guard on the whole pipeline.

THE FAILURE MODE THIS EXISTS TO STOP
The frontier model reads a briefing and cannot see the filings behind it. If the local
model paraphrases a sentence and presents it as a quotation, the frontier model reasons
confidently on top of a fabrication and there is nothing downstream that could notice.
So a quote is not trusted; it is *located*. A claim survives only if its quote can be
found in the retained document, and the text stored on the Claim is then the document's
own bytes, not the model's rendering of them.

THREE OUTCOMES, COUNTED SEPARATELY
  exact      -- the quote is already a literal substring
  whitespace -- it matches once runs of whitespace are treated as equivalent
  fabricated -- it does not appear at all; the claim is dropped

The distinction is not pedantry. `whitespace` is a formatting artifact (models
re-wrap lines, and this corpus is full of hard-wrapped filing text); `fabricated` is the
model inventing evidence. Collapsing them into one "verification rate" would hide the
only number that measures honesty. The baseline report headlines the *exact* rate
precisely so that repairs cannot flatter it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_WS = re.compile(r"\s+")

# Below this a "quote" is not evidence -- a 5-character span appears in any document by
# chance, so verifying it proves nothing about whether the model read the source.
MIN_QUOTE_CHARS = 24


class QuoteStatus(StrEnum):
    EXACT = "exact"
    WHITESPACE = "whitespace"
    FABRICATED = "fabricated"
    TOO_SHORT = "too_short"


@dataclass(frozen=True)
class Resolution:
    status: QuoteStatus
    start: int | None = None
    end: int | None = None
    text: str | None = None      # the DOCUMENT's bytes, never the model's

    @property
    def ok(self) -> bool:
        return self.status in (QuoteStatus.EXACT, QuoteStatus.WHITESPACE)


def _normalise_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse whitespace, keeping a map from normalised index -> original index.

    The map is what makes the repair safe: after matching in normalised space, the span
    is read back out of the *original* string, so the stored quote is still literally
    present in the document. Repairing by substituting the model's text would defeat the
    check it is supposed to pass.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_ws = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_ws and out:
                out.append(" ")
                idx.append(i)
            prev_ws = True
        else:
            out.append(ch)
            idx.append(i)
            prev_ws = False
    return "".join(out), idx


def resolve(document: str, quote: str, min_chars: int = MIN_QUOTE_CHARS) -> Resolution:
    """Locate `quote` in `document`, tolerating whitespace differences only."""
    q = (quote or "").strip()
    if len(q) < min_chars:
        return Resolution(QuoteStatus.TOO_SHORT)

    i = document.find(q)
    if i >= 0:
        return Resolution(QuoteStatus.EXACT, i, i + len(q), document[i:i + len(q)])

    norm_doc, idx = _normalise_with_map(document)
    norm_q = _WS.sub(" ", q).strip()
    if not norm_q:
        return Resolution(QuoteStatus.TOO_SHORT)

    j = norm_doc.find(norm_q)
    if j < 0:
        return Resolution(QuoteStatus.FABRICATED)

    start = idx[j]
    end = idx[j + len(norm_q) - 1] + 1
    return Resolution(QuoteStatus.WHITESPACE, start, end, document[start:end])


def resolve_in_any(documents: dict[str, str], quote: str,
                   preferred: str | None = None,
                   min_chars: int = MIN_QUOTE_CHARS) -> tuple[str | None, Resolution]:
    """Resolve against the cited document, falling back to a scan of the others.

    Models cite the wrong id surprisingly often while quoting a real passage. That is a
    different error from fabrication and worth separating: the evidence exists, the
    pointer is wrong. Recovering it keeps the claim, and the caller records that the
    citation was corrected rather than pretending the model got it right.
    """
    if preferred and preferred in documents:
        r = resolve(documents[preferred], quote, min_chars)
        if r.ok:
            return preferred, r
        if r.status == QuoteStatus.TOO_SHORT:
            return None, r

    for sid, doc in documents.items():
        if sid == preferred:
            continue
        r = resolve(doc, quote, min_chars)
        if r.ok:
            return sid, r
    return None, Resolution(QuoteStatus.FABRICATED)
