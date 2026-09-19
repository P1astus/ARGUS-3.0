"""Model output -> validated `Claim`s, with every rejection counted.

The parser is deliberately unforgiving in one direction and forgiving in another.

Forgiving about *format*: a base model emits stray prose, trailing commas, half a line at
the token limit. None of that is evidence of dishonesty, so malformed lines are dropped
and counted rather than treated as failure of the whole extraction.

Unforgiving about *evidence*: a sourced claim whose quote cannot be located in the
document is dropped outright. It is never downgraded to an inference claim -- that would
launder a fabrication into an acceptable-looking output and quietly inflate the very
metric the Phase 4 gate measures. The claim is discarded and the fabrication is recorded.

Everything rejected lands in `ExtractionStats`, because a pipeline that only reports what
survived cannot tell you whether the model is getting better or just quieter.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from argus.contracts.briefing import Claim, ClaimKind
from argus.engine.extract.prompts import FEWSHOT_INFERENCE_TEXTS
from argus.engine.extract.quotes import QuoteStatus, resolve_in_any

log = logging.getLogger(__name__)

_LINE = re.compile(r"^\s*(\{.*\})\s*$")
MAX_TEXT_CHARS = 400
MAX_QUOTE_CHARS = 1200


@dataclass
class ExtractionStats:
    """Per-extraction accounting. Every counter is a rejection reason except the first."""

    lines_seen: int = 0
    lines_parsed: int = 0
    malformed_json: int = 0
    missing_fields: int = 0
    unknown_source: int = 0        # cited an id that was never shown
    quote_exact: int = 0
    quote_whitespace: int = 0      # located after whitespace normalisation
    quote_recited: int = 0         # right quote, wrong source id -- pointer corrected
    quote_too_short: int = 0
    quote_fabricated: int = 0      # not present in ANY shown document; claim dropped
    inference_claims: int = 0
    sourced_claims: int = 0
    inference_with_citation: int = 0   # tagged inference but cited a source anyway
    inference_copied_fewshot: int = 0  # verbatim copy of a demonstrated example, dropped
    rejected: list[dict] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return self.sourced_claims + self.inference_claims

    @property
    def quote_attempts(self) -> int:
        return (self.quote_exact + self.quote_whitespace + self.quote_recited
                + self.quote_too_short + self.quote_fabricated)

    @property
    def exact_quote_rate(self) -> float:
        """Headline honesty metric: share of quote attempts copied character-perfect.

        Reported ahead of the repaired rate on purpose. If only the post-repair number
        were tracked, a model that paraphrased everything and a model that copied
        everything would score identically.
        """
        n = self.quote_attempts
        return self.quote_exact / n if n else float("nan")

    @property
    def fabrication_rate(self) -> float:
        n = self.quote_attempts
        return self.quote_fabricated / n if n else float("nan")

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "rejected"}
        d.update(accepted=self.accepted, quote_attempts=self.quote_attempts,
                 exact_quote_rate=self.exact_quote_rate,
                 fabrication_rate=self.fabrication_rate,
                 n_rejected_examples=len(self.rejected))
        return d

    def merge(self, other: ExtractionStats) -> None:
        for k, v in other.__dict__.items():
            if k == "rejected":
                self.rejected.extend(v[:5])
            else:
                setattr(self, k, getattr(self, k) + v)


def iter_json_lines(text: str) -> list[dict]:
    """Pull JSON objects out of a generation, one per line, skipping anything else."""
    out = []
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            out.append({"__malformed__": raw[:200]})
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_claims(text: str, documents: dict[str, str],
                 display_to_source: dict[str, str],
                 stats: ExtractionStats | None = None) -> tuple[list[Claim], ExtractionStats]:
    """Turn a generation into Claims.

    `documents` maps provenance source_id -> the exact text retained for it.
    `display_to_source` maps the short label shown in the prompt (S1, S2...) to that id;
    the model sees short labels because 24-hex-character ids invite transcription errors
    that would be indistinguishable from citing the wrong document.
    """
    st = stats or ExtractionStats()
    claims: list[Claim] = []

    for obj in iter_json_lines(text):
        st.lines_seen += 1

        if "__malformed__" in obj:
            st.malformed_json += 1
            st.rejected.append({"reason": "malformed_json", "line": obj["__malformed__"]})
            continue

        kind = str(obj.get("kind", "")).strip().lower()
        body = str(obj.get("text", "")).strip()
        if kind not in ("sourced", "inference") or not body:
            st.missing_fields += 1
            st.rejected.append({"reason": "missing_fields", "line": str(obj)[:200]})
            continue
        st.lines_parsed += 1
        body = body[:MAX_TEXT_CHARS]

        if kind == "inference":
            if body in FEWSHOT_INFERENCE_TEXTS:
                # The model reproduced one of the prompt's own worked examples verbatim
                # instead of reasoning about the passages it was actually shown -- a base
                # model over-anchoring on a demonstrated exemplar's CONTENT rather than
                # just its format/pattern. Measured live: a GOOGL extraction's only
                # inference line was character-for-character the MU/memory demo's own
                # sentence, asserting a revenue DECLINE while GOOGL's sourced claims in
                # the same briefing showed revenue up 24%. That is not evidence about the
                # ticker being extracted; it is prompt leakage, dropped the same way a
                # fabricated sourced quote is dropped rather than kept and mislabelled.
                st.inference_copied_fewshot += 1
                st.rejected.append({"reason": "inference_copied_fewshot", "text": body})
                continue
            if obj.get("source") or obj.get("quote"):
                # The model tagged it inference but attached evidence. Honour the tag and
                # strip the citation: the schema forbids the combination, and the tag is
                # the model's own statement that it is not asserting the source said this.
                st.inference_with_citation += 1
            claims.append(Claim(text=body, kind=ClaimKind.INFERENCE))
            st.inference_claims += 1
            continue

        quote = str(obj.get("quote") or "")[:MAX_QUOTE_CHARS]
        display = str(obj.get("source") or "").strip()
        preferred = display_to_source.get(display)
        if display and preferred is None:
            st.unknown_source += 1

        sid, res = resolve_in_any(documents, quote, preferred=preferred)

        if res.status == QuoteStatus.TOO_SHORT:
            st.quote_too_short += 1
            st.rejected.append({"reason": "quote_too_short", "quote": quote[:120],
                                "text": body[:120]})
            continue
        if not res.ok or sid is None:
            st.quote_fabricated += 1
            st.rejected.append({"reason": "quote_not_in_any_source", "quote": quote[:200],
                                "cited": display, "text": body[:120]})
            continue

        if res.status == QuoteStatus.EXACT:
            st.quote_exact += 1
        else:
            st.quote_whitespace += 1
        if preferred is not None and sid != preferred:
            st.quote_recited += 1

        # `res.text` is the DOCUMENT's span, not the model's rendering of it, so the
        # stored quote is verbatim by construction and Briefing.unverified_quotes()
        # measures retrieval/provenance integrity rather than the model's typing.
        claims.append(Claim(text=body, kind=ClaimKind.SOURCED,
                            source_id=sid, quote=res.text))
        st.sourced_claims += 1

    return claims, st
