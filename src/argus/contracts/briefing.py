"""Stage 2-4 output schema: sourced briefings.

THE PHASE 4 GATE, MADE MECHANICAL
The gate is "manually audit briefings to confirm the model correctly separates 'confirmed
from source' vs 'my inference'". Auditing free-form prose for that is subjective and does
not scale.

So the schema enforces it instead: every Claim carries EITHER a source id plus the exact
quoted span that supports it, OR an explicit INFERENCE tag. There is no third option -- a
Claim that cites a source must quote it, and the quote must be verifiable against the
retained raw document. The audit becomes "sample 50 claims, check each span resolves",
which is mechanical and can be automated.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from argus.contracts.context import (AnalystConsensus, InsiderActivity, MacroContext,
                                     PriceSnapshot)
from argus.contracts.quant import SubSegment


class ClaimKind(StrEnum):
    SOURCED = "sourced"      # supported by a quoted span in a retained document
    INFERENCE = "inference"  # the model's own reasoning; NOT attributable to a source


class SourceType(StrEnum):
    SEC_FILING = "sec_filing"
    NEWS = "news"
    WEB = "web"
    PRICE_DATA = "price_data"


class Source(BaseModel):
    """A retrieved document, retained so claims can be audited against it."""

    source_id: str = Field(description="Stable id; also the key in the provenance store")
    source_type: SourceType
    url: str | None = None
    title: str | None = None
    published: date | None = None
    retrieved_at: datetime
    content_sha256: str = Field(description="Hash of the retained raw content")

    # Licensing matters: EDGAR is public domain, most news APIs prohibit redistribution
    # and model training. Recorded per source so a corpus build can filter on it rather
    # than relying on someone remembering the terms.
    redistributable: bool = False
    training_permitted: bool = False


class Claim(BaseModel):
    """One assertion in a briefing, with its evidential status."""

    text: str
    kind: ClaimKind
    source_id: str | None = None
    quote: str | None = Field(
        default=None,
        description="Verbatim span from the source that supports `text`. Must appear in "
                    "the retained document -- this is what makes the audit mechanical.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_evidence(self) -> Claim:
        if self.kind == ClaimKind.SOURCED:
            if not self.source_id or not self.quote:
                raise ValueError(
                    "a SOURCED claim requires both source_id and a verbatim quote; "
                    "if the model is reasoning rather than citing, tag it INFERENCE")
        else:
            if self.source_id or self.quote:
                raise ValueError(
                    "an INFERENCE claim must not cite a source -- that is precisely the "
                    "conflation the Phase 4 gate exists to catch")
        return self


class Briefing(BaseModel):
    """Sub-segment-aware briefing assembled from tool output."""

    ticker: str
    sub_segment: SubSegment
    as_of: date
    claims: list[Claim]
    sources: list[Source]
    tool_calls: int = 0

    # Structured appendices -- price/insider/macro data, displayed directly rather than
    # run through claim extraction (contracts/context.py's docstring explains why: there
    # is nothing to paraphrase in a number pulled straight from a provider, so the
    # quote-verification machinery that exists to catch prose fabrication does not apply).
    # All optional and default to None/empty so existing Briefings (baseline artifacts,
    # reconstructed SFT briefings) keep validating unchanged.
    price: PriceSnapshot | None = None
    insider_activity: InsiderActivity | None = None
    macro: MacroContext | None = None
    analyst: AnalystConsensus | None = None

    @model_validator(mode="after")
    def _claims_reference_known_sources(self) -> Briefing:
        known = {s.source_id for s in self.sources}
        for c in self.claims:
            if c.kind == ClaimKind.SOURCED and c.source_id not in known:
                raise ValueError(
                    f"claim cites unknown source {c.source_id!r}; every cited source must "
                    "be retained so the quote can be verified")
        return self

    @property
    def sourced_fraction(self) -> float:
        """Share of claims backed by a quoted source.

        Tracked because it is the headline number for the Phase 4 audit: a briefing that
        is 90% inference is not a briefing, it is speculation with citations attached.
        """
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.kind == ClaimKind.SOURCED) / len(self.claims)

    def unverified_quotes(self, documents: dict[str, str]) -> list[Claim]:
        """Claims whose quote does not appear in the retained document.

        The mechanical form of the Phase 4 audit. A non-empty result means the model
        fabricated or paraphrased a quote while presenting it as verbatim.
        """
        bad = []
        for c in self.claims:
            if c.kind != ClaimKind.SOURCED:
                continue
            doc = documents.get(c.source_id or "")
            if doc is None or (c.quote or "") not in doc:
                bad.append(c)
        return bad
