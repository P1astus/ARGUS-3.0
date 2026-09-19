"""Stages 2-4: sub-segment-aware briefing assembly.

Tools retrieve, this module assembles, and the Claim schema enforces the separation
between what a source said and what the model concluded. Every retrieved document is
retained in the provenance store at retrieval time, so a claim's quote can be verified
later against exactly the bytes the model saw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from argus.contracts.briefing import Briefing, Claim, ClaimKind, Source, SourceType
from argus.contracts.quant import SubSegment
from argus.tools.provenance import ProvenanceStore
from argus.tools.taxonomy import prompt_context

log = logging.getLogger(__name__)


class Retriever(Protocol):
    """One evidence source (EDGAR, news API, web search)."""

    name: str

    def fetch(self, ticker: str, as_of: date, limit: int) -> list[tuple[str, dict]]:
        """Return [(content, metadata)] for documents dated on or before `as_of`."""
        ...

    def source_type(self) -> SourceType: ...

    def license_flags(self) -> tuple[bool, bool]:
        """(redistributable, training_permitted).

        EDGAR is public domain. Most news APIs prohibit both -- recorded per source so a
        later corpus build can filter mechanically instead of relying on memory.
        """
        ...


@dataclass
class BriefingConfig:
    max_docs_per_source: int = 5
    lookback_days: int = 90
    require_sourced_fraction: float = 0.5
    include_taxonomy: bool = True


class BriefingBuilder:
    """Assembles briefings from registered retrievers."""

    def __init__(self, store: ProvenanceStore, retrievers: list[Retriever] | None = None,
                 config: BriefingConfig | None = None) -> None:
        self.store = store
        self.retrievers = retrievers or []
        self.config = config or BriefingConfig()

    def gather(self, ticker: str, as_of: date) -> list[Source]:
        """Retrieve and retain documents. No claims are made here.

        Retrieval and claim-making are separated on purpose: a retriever that also
        asserted things would make it impossible to tell later whether a claim came from
        a document or from the model.
        """
        sources: list[Source] = []
        for r in self.retrievers:
            try:
                docs = r.fetch(ticker, as_of, self.config.max_docs_per_source)
            except Exception as e:
                log.warning("%s retrieval failed for %s: %s", r.name, ticker,
                            type(e).__name__)
                continue

            redistributable, training_ok = r.license_flags()
            for content, meta in docs:
                sources.append(self.store.put(
                    content=content, source_type=r.source_type(),
                    url=meta.get("url"), title=meta.get("title"),
                    published=meta.get("published"),
                    redistributable=redistributable, training_permitted=training_ok,
                ))
        return sources

    def context_prompt(self, ticker: str, seg: SubSegment, sources: list[Source]) -> str:
        """Render the prompt shown to the model, including the cycle framing."""
        parts = [f"TICKER: {ticker}", f"AS OF: {date.today().isoformat()}"]
        if self.config.include_taxonomy:
            parts.append("")
            parts.append(prompt_context(seg))

        parts += ["", "SOURCES (cite by id; quote verbatim when asserting a fact):"]
        for s in sources:
            text = self.store.get_text(s.source_id) or ""
            parts.append(f"\n[{s.source_id}] {s.title or s.url or s.source_type} "
                         f"({s.published or 'undated'})\n{text[:4000]}")

        parts += ["", (
            "Every factual assertion must either quote one of the sources above verbatim "
            "and cite its id, or be marked as your own inference. Do not present "
            "inference as sourced -- separating the two is the point of this step."
        )]
        return "\n".join(parts)

    def build(self, ticker: str, seg: SubSegment, as_of: date,
              claims: list[Claim] | None = None) -> Briefing:
        """Assemble a briefing.

        With `claims=None` the result is sources-only -- valid, and what the funnel uses
        before Stage 5 exists.
        """
        sources = self.gather(ticker, as_of)
        briefing = Briefing(ticker=ticker, sub_segment=seg, as_of=as_of,
                            claims=claims or [], sources=sources,
                            tool_calls=len(self.retrievers))

        if claims and briefing.sourced_fraction < self.config.require_sourced_fraction:
            log.warning(
                "%s: only %.0f%% of claims are sourced (threshold %.0f%%) -- a briefing "
                "that is mostly inference is speculation with citations attached",
                ticker, briefing.sourced_fraction * 100,
                self.config.require_sourced_fraction * 100)

        return briefing
