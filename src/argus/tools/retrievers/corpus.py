"""The corpus retriever, behind the `Retriever` Protocol.

Lets `BriefingBuilder` draw on the local EDGAR index through the same interface a news or
web retriever would use, so the funnel does not need to know that this evidence came from
disk rather than the network.

One property distinguishes it from every other retriever that will sit behind this
Protocol: it is *reproducible*. A news API returns something different tomorrow; this
returns the same passages for the same (ticker, as_of) forever, because the corpus is
frozen at the 2025-12-31 cutoff. That is what makes it usable as the constant in a
before/after CPT comparison -- the arms differ in extraction, not in what they read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from argus.contracts.briefing import SourceType
from argus.contracts.quant import SubSegment
from argus.engine.extract.loop import segment_queries
from argus.engine.retrieval.search import SearchConfig, Searcher

log = logging.getLogger(__name__)


@dataclass
class CorpusRetriever:
    searcher: Searcher
    segment_of: dict[str, SubSegment] = field(default_factory=dict)
    default_segment: SubSegment = SubSegment.FABLESS
    lookback_days: int = 400
    name: str = "corpus"

    def source_type(self) -> SourceType:
        return SourceType.SEC_FILING

    def license_flags(self) -> tuple[bool, bool]:
        # EDGAR filings are US government works: public domain. The only source in this
        # project where both flags are unambiguously true.
        return (True, True)

    def fetch(self, ticker: str, as_of: date, limit: int) -> list[tuple[str, dict]]:
        seg = self.segment_of.get(ticker, self.default_segment)
        since = as_of - timedelta(days=self.lookback_days)
        cfg = SearchConfig(top_k=3, max_per_doc=2)

        seen: dict[int, object] = {}
        for q in segment_queries(seg):
            for h in self.searcher.search(q, tickers=[ticker], as_of=as_of,
                                          since=since, config=cfg):
                seen.setdefault(h.ref.chunk_id, h)
            if len(seen) >= limit * 3:
                break

        hits = sorted(seen.values(), key=lambda h: -h.score)[:limit]
        out = []
        for h in hits:
            _, _, text = h.passage()
            r = h.ref
            out.append((text, {
                "url": None,
                "title": f"{r.ticker} {r.source_type} {r.doc_date} {r.section or ''}".strip(),
                "published": r.doc_date,
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
            }))
        return out
