"""Live-sourced briefing: same extraction + quote-verification pipeline as
`ExtractionLoop.run()`, sourced from `LiveEdgarRetriever` instead of a pre-built index.

WHY THIS EXISTS
The pre-built corpus has a hard cutoff (2025-12-31 as of this writing) -- `extract brief`
against it cannot see anything filed after that date no matter what `as_of` is passed. A
briefing genuinely "as of today" needs filings the index was never built with. `live_edgar.
py` (handoff §12g) already solves the fetch side of this -- CIK resolution, most-recent-
filings lookup, exhibit-following -- but was never connected to the claim-extraction /
quote-verification half of the pipeline that makes a briefing trustworthy. This is that
connection.

WHAT IS DIFFERENT FROM THE INDEXED PATH
No retrieval ranking: `LiveEdgarRetriever.fetch()` returns whole extracted filing sections
(the most recent `limit` filings), not BM25/dense-ranked chunks against a query. For "what
has this company said most recently", that is the right unit anyway -- ranking exists to
find a needle in ~46 tickers' full history, not to filter three or four recent filings.
`extract_claims_from_sources` (loop.py) is reused unchanged, so verification behaves
identically to the indexed path -- only how passages were found differs.

WHAT THIS IS NOT
Not a replacement for `extract brief` against the corpus for historical/backtest dates --
slower (live network round trips), narrower (most-recent filings only, no historical
depth), and outside the retrieval-quality measurements in docs/retrieval_and_extraction.md.
Use it specifically for "as of today" or any date past the corpus cutoff.
"""

from __future__ import annotations

import time
from datetime import date

from argus.contracts.briefing import SourceType
from argus.contracts.quant import SubSegment
from argus.engine.extract.loop import ExtractConfig, ExtractionResult, extract_claims_from_sources
from argus.tools.provenance import ProvenanceStore
from argus.tools.retrievers.live_edgar import LiveEdgarRetriever


def run_live(ticker: str, seg: SubSegment, as_of: date, retriever: LiveEdgarRetriever,
            generator, store: ProvenanceStore, config: ExtractConfig | None = None,
            limit: int = 8) -> ExtractionResult:
    """Fetch the `limit` most recent filings on or before `as_of` directly from SEC,
    then run the same extract-and-verify pipeline the indexed loop uses.

    `limit` bounds live network round trips (one fetch per filing/exhibit) -- deliberately
    small since this reaches back only far enough for "what has this company said
    recently", not for historical depth.
    """
    cfg = config or ExtractConfig()
    t0 = time.time()

    passages = retriever.fetch(ticker, as_of, limit)

    sources = []
    documents: dict[str, str] = {}
    for text, meta in passages:
        src = store.put(
            content=text, source_type=SourceType.SEC_FILING,
            url=meta.get("url"), title=meta.get("title"), published=meta.get("published"),
            # Same reasoning as loop.py's retain(): EDGAR filings are public domain.
            redistributable=True, training_permitted=True)
        if src.source_id not in documents:
            sources.append(src)
            documents[src.source_id] = text

    claims, stats, raw = extract_claims_from_sources(
        ticker, seg, sources, documents, generator, cfg)

    from argus.contracts.briefing import Briefing
    briefing = Briefing(ticker=ticker, sub_segment=seg, as_of=as_of,
                        claims=claims, sources=sources, tool_calls=1)

    return ExtractionResult(briefing=briefing, stats=stats, seconds=time.time() - t0,
                            raw_outputs=raw)
