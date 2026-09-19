"""The local extraction loop: retrieve -> show -> extract -> Briefing.

WHAT THE LOOP IS FOR
This is the step the whole architecture is arranged around. A 10-K is 73-100k tokens; a
briefing is 2-5k. The compression has to happen locally or the frontier budget is spent
on reading rather than on judgment. And it has to happen *without* the local model being
trusted, because at 103M CPT tokens it will not have reliable factual recall (handoff §9)
-- so every factual line in a briefing is a span the model was shown, located back in the
retained document.

WHY QUERIES COME FROM THE TAXONOMY
The retrieval queries are generated from the sub-segment profile in `tools.taxonomy`,
not from the model. Letting the model choose what to search for would make the briefing's
coverage depend on the model's priors, which is precisely the thing CPT is supposed to
change -- and would make a before/after CPT comparison uninterpretable, because the two
arms would have read different documents. Fixed queries per segment mean the CPT delta
is measured on extraction, holding retrieval constant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from argus.contracts.briefing import Briefing, Claim, Source, SourceType
from argus.contracts.quant import SubSegment
from argus.engine.extract.claims import ExtractionStats, parse_claims
from argus.engine.extract.prompts import STOP_SEQUENCES, extraction_prompt
from argus.engine.retrieval.search import Hit, SearchConfig, Searcher
from argus.tools.provenance import ProvenanceStore
from argus.tools.taxonomy import PROFILES

log = logging.getLogger(__name__)

EDGAR_BASE = "https://www.sec.gov/Archives/edgar/data/"


@dataclass
class ExtractConfig:
    lookback_days: int = 400        # ~4 quarters, so a year-ago comparison is retrievable
    passages_per_query: int = 3
    max_passages: int = 10

    # ONE passage per generation. MEASURED: at five passages per call the model spent its
    # entire 1100-token budget on the first one -- ten claims off a single bulleted risk
    # factor -- and then degenerated into copying source text verbatim. Passages 2-5 were
    # never reached, so eight of ten retrieved passages contributed nothing to the
    # briefing while the audit reported a flawless 100% exact-quote rate. That is the
    # worst possible combination: a metric that looks perfect because most of the evidence
    # was silently skipped. One passage per call makes coverage structural.
    batch_passages: int = 1
    max_tokens: int = 400

    # Per-section quotas, because filing sections are not equally informative and the
    # boilerplate-heavy ones match almost any query lexically.
    #
    # MEASURED over 40 passages retrieved across NVDA/MU/AMAT/ADI with no cap: 40% came
    # from 10-K `business` (8) and risk factors (8), against 16 from MD&A and earnings
    # releases. Risk factors are also the most-recycled text in a filing -- the corpus
    # dedup pass found 18.4% within-issuer near-duplicates, concentrated there -- and
    # `quantitative_qualitative` is Item 7A, which is interest-rate and FX sensitivity
    # boilerplate that says nothing about the cycle.
    #
    # Uncapped (the quarter's actual news): `mdna`, and `full` when it is an 8-K/EX
    # earnings release.
    section_caps: dict[str, int] = field(default_factory=lambda: {
        "risk_factors": 2,
        "business": 1,
        "quantitative_qualitative": 1,
    })

    # Handoff §6: EDGAR 8-K *primary* documents are cover pages (~0.5-1.1k tokens); the
    # substance is in the EX-99 exhibits, which arrive as source_type `8-K/EX`. Capped
    # rather than excluded, because an occasional Item 2.02 body does carry content.
    max_cover_page_8k: int = 1


@dataclass
class ExtractionResult:
    briefing: Briefing
    stats: ExtractionStats
    hits: list[Hit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    seconds: float = 0.0
    raw_outputs: list[str] = field(default_factory=list)
    passage_map: dict[str, str] = field(default_factory=dict)  # source_id -> doc_id

    @property
    def audit(self) -> dict:
        return {"ticker": self.briefing.ticker,
                "sub_segment": str(self.briefing.sub_segment),
                "as_of": str(self.briefing.as_of),
                "n_sources": len(self.briefing.sources),
                "n_claims": len(self.briefing.claims),
                "sourced_fraction": self.briefing.sourced_fraction,
                "seconds": round(self.seconds, 1),
                **self.stats.as_dict()}


def segment_queries(seg: SubSegment) -> list[str]:
    """One query per thing that matters for this sub-segment.

    Metrics and watch-items are kept as separate queries rather than concatenated: a
    single long query retrieves passages that are vaguely about everything, and BM25 in
    particular degrades badly when a rare decisive term ("book-to-bill") is diluted by
    twenty common ones.
    """
    p = PROFILES[seg]
    qs = [f"{m} results and trend" for m in p.key_metrics]
    qs += list(p.watch_items)
    qs += ["guidance for the next quarter revenue and margin",
           "inventory levels and channel inventory commentary",
           "demand outlook and order patterns"]
    return qs


class ExtractionLoop:
    def __init__(self, searcher: Searcher, generator, store: ProvenanceStore,
                 config: ExtractConfig | None = None) -> None:
        self.searcher = searcher
        self.generator = generator
        self.store = store
        self.config = config or ExtractConfig()

    # ---------------------------------------------------------------- retrieve

    def retrieve(self, ticker: str, seg: SubSegment, as_of: date) -> list[Hit]:
        cfg = self.config
        since = as_of - timedelta(days=cfg.lookback_days)
        seen: dict[int, Hit] = {}
        search_cfg = SearchConfig(top_k=cfg.passages_per_query, max_per_doc=2)

        for q in segment_queries(seg):
            for h in self.searcher.search(q, tickers=[ticker], as_of=as_of,
                                          since=since, config=search_cfg):
                # First query to surface a chunk wins. Later duplicates are dropped
                # rather than re-ranked, so a passage cannot gain weight merely by being
                # generically relevant to several queries.
                seen.setdefault(h.ref.chunk_id, h)

        ranked = sorted(seen.values(), key=lambda h: -h.score)
        used: dict[str, int] = {}
        hits: list[Hit] = []
        for h in ranked:
            sec = h.ref.section or "unknown"
            # Plain 8-K cover pages get their own budget, separate from the 8-K/EX
            # exhibits they share a `section` value with.
            key = "_cover_8k" if h.ref.source_type == "8-K" else sec
            cap = (cfg.max_cover_page_8k if key == "_cover_8k"
                   else cfg.section_caps.get(sec))
            if cap is not None and used.get(key, 0) >= cap:
                continue
            used[key] = used.get(key, 0) + 1
            hits.append(h)
            if len(hits) >= cfg.max_passages:
                break

        hits.sort(key=lambda h: (h.ref.doc_date or date.min, h.ref.ordinal))
        return hits

    # ----------------------------------------------------------------- retain

    def retain(self, hits: list[Hit]) -> tuple[list[Source], dict[str, str], dict[str, str]]:
        """Store the exact passage text shown to the model.

        The provenance record is the *passage*, not the whole filing. That makes the
        audit answer the question that matters -- did the model quote what it was shown --
        rather than the weaker question of whether the quote appears somewhere in a
        300KB 10-K it only saw 2k characters of.
        """
        sources: list[Source] = []
        documents: dict[str, str] = {}
        doc_of: dict[str, str] = {}

        for h in hits:
            _, _, text = h.passage()
            ref = h.ref
            label = (f"{ref.ticker} {ref.source_type} {ref.doc_date} "
                     f"{ref.section or ''}".strip())
            src = self.store.put(
                content=text, source_type=SourceType.SEC_FILING,
                url=None, title=label, published=ref.doc_date,
                # EDGAR filings are US government works: public domain, and the only
                # source in this corpus for which training use is unambiguously allowed.
                redistributable=True, training_permitted=True)
            if src.source_id not in documents:
                sources.append(src)
                documents[src.source_id] = text
                doc_of[src.source_id] = ref.doc_id
        return sources, documents, doc_of

    # ---------------------------------------------------------------- extract

    def run(self, ticker: str, seg: SubSegment, as_of: date) -> ExtractionResult:
        t0 = time.time()
        hits = self.retrieve(ticker, seg, as_of)
        sources, documents, doc_of = self.retain(hits)
        claims, stats, raw = extract_claims_from_sources(
            ticker, seg, sources, documents, self.generator, self.config)

        briefing = Briefing(ticker=ticker, sub_segment=seg, as_of=as_of,
                            claims=claims, sources=sources,
                            tool_calls=len(segment_queries(seg)))

        return ExtractionResult(briefing=briefing, stats=stats, hits=hits,
                                queries=segment_queries(seg),
                                seconds=time.time() - t0, raw_outputs=raw,
                                passage_map=doc_of)


def extract_claims_from_sources(ticker: str, seg: SubSegment, sources: list[Source],
                                documents: dict[str, str], generator,
                                config: ExtractConfig
                                ) -> tuple[list[Claim], ExtractionStats, list[str]]:
    """The retrieve-agnostic half of the loop: given already-retained passages, batch
    them through the model and verify quotes. Shared between `ExtractionLoop.run()`
    (pre-built index) and `live.py`'s live-retrieval path (engine/extract/live.py) so the
    two never drift on batching/quote-verification behaviour -- only how `sources` and
    `documents` were obtained differs between them.
    """
    claims: list[Claim] = []
    stats = ExtractionStats()
    raw: list[str] = []

    # Passages are extracted in batches rather than all at once. A base model's copying
    # accuracy degrades with prompt length, and the failure is silent -- it keeps
    # producing well-formed lines whose quotes drift. Smaller batches also mean a
    # truncated generation costs a few claims rather than all of them.
    b = max(1, config.batch_passages)
    for i in range(0, len(sources), b):
        batch = sources[i:i + b]
        display = {f"S{j + 1}": s.source_id for j, s in enumerate(batch)}
        passages = [(f"S{j + 1}", s.title or "", documents[s.source_id])
                    for j, s in enumerate(batch)]

        prompt = extraction_prompt(ticker, seg, passages)
        out = generator.generate(prompt, max_tokens=config.max_tokens)
        raw.append(out)

        batch_docs = {s.source_id: documents[s.source_id] for s in batch}
        parsed, _ = parse_claims(out, batch_docs, display, stats)
        claims.extend(parsed)

    return claims, stats, raw
