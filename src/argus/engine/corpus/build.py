"""Corpus build orchestration: discover -> fetch -> clean -> dedupe -> pack.

Every stage is resumable via the catalog, so an interrupted run continues rather than
restarts. That matters: acquisition alone is 2-4 hours against a rate-limited API.

Stage order is deliberate. Dedup runs AFTER cleaning because near-duplicate detection
should compare narrative text, not HTML boilerplate -- two filings with identical wrappers
and different content would otherwise look like duplicates.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from argus.engine.corpus.catalog import Catalog, DocStatus, Document
from argus.engine.corpus.clean import extract_narrative
from argus.engine.corpus.dedupe import dedupe_partition, exact_hash_pass
from argus.engine.corpus.sources.edgar import EdgarClient, discover, fetch_pending

log = logging.getLogger(__name__)

SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class CorpusConfig:
    universe: str = "configs/universe/semis_v1.yaml"
    start: date = date(2010, 1, 1)

    # Matches the base model's own knowledge cutoff. Setting it earlier would not buy a
    # cleaner eval window -- the base model has already read everything before its own
    # cutoff -- so an earlier date costs training tokens for integrity we do not gain.
    cutoff: date = date(2025, 12, 31)

    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")
    include_exhibits: bool = True

    root: Path = Path("data/corpus")
    dedup_threshold: float = 0.8
    seq_len: int = 4096
    shard_tokens: int = 8_000_000
    tokenizer_model: str = "mlx-community/Ministral-3-14B-Base-2512-4bit"

    clean_dir: Path = field(init=False)
    shard_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.clean_dir = self.root / "clean"
        self.shard_dir = self.root / "shards"


def resolve_ciks(tickers: list[str], user_agent: str) -> dict[str, str]:
    """Map tickers to CIKs using the SEC's official mapping file.

    Fetched rather than hardcoded so the mapping stays correct as tickers change hands.
    """
    req = urllib.request.Request(SEC_TICKER_MAP, headers={"User-Agent": user_agent})
    raw = json.loads(urllib.request.urlopen(req, timeout=30).read())

    by_ticker = {v["ticker"].upper(): str(v["cik_str"]) for v in raw.values()}
    out, missing = {}, []
    for t in tickers:
        cik = by_ticker.get(t.upper())
        if cik:
            out[t] = cik
        else:
            missing.append(t)

    if missing:
        # Expected for foreign ADRs that file 20-F rather than 10-K, and for names whose
        # SEC registrant differs from the trading symbol.
        log.info("no SEC CIK for %d tickers (likely foreign filers): %s",
                 len(missing), ", ".join(sorted(missing)))
    return out


def stage_discover(cfg: CorpusConfig, catalog: Catalog, client: EdgarClient,
                   tickers: dict[str, str]) -> int:
    log.info("STAGE 1/5 discover: %d issuers, %s .. %s",
             len(tickers), cfg.start, cfg.cutoff)
    return discover(client, catalog, tickers, cfg.start, cfg.cutoff,
                    forms=cfg.forms, include_exhibits=cfg.include_exhibits)


def stage_fetch(cfg: CorpusConfig, catalog: Catalog, client: EdgarClient,
                limit: int | None = None) -> int:
    log.info("STAGE 2/5 fetch (rate-limited; resumable)")
    return fetch_pending(client, catalog, raw_dir=cfg.root / "raw" / "edgar", limit=limit)


def stage_clean(cfg: CorpusConfig, catalog: Catalog) -> int:
    """Extract narrative sections. One input document may yield several section rows."""
    log.info("STAGE 3/5 clean")
    cfg.clean_dir.mkdir(parents=True, exist_ok=True)
    todo = catalog.pending(DocStatus.FETCHED)
    written = 0

    for doc in todo:
        if not doc.raw_path or not Path(doc.raw_path).exists():
            doc.status, doc.drop_reason = DocStatus.DROPPED, "raw_missing"
            catalog.upsert(doc)
            continue

        raw = Path(doc.raw_path).read_text(errors="ignore")
        sections = extract_narrative(raw, form_type=doc.source_type)

        if not sections:
            doc.status, doc.drop_reason = DocStatus.DROPPED, "no_narrative_text"
            catalog.upsert(doc)
            continue

        for name, text in sections.items():
            # Sections become their own catalog rows so dedup and packing operate at the
            # granularity that actually repeats -- risk factors recur, whole filings do not.
            sec_id = Document.make_id("edgar", doc.url or doc.doc_id, name)
            path = cfg.clean_dir / f"{sec_id}.txt"
            path.write_text(text)
            catalog.upsert(Document(
                doc_id=sec_id, source=doc.source, source_type=doc.source_type,
                ticker=doc.ticker, cik=doc.cik, url=doc.url, doc_date=doc.doc_date,
                clean_path=str(path), char_len=len(text), section=name,
                status=DocStatus.CLEANED, license_ok=doc.license_ok,
            ))
            written += 1

        doc.status = DocStatus.DROPPED
        doc.drop_reason = "expanded_into_sections"
        catalog.upsert(doc)

    log.info("  cleaned into %d section documents", written)
    return written


def stage_dedupe(cfg: CorpusConfig, catalog: Catalog) -> dict:
    """Near-duplicate removal, partitioned by issuer.

    Partitioning is the whole trick: near-duplication is overwhelmingly within-issuer
    across time, so ~100 small independent problems replace one large one.
    """
    log.info("STAGE 4/5 dedupe (partitioned by CIK)")
    by_cik = {k: [d for d in v if d.status == DocStatus.CLEANED]
              for k, v in catalog.by_cik(status=DocStatus.CLEANED).items()}

    total_in = total_dropped = 0
    for cik, docs in by_cik.items():
        if not docs:
            continue
        texts = {d.doc_id: Path(d.clean_path).read_text()
                 for d in docs if d.clean_path and Path(d.clean_path).exists()}
        if len(texts) < 2:
            for d in docs:
                d.status = DocStatus.DEDUPED
                catalog.upsert(d)
            total_in += len(texts)
            continue

        res = dedupe_partition(texts, threshold=cfg.dedup_threshold)
        total_in += res.n_input
        total_dropped += len(res.drop)

        for d in docs:
            if d.doc_id in res.drop:
                d.status = DocStatus.DROPPED
                d.drop_reason = "near_duplicate"
                d.dedup_cluster = res.drop[d.doc_id]
            else:
                d.status = DocStatus.DEDUPED
                d.dedup_cluster = res.clusters.get(d.doc_id)
            catalog.upsert(d)

        log.info("  cik %s: %d docs, %d near-duplicates removed (%.0f%%)",
                 cik, res.n_input, len(res.drop), res.drop_rate * 100)

    # Cheap global pass for cross-issuer boilerplate (shared legal language).
    survivors = [d for v in catalog.by_cik(status=DocStatus.DEDUPED).values() for d in v]
    texts = {d.doc_id: Path(d.clean_path).read_text()
             for d in survivors if d.clean_path and Path(d.clean_path).exists()}
    cross = exact_hash_pass(texts)
    for d in survivors:
        if d.doc_id in cross:
            d.status, d.drop_reason = DocStatus.DROPPED, "near_duplicate"
            catalog.upsert(d)

    stats = {"input": total_in, "within_issuer_dropped": total_dropped,
             "cross_issuer_dropped": len(cross),
             "rate": (total_dropped + len(cross)) / max(total_in, 1)}
    log.info("  dedupe: %d/%d removed (%.1f%%)",
             total_dropped + len(cross), total_in, stats["rate"] * 100)
    return stats


def stage_pack(cfg: CorpusConfig, catalog: Catalog) -> dict:
    """Tokenise and write memmap shards."""
    log.info("STAGE 5/5 pack (tokenising with %s)", cfg.tokenizer_model)
    from transformers import AutoTokenizer

    from argus.engine.corpus.pack import ChunkSpec, ShardWriter, chunk_text

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_model)
    writer = ShardWriter(cfg.shard_dir, shard_tokens=cfg.shard_tokens, seq_len=cfg.seq_len)
    spec = ChunkSpec(window=cfg.seq_len)

    docs = [d for v in catalog.by_cik(status=DocStatus.DEDUPED).values() for d in v]
    # Chronological order so the corpus is not organised by issuer -- a model fed all of
    # one company then all of the next would see a highly non-stationary stream.
    docs.sort(key=lambda d: (d.doc_date or date.min, d.doc_id))

    for i, d in enumerate(docs):
        if not d.clean_path or not Path(d.clean_path).exists():
            continue
        text = Path(d.clean_path).read_text()
        chunks = chunk_text(text, tok, spec)
        tokens = sum(len(c) for c in chunks)
        for c in chunks:
            writer.add(c)
        d.token_len = tokens
        d.status = DocStatus.PACKED
        catalog.upsert(d)
        if (i + 1) % 500 == 0:
            log.info("  packed %d/%d documents", i + 1, len(docs))

    return writer.close()


def build(cfg: CorpusConfig, user_agent: str, skip_fetch: bool = False,
          skip_discovery: bool = False, limit: int | None = None) -> dict:
    """Run the full pipeline. Safe to re-run: each stage picks up where it left off."""
    import yaml

    catalog = Catalog(cfg.root / "catalog.sqlite")
    client = EdgarClient(user_agent=user_agent)

    # Discovery is idempotent but re-walking ~22k filings costs ~1h. Once the catalog
    # holds them, skip straight to fetching.
    if not skip_discovery:
        universe = yaml.safe_load(Path(cfg.universe).read_text())
        symbols = [e["symbol"] for entries in universe["tickers"].values() for e in entries]
        tickers = resolve_ciks(symbols, user_agent)
        log.info("resolved %d/%d tickers to CIKs", len(tickers), len(symbols))
        stage_discover(cfg, catalog, client, tickers)
    else:
        log.info("STAGE 1/5 discover: SKIPPED (using existing catalog)")
    if not skip_fetch:
        stage_fetch(cfg, catalog, client, limit=limit)
    stage_clean(cfg, catalog)
    dedup_stats = stage_dedupe(cfg, catalog)

    # Assert BEFORE packing: a single post-cutoff document invalidates the Phase 3 eval,
    # and it is far cheaper to catch that here than after tokenising the whole corpus.
    catalog.assert_cutoff(cfg.cutoff)
    catalog.assert_licensed()

    shard_index = stage_pack(cfg, catalog)

    manifest = catalog.manifest(cutoff=cfg.cutoff)
    manifest["dedup"] = dedup_stats
    manifest["shards"] = {k: v for k, v in shard_index.items() if k != "shards"}
    (cfg.root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
