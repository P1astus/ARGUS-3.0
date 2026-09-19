"""Build the chunk index from the corpus catalog.

Two passes, deliberately separated:

  1. chunk + lexical -- fast (single-digit minutes), no model, no GPU. Produces a working
     BM25 index on its own, so retrieval is usable and measurable before deciding whether
     dense embeddings earn their place.
  2. embed -- ~35 minutes for the full corpus. Resumable: it fills the vector memmap in
     chunk_id order and records how far it got, so an interrupted run continues rather
     than restarting.

Splitting them is what makes "does dense actually help here?" an answerable question
instead of an assumption baked into a single build step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from argus.engine.corpus.catalog import Catalog, DocStatus
from argus.engine.retrieval.chunk import ChunkSpec, chunk_document
from argus.engine.retrieval.embed import EmbedConfig, Embedder
from argus.engine.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    corpus_root: Path = Path("data/corpus")
    index_root: Path = Path("data/index")
    spec: ChunkSpec = ChunkSpec()
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    min_doc_chars: int = 400     # below this an EDGAR doc is a cover page, not content


def build_lexical(cfg: IndexConfig | None = None, limit: int | None = None,
                  reset: bool = True) -> dict:
    """Pass 1: chunk every retained document and build the BM25 index.

    `reset=False` is the incremental path: it chunks only documents not already in the
    index (`store.indexed_docs()`), so growing the corpus with a second universe -- e.g.
    adding tech_v1.yaml alongside semis_v1.yaml -- appends rather than re-chunking and
    duplicating everything already there.
    """
    cfg = cfg or IndexConfig()
    cat = Catalog(cfg.corpus_root / "catalog.sqlite")
    store = ChunkStore(cfg.index_root)
    if reset:
        store.reset()
        already: set[str] = set()
    else:
        already = store.indexed_docs()

    docs = cat.pending(str(DocStatus.PACKED))
    docs.sort(key=lambda d: (d.ticker or "", d.doc_date or "", d.doc_id))
    if already:
        n_before = len(docs)
        docs = [d for d in docs if d.doc_id not in already]
        log.info("incremental: %d/%d documents already indexed, %d new",
                 n_before - len(docs), n_before, len(docs))
    if limit:
        docs = docs[:limit]

    t0 = time.time()
    next_id = store.max_id() + 1
    n_docs = n_chunks = n_skipped = 0
    rows: list[tuple] = []
    texts: list[str] = []
    batch_first = next_id

    def flush() -> None:
        nonlocal rows, texts, batch_first
        if rows:
            store.add_batch(rows, texts, batch_first)
            rows, texts = [], []
            batch_first = next_id

    for doc in docs:
        if not doc.clean_path:
            n_skipped += 1
            continue
        p = Path(doc.clean_path)
        if not p.exists():
            n_skipped += 1
            continue
        text = p.read_text()
        if len(text) < cfg.min_doc_chars:
            n_skipped += 1
            continue

        for ch in chunk_document(doc.doc_id, text, cfg.spec):
            rows.append((next_id, doc.doc_id, ch.ordinal, ch.start, ch.end,
                         len(ch.text), doc.ticker, doc.cik,
                         doc.doc_date.isoformat() if doc.doc_date else None,
                         doc.source_type, doc.section, str(p)))
            texts.append(ch.text)
            next_id += 1
            n_chunks += 1
        n_docs += 1

        if len(rows) >= 5000:
            flush()
        if n_docs % 2000 == 0:
            log.info("chunked %d/%d docs -> %d chunks (%.0fs)",
                     n_docs, len(docs), n_chunks, time.time() - t0)
    flush()

    stats = {"documents": n_docs, "skipped": n_skipped, "chunks": n_chunks,
             "seconds": round(time.time() - t0, 1),
             "spec": {"target": cfg.spec.target, "overlap": cfg.spec.overlap}}
    if reset:
        # A full rebuild invalidates any prior embedding progress. An incremental append
        # must NOT touch embedded_through -- the chunks it already recorded are still
        # valid rows in the (about to be grown) vector memmap, and resetting this would
        # make build_dense recompute embeddings it already has.
        store.set_meta(lexical=stats, embedded_through=0)
    else:
        store.set_meta(lexical=stats)
    log.info("lexical index: %d new chunks from %d new docs in %.0fs (%d skipped)",
             n_chunks, n_docs, stats["seconds"], n_skipped)
    return stats


def build_dense(cfg: IndexConfig | None = None, resume: bool = True,
                progress_every: int = 20000, budget_seconds: float | None = None) -> dict:
    """Pass 2: fill the vector memmap. Resumable at `embedded_through`.

    Runs at ~148 chunks/s, so ~35 min for the full corpus. Getting there required
    bounding MLX's buffer cache -- see the LENGTH_BUCKET note in `embed.py`; without it
    throughput collapsed to under 1 chunk/s after about 60 seconds and the build would
    have taken days. `budget_seconds` stops cleanly at a checkpoint so the build can be
    run in slices under a wall-clock cap.
    """
    cfg = cfg or IndexConfig()
    store = ChunkStore(cfg.index_root)
    total = store.count()
    if total == 0:
        raise RuntimeError("no chunks indexed; run build_lexical first")

    dim = cfg.embed.dim
    done = int(store.get_meta("embedded_through", 0)) if resume else 0
    existing = store.vectors

    if existing is None:
        vecs = store.open_vectors(dim, total, mode="w+")
        done = 0
    elif existing.shape[0] < total:
        # The chunk table grew past the vector file -- expected after an incremental
        # `build_lexical(reset=False)` appended a new universe's chunks. Grow in place so
        # the `done` rows already embedded (semis, previously) are preserved and only the
        # new rows need encoding, rather than wiping and recomputing everything.
        log.info("vectors cover %d/%d chunks; growing in place, not rebuilding",
                 existing.shape[0], total)
        vecs = store.grow_vectors(dim, total)
    elif existing.shape[0] > total or existing.shape[1] != dim:
        log.warning("vector shape %s incompatible with (%d, %d); rebuilding",
                   existing.shape, total, dim)
        vecs = store.open_vectors(dim, total, mode="w+")
        done = 0
    else:
        vecs = store.open_vectors(dim, total, mode="r+")

    emb = Embedder(cfg.embed)
    start_at = done
    t0 = time.time()
    batch = 512   # chunk_ids per DB round-trip; the encoder sub-batches internally

    while done < total:
        ids = list(range(done + 1, min(done + batch, total) + 1))
        refs = store.get(ids)
        if not refs:
            break
        mat = emb.encode([r.text() for r in refs], is_query=False)
        vecs[refs[0].chunk_id - 1: refs[-1].chunk_id] = mat.astype(np.float16)
        done = refs[-1].chunk_id
        if done % progress_every < batch:
            elapsed = max(time.time() - t0, 1e-9)
            rate = (done - start_at) / elapsed
            log.info("embedded %d/%d (%.0f chunks/s, eta %.0f min)",
                     done, total, rate, (total - done) / max(rate, 1e-9) / 60)
            vecs.flush()
            store.set_meta(embedded_through=done)

        if budget_seconds and (time.time() - t0) > budget_seconds:
            log.info("budget reached at %d/%d; rerun to resume", done, total)
            break

    vecs.flush()
    complete = done >= total
    store.set_meta(embedded_through=done,
                   dense={"model": cfg.embed.model, "dim": dim, "chunks": done,
                          "total": total, "complete": complete,
                          "seconds": round(time.time() - t0, 1)})
    log.info("dense index: %d/%d vectors in %.0fs (complete=%s)",
             done, total, time.time() - t0, complete)
    return {"chunks": done, "total": total, "complete": complete,
            "seconds": round(time.time() - t0, 1)}
