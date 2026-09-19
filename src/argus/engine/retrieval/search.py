"""Retrieval: hard filter, then BM25 and dense, then fusion.

ORDER MATTERS
The metadata filter runs first and in SQL. `as_of` is a leakage boundary -- a briefing
dated 2024-06-01 must be unable to see a filing published 2024-08-01, and "unable" has to
mean not-a-candidate, not merely ranked low. Every measured retrieval number in this
project is only meaningful because the filter is upstream of scoring.

WHY BOTH SCORERS
BM25 is exact-term: unbeatable for "book-to-bill", "days of inventory", a specific dollar
figure. Dense is paraphrase-tolerant: it finds a passage describing distributor
overstocking when the query says "channel inventory". They fail differently, which is the
only good reason to run two. Reciprocal-rank fusion combines them without needing their
scores to be commensurable -- BM25 returns negative log-odds-ish values and cosine
returns [-1,1], and any weighted sum of the two is a hyperparameter waiting to be
overfit.

Whether dense actually earns its keep on this corpus is measured, not assumed --
`argus.engine.retrieval.evaluate` scores each mode separately.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from argus.engine.retrieval.chunk import expand
from argus.engine.retrieval.embed import Embedder
from argus.engine.retrieval.store import ChunkRef, ChunkStore, read_clean

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\./]*")

# Deliberately short. A long stoplist would strip terms that carry meaning in filings
# ("we", "may", "not" all matter in risk language); these are the ones that only ever
# add noise to an OR-query.
_STOP = frozenset("""a an the of and or in on for to with by from as at is are was were
be been being that this these those it its their our your his her they we you i""".split())


@dataclass
class SearchConfig:
    top_k: int = 12
    candidate_k: int = 60          # per-scorer depth before fusion
    rrf_k: int = 60                # RRF damping; 60 is the value from the original paper
    max_per_doc: int = 3           # stop one 10-K monopolising a briefing
    expand_before: int = 600       # chars of left context handed to the reading model
    expand_after: int = 900
    mode: str = "hybrid"           # hybrid | bm25 | dense


@dataclass
class Hit:
    ref: ChunkRef
    score: float
    rank_bm25: int | None = None
    rank_dense: int | None = None
    components: dict = field(default_factory=dict)

    def passage(self, cfg: SearchConfig | None = None) -> tuple[int, int, str]:
        """(start, end, text) of the widened span, still a literal substring."""
        cfg = cfg or SearchConfig()
        doc = read_clean(self.ref.clean_path)
        lo, hi = expand(doc, self.ref.start, self.ref.end,
                        cfg.expand_before, cfg.expand_after)
        return lo, hi, doc[lo:hi]


def fts_query(text: str) -> str:
    """Turn free text into an FTS5 OR-query of quoted terms.

    Quoting each term is not cosmetic: FTS5 treats bare `-` and `.` as syntax, and semi
    filings are full of `book-to-bill`, `10-K`, `R&D`. An unquoted query on this corpus
    raises `fts5: syntax error` on entirely ordinary inputs.
    """
    terms = [t.lower() for t in _WORD.findall(text)]
    terms = [t for t in terms if t not in _STOP and len(t) > 1]
    if not terms:
        return ""
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append('"' + t.replace('"', '') + '"')
    return " OR ".join(out)


class Searcher:
    def __init__(self, store: ChunkStore, embedder: Embedder | None = None,
                 config: SearchConfig | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or SearchConfig()

    # ------------------------------------------------------------- components

    def _bm25(self, query: str, candidates: list[int], k: int) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match or not candidates:
            return []
        # The candidate set goes into a temp table rather than an `IN (...)` list: a
        # ticker with fifteen years of filings routinely exceeds ten thousand chunks,
        # which is past what is sane to bind as parameters.
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("CREATE TEMP TABLE cand (id INTEGER PRIMARY KEY)")
            conn.executemany("INSERT OR IGNORE INTO cand (id) VALUES (?)",
                             [(i,) for i in candidates])
            try:
                rows = conn.execute(
                    """SELECT f.rowid, bm25(chunks_fts) AS s
                       FROM chunks_fts f JOIN cand ON cand.id = f.rowid
                       WHERE chunks_fts MATCH ?
                       ORDER BY s LIMIT ?""", (match, k)).fetchall()
            except sqlite3.OperationalError as e:
                log.warning("FTS query failed (%s) for %r", e, match[:120])
                return []
        finally:
            conn.close()
        # bm25() returns *more negative = better*; flip so higher is better everywhere.
        return [(int(r[0]), -float(r[1])) for r in rows]

    def _dense(self, query: str, candidates: list[int], k: int) -> list[tuple[int, float]]:
        vecs = self.store.vectors
        if vecs is None or self.embedder is None or not candidates:
            return []
        rows = np.asarray([c - 1 for c in candidates], dtype=np.int64)
        rows = rows[(rows >= 0) & (rows < vecs.shape[0])]
        if rows.size == 0:
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        sub = np.asarray(vecs[rows], dtype=np.float32)
        sims = sub @ q
        top = np.argpartition(-sims, min(k, sims.size - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(rows[i]) + 1, float(sims[i])) for i in top]

    # -------------------------------------------------------------- interface

    def search(self, query: str, tickers: list[str] | None = None,
               as_of: date | None = None, since: date | None = None,
               source_types: list[str] | None = None,
               sections: list[str] | None = None,
               config: SearchConfig | None = None) -> list[Hit]:
        cfg = config or self.config
        candidates = self.store.candidate_ids(
            tickers=tickers, as_of=as_of, since=since,
            source_types=source_types, sections=sections)
        if not candidates:
            return []

        want_bm25 = cfg.mode in ("hybrid", "bm25")
        want_dense = cfg.mode in ("hybrid", "dense")
        bm = self._bm25(query, candidates, cfg.candidate_k) if want_bm25 else []
        dn = self._dense(query, candidates, cfg.candidate_k) if want_dense else []

        fused: dict[int, dict] = {}
        for rank, (cid, s) in enumerate(bm, start=1):
            fused.setdefault(cid, {})["bm25"] = (rank, s)
        for rank, (cid, s) in enumerate(dn, start=1):
            fused.setdefault(cid, {})["dense"] = (rank, s)

        scored = []
        for cid, parts in fused.items():
            rrf = sum(1.0 / (cfg.rrf_k + r) for r, _ in parts.values())
            scored.append((cid, rrf, parts))
        scored.sort(key=lambda x: -x[1])

        refs = {r.chunk_id: r for r in self.store.get([c for c, _, _ in scored])}
        out: list[Hit] = []
        per_doc: dict[str, int] = {}
        for cid, rrf, parts in scored:
            ref = refs.get(cid)
            if ref is None:
                continue
            if per_doc.get(ref.doc_id, 0) >= cfg.max_per_doc:
                continue
            per_doc[ref.doc_id] = per_doc.get(ref.doc_id, 0) + 1
            out.append(Hit(
                ref=ref, score=rrf,
                rank_bm25=parts.get("bm25", (None,))[0],
                rank_dense=parts.get("dense", (None,))[0],
                components={k: v[1] for k, v in parts.items()}))
            if len(out) >= cfg.top_k:
                break
        return out
