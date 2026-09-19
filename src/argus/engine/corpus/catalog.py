"""Document catalog -- the resumability backbone of the corpus pipeline.

A flat directory of ~18,000 documents is unmanageable and, more importantly,
un-resumable: a 2-4 hour rate-limited crawl that dies at 90% must not restart from zero.

The catalog also turns two things from bookkeeping into queries:

  * the corpus manifest (token counts, dedup rate, source mix)
  * the HARD DATE CUTOFF assertion -- with several sources having different date
    semantics, "no document postdates the cutoff" must be checkable mechanically rather
    than trusted. A single post-cutoff document invalidates the Phase 3 eval, because the
    model would have read the future it is being tested on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path

log = logging.getLogger(__name__)


class DocStatus(StrEnum):
    DISCOVERED = "discovered"   # known to exist, not yet fetched
    FETCHED = "fetched"         # raw content stored
    CLEANED = "cleaned"         # narrative text extracted
    DEDUPED = "deduped"         # survived near-duplicate removal
    PACKED = "packed"           # tokenised into a shard
    DROPPED = "dropped"         # excluded; drop_reason says why


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source          TEXT NOT NULL,        -- edgar | fedreg | govinfo | courtlistener | wikipedia | replay
    source_type     TEXT,                 -- 10-K, 10-Q, 8-K, rule, opinion, ...
    ticker          TEXT,
    cik             TEXT,
    url             TEXT,
    doc_date        DATE,                 -- publication/filing date: the cutoff key
    title           TEXT,
    raw_path        TEXT,
    raw_sha256      TEXT,
    raw_bytes       INTEGER,
    clean_path      TEXT,
    char_len        INTEGER,
    token_len       INTEGER,
    section         TEXT,
    status          TEXT NOT NULL,
    drop_reason     TEXT,
    dedup_cluster   TEXT,
    shard_id        INTEGER,
    license_ok      INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_status  ON documents (status);
CREATE INDEX IF NOT EXISTS idx_doc_source  ON documents (source);
CREATE INDEX IF NOT EXISTS idx_doc_date    ON documents (doc_date);
CREATE INDEX IF NOT EXISTS idx_doc_cik     ON documents (cik);
CREATE INDEX IF NOT EXISTS idx_doc_cluster ON documents (dedup_cluster);
"""


@dataclass
class Document:
    doc_id: str
    source: str
    source_type: str | None = None
    ticker: str | None = None
    cik: str | None = None
    url: str | None = None
    doc_date: date | None = None
    title: str | None = None
    raw_path: str | None = None
    raw_sha256: str | None = None
    raw_bytes: int | None = None
    clean_path: str | None = None
    char_len: int | None = None
    token_len: int | None = None
    section: str | None = None
    status: str = DocStatus.DISCOVERED
    drop_reason: str | None = None
    dedup_cluster: str | None = None
    shard_id: int | None = None
    license_ok: bool = True

    @staticmethod
    def make_id(source: str, url: str, section: str | None = None) -> str:
        key = f"{source}|{url}|{section or ''}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]


class Catalog:
    def __init__(self, path: str | Path = "data/corpus/catalog.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(self, doc: Document) -> None:
        now = datetime.now(timezone.utc)
        with self._conn() as c:
            c.execute("""
                INSERT INTO documents (doc_id, source, source_type, ticker, cik, url,
                    doc_date, title, raw_path, raw_sha256, raw_bytes, clean_path,
                    char_len, token_len, section, status, drop_reason, dedup_cluster,
                    shard_id, license_ok, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status=excluded.status, raw_path=excluded.raw_path,
                    raw_sha256=excluded.raw_sha256, raw_bytes=excluded.raw_bytes,
                    clean_path=excluded.clean_path, char_len=excluded.char_len,
                    token_len=excluded.token_len, drop_reason=excluded.drop_reason,
                    dedup_cluster=excluded.dedup_cluster, shard_id=excluded.shard_id,
                    updated_at=excluded.updated_at
            """, (doc.doc_id, doc.source, doc.source_type, doc.ticker, doc.cik, doc.url,
                  doc.doc_date, doc.title, doc.raw_path, doc.raw_sha256, doc.raw_bytes,
                  doc.clean_path, doc.char_len, doc.token_len, doc.section, str(doc.status),
                  doc.drop_reason, doc.dedup_cluster, doc.shard_id,
                  1 if doc.license_ok else 0, now, now))

    def upsert_many(self, docs: list[Document]) -> None:
        for d in docs:
            self.upsert(d)

    def pending(self, status: str, limit: int | None = None) -> list[Document]:
        """Documents awaiting a given stage -- this is what makes every stage resumable."""
        q = "SELECT * FROM documents WHERE status = ?"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [self._row_to_doc(r) for r in c.execute(q, (status,)).fetchall()]

    def by_cik(self, status: str | None = None) -> dict[str, list[Document]]:
        """Group documents by issuer.

        Used by dedupe: near-duplication in filings is overwhelmingly WITHIN an issuer
        across time (recycled risk factors, quarterly boilerplate), so partitioning by CIK
        turns one large MinHash problem into ~100 small independent ones.
        """
        q = "SELECT * FROM documents"
        params: tuple = ()
        if status:
            q += " WHERE status = ?"
            params = (status,)
        out: dict[str, list[Document]] = {}
        with self._conn() as c:
            for r in c.execute(q, params).fetchall():
                out.setdefault(r["cik"] or "_none", []).append(self._row_to_doc(r))
        return out

    @staticmethod
    def _row_to_doc(r: sqlite3.Row) -> Document:
        d = dict(r)
        d.pop("created_at", None)
        d.pop("updated_at", None)
        d["license_ok"] = bool(d.get("license_ok", 1))
        if isinstance(d.get("doc_date"), str):
            try:
                d["doc_date"] = date.fromisoformat(d["doc_date"])
            except ValueError:
                d["doc_date"] = None
        return Document(**d)

    # ------------------------------------------------------------- assertions

    def assert_cutoff(self, cutoff: date) -> None:
        """Fail if any retained document postdates the corpus cutoff.

        The Phase 3 eval only means anything if the model has not read the period it is
        tested on. This is checked as a query rather than trusted, because the corpus
        spans sources with different date semantics (filing date, publication date,
        decision date) and one mis-parsed field silently invalidates the whole gate.
        """
        with self._conn() as c:
            rows = c.execute(
                """SELECT doc_id, source, doc_date FROM documents
                   WHERE doc_date > ? AND status != ?""",
                (cutoff, str(DocStatus.DROPPED))).fetchall()
        if rows:
            sample = ", ".join(f"{r['source']}:{r['doc_id'][:8]}@{r['doc_date']}"
                               for r in rows[:5])
            raise AssertionError(
                f"{len(rows)} retained documents postdate the corpus cutoff {cutoff}. "
                f"The Phase 3 eval would be contaminated. Examples: {sample}")

    def assert_licensed(self) -> None:
        """Fail if any retained document is flagged as not permitted for training."""
        with self._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM documents WHERE license_ok = 0 AND status != ?",
                (str(DocStatus.DROPPED),)).fetchone()[0]
        if n:
            raise AssertionError(
                f"{n} retained documents are flagged license_ok=0. News archives and "
                "third-party transcripts commonly prohibit training use.")

    # --------------------------------------------------------------- manifest

    def manifest(self, cutoff: date | None = None) -> dict:
        """Corpus summary -- a query, not hand-maintained bookkeeping."""
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            by_status = {r[0]: r[1] for r in c.execute(
                "SELECT status, COUNT(*) FROM documents GROUP BY status")}
            by_source = {r[0]: {"docs": r[1], "tokens": r[2] or 0} for r in c.execute(
                """SELECT source, COUNT(*), SUM(token_len) FROM documents
                   WHERE status IN (?,?) GROUP BY source""",
                (str(DocStatus.DEDUPED), str(DocStatus.PACKED)))}
            tokens = c.execute(
                "SELECT SUM(token_len) FROM documents WHERE status IN (?,?)",
                (str(DocStatus.DEDUPED), str(DocStatus.PACKED))).fetchone()[0] or 0
            dropped_dup = c.execute(
                "SELECT COUNT(*) FROM documents WHERE drop_reason = 'near_duplicate'"
            ).fetchone()[0]
            date_range = c.execute(
                "SELECT MIN(doc_date), MAX(doc_date) FROM documents WHERE status != ?",
                (str(DocStatus.DROPPED),)).fetchone()

        m = {
            "total_documents": total,
            "by_status": by_status,
            "by_source": by_source,
            "total_tokens": int(tokens),
            "near_duplicates_removed": dropped_dup,
            "dedup_rate": dropped_dup / max(total, 1),
            "date_range": {"min": date_range[0], "max": date_range[1]},
            "cutoff": str(cutoff) if cutoff else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        m["sha256"] = hashlib.sha256(
            json.dumps({k: v for k, v in m.items() if k != "generated_at"},
                       sort_keys=True, default=str).encode()).hexdigest()
        return m
