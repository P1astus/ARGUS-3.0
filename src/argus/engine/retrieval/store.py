"""The chunk index: metadata + FTS5 + a dense vector memmap.

WHAT IS AND IS NOT STORED
Chunk *text* is deliberately not stored. It is `clean_doc[start:end]` and the clean
documents are already on disk under `data/corpus/clean/`; duplicating 466MB of text into
SQLite would buy nothing and would introduce a second copy that could drift from the one
quotes are verified against. The index stores offsets and reads through to the file.

The FTS5 table is therefore contentless (`content=''`): it keeps the inverted index for
BM25 and no copy of the text. That is also why the index is build-once/rebuild rather
than incrementally editable -- contentless FTS5 cannot be updated in place, and a corpus
rebuild is ~2.5h anyway, so there is no incremental case worth the complexity.

Vectors live in a float16 memmap keyed by `chunk_id - 1`, pre-normalised so search is a
dot product. 384 dims x float16 is 0.75KB/chunk; at ~300k chunks that is ~230MB, small
enough to load resident and brute-force, which removes an ANN index (and its recall
error) from the system entirely.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY,   -- 1-based; vector row is chunk_id - 1
    doc_id      TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    start       INTEGER NOT NULL,
    end         INTEGER NOT NULL,
    n_chars     INTEGER NOT NULL,
    ticker      TEXT,
    cik         TEXT,
    doc_date    DATE,
    source_type TEXT,
    section     TEXT,
    clean_path  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ch_doc    ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_ch_ticker ON chunks (ticker, doc_date);
CREATE INDEX IF NOT EXISTS idx_ch_date   ON chunks (doc_date);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    content='',
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class ChunkRef:
    """A chunk as retrieval hands it back: identity, provenance, and its span."""

    chunk_id: int
    doc_id: str
    ordinal: int
    start: int
    end: int
    ticker: str | None
    doc_date: date | None
    source_type: str | None
    section: str | None
    clean_path: str

    def text(self) -> str:
        """The exact substring the chunk denotes. Reads through to the clean document."""
        return read_clean(self.clean_path)[self.start:self.end]


@lru_cache(maxsize=64)
def read_clean(path: str) -> str:
    """Cached read of a clean document.

    64 documents is comfortably more than one briefing touches, and caching matters
    because expanding N hits from one 10-K would otherwise re-read a 300KB file N times.
    """
    return Path(path).read_text()


class ChunkStore:
    def __init__(self, root: str | Path = "data/index") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "chunks.sqlite"
        self.vec_path = self.root / "vectors.f16.npy"
        with self._conn() as c:
            c.executescript(SCHEMA)
        self._vectors: np.ndarray | None = None

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ write

    def reset(self) -> None:
        """Drop and recreate. Contentless FTS5 has no usable in-place update path."""
        with self._conn() as c:
            c.executescript(
                "DROP TABLE IF EXISTS chunks;"
                "DROP TABLE IF EXISTS chunks_fts;"
                "DROP TABLE IF EXISTS meta;")
            c.executescript(SCHEMA)
        self.vec_path.unlink(missing_ok=True)
        self._vectors = None
        read_clean.cache_clear()

    def add_batch(self, rows: list[tuple], texts: list[str], first_id: int) -> None:
        """Insert chunks and their FTS entries under explicit, contiguous ids.

        Ids are assigned by the caller rather than by AUTOINCREMENT because the vector
        memmap is positional: row i must be chunk i+1, and letting SQLite pick would
        silently decouple the two on any retry.
        """
        with self._conn() as c:
            c.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, ordinal, start, end, n_chars,
                       ticker, cik, doc_date, source_type, section, clean_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            c.executemany(
                "INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)",
                [(first_id + i, t) for i, t in enumerate(texts)])

    def set_meta(self, **kv) -> None:
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                          [(k, json.dumps(v, default=str)) for k, v in kv.items()])

    def get_meta(self, key: str, default=None):
        with self._conn() as c:
            r = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    # ------------------------------------------------------------------- read

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def max_id(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COALESCE(MAX(chunk_id), 0) FROM chunks").fetchone()[0]

    def indexed_docs(self) -> set[str]:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT DISTINCT doc_id FROM chunks")}

    def get(self, chunk_ids: list[int]) -> list[ChunkRef]:
        if not chunk_ids:
            return []
        q = ("SELECT * FROM chunks WHERE chunk_id IN "
             f"({','.join('?' * len(chunk_ids))})")
        with self._conn() as c:
            rows = {r["chunk_id"]: _to_ref(r) for r in c.execute(q, chunk_ids)}
        return [rows[i] for i in chunk_ids if i in rows]   # preserve caller ranking

    def candidate_ids(self, tickers: list[str] | None = None,
                      as_of: date | None = None, since: date | None = None,
                      source_types: list[str] | None = None,
                      sections: list[str] | None = None) -> list[int]:
        """Chunk ids passing the hard metadata filter.

        The filter runs BEFORE any scoring, and it is the part of retrieval most likely
        to be load-bearing: `as_of` is a leakage boundary, not a preference. A chunk from
        a filing published after the decision date must never be scoreable, so it is
        excluded in SQL rather than down-weighted.
        """
        where, params = ["1=1"], []
        if tickers:
            where.append(f"ticker IN ({','.join('?' * len(tickers))})")
            params += tickers
        if as_of:
            where.append("doc_date <= ?")
            params.append(as_of.isoformat())
        if since:
            where.append("doc_date >= ?")
            params.append(since.isoformat())
        if source_types:
            where.append(f"source_type IN ({','.join('?' * len(source_types))})")
            params += source_types
        if sections:
            where.append(f"section IN ({','.join('?' * len(sections))})")
            params += sections
        with self._conn() as c:
            return [r[0] for r in c.execute(
                f"SELECT chunk_id FROM chunks WHERE {' AND '.join(where)}", params)]

    def neighbours(self, ref: ChunkRef, radius: int = 1) -> list[ChunkRef]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM chunks WHERE doc_id = ? AND ordinal BETWEEN ? AND ?
                   ORDER BY ordinal""",
                (ref.doc_id, ref.ordinal - radius, ref.ordinal + radius)).fetchall()
        return [_to_ref(r) for r in rows]

    # ---------------------------------------------------------------- vectors

    def open_vectors(self, dim: int, n: int, mode: str = "r+") -> np.ndarray:
        """Open (creating if needed) the vector memmap."""
        if mode == "w+" or not self.vec_path.exists():
            arr = np.lib.format.open_memmap(
                self.vec_path, mode="w+", dtype=np.float16, shape=(n, dim))
            return arr
        return np.lib.format.open_memmap(self.vec_path, mode=mode)

    def grow_vectors(self, dim: int, new_n: int) -> np.ndarray:
        """Extend the vector memmap to `new_n` rows, preserving every row already there.

        Exists so adding a new corpus (e.g. a second universe) does not force re-embedding
        chunks that were already embedded -- growing in place vs rebuilding from scratch
        was the difference, first measured on the semis corpus, between minutes and
        another ~35-minute pass. `np.memmap` cannot be resized in place, so this writes a
        new file, copies the old rows across, and atomically replaces the original.
        """
        old = self.vectors
        old_n = 0 if old is None else old.shape[0]
        if old_n >= new_n:
            return self.open_vectors(dim, old_n, mode="r+")

        tmp = self.vec_path.with_suffix(".growing.npy")
        new = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(new_n, dim))
        if old_n:
            new[:old_n] = np.asarray(old)
        new.flush()
        del new
        self._vectors = None
        tmp.replace(self.vec_path)
        return np.lib.format.open_memmap(self.vec_path, mode="r+")

    @property
    def vectors(self) -> np.ndarray | None:
        """Read-only view of the vector matrix, or None if it was never built."""
        if self._vectors is None and self.vec_path.exists():
            self._vectors = np.load(self.vec_path, mmap_mode="r")
        return self._vectors


def _to_ref(r: sqlite3.Row) -> ChunkRef:
    d = r["doc_date"]
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except ValueError:
            d = None
    return ChunkRef(chunk_id=r["chunk_id"], doc_id=r["doc_id"], ordinal=r["ordinal"],
                    start=r["start"], end=r["end"], ticker=r["ticker"], doc_date=d,
                    source_type=r["source_type"], section=r["section"],
                    clean_path=r["clean_path"])
