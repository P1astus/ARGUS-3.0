"""Retained-document store.

The Phase 4 gate requires that every sourced claim be checkable against the document it
cites. That is only possible if the raw document is retained at retrieval time -- refetching
later gives a different page, and for news often no page at all.

The store is content-addressed, so the same document retrieved twice is stored once, and a
claim's quote can be verified against exactly the bytes the model was shown.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from argus.contracts.briefing import Briefing, Claim, ClaimKind, Source, SourceType

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source_id       TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    url             TEXT,
    title           TEXT,
    published       DATE,
    retrieved_at    TIMESTAMP NOT NULL,
    content_sha256  TEXT NOT NULL,
    content_path    TEXT NOT NULL,
    redistributable INTEGER NOT NULL DEFAULT 0,
    training_permitted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_prov_url ON documents (url);
"""


class ProvenanceStore:
    def __init__(self, root: str | Path = "data/provenance") -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "provenance.sqlite"
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def put(self, content: str, source_type: SourceType, url: str | None = None,
            title: str | None = None, published=None,
            redistributable: bool = False, training_permitted: bool = False) -> Source:
        """Retain a document and return its Source record."""
        digest = hashlib.sha256(content.encode()).hexdigest()
        source_id = digest[:24]
        path = self.blobs / f"{source_id}.txt"
        if not path.exists():
            path.write_text(content)

        src = Source(
            source_id=source_id, source_type=source_type, url=url, title=title,
            published=published, retrieved_at=datetime.now(timezone.utc),
            content_sha256=digest, redistributable=redistributable,
            training_permitted=training_permitted,
        )
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO documents
                (source_id, source_type, url, title, published, retrieved_at,
                 content_sha256, content_path, redistributable, training_permitted)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (source_id, str(source_type), url, title, published, src.retrieved_at,
                 digest, str(path), int(redistributable), int(training_permitted)))
        return src

    def get_text(self, source_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT content_path FROM documents WHERE source_id = ?",
                            (source_id,)).fetchone()
        if row is None:
            return None
        p = Path(row["content_path"])
        return p.read_text() if p.exists() else None

    def documents_for(self, briefing: Briefing) -> dict[str, str]:
        out = {}
        for s in briefing.sources:
            t = self.get_text(s.source_id)
            if t is not None:
                out[s.source_id] = t
        return out


def audit_briefing(briefing: Briefing, store: ProvenanceStore) -> dict:
    """The Phase 4 gate, executed rather than eyeballed.

    Every SOURCED claim must quote a span that literally appears in the retained document.
    A model that paraphrases while presenting a quote as verbatim is doing precisely what
    the gate exists to catch, and prose review would not reliably notice.
    """
    docs = store.documents_for(briefing)
    unverified = briefing.unverified_quotes(docs)
    n_sourced = sum(1 for c in briefing.claims if c.kind == ClaimKind.SOURCED)

    return {
        "n_claims": len(briefing.claims),
        "n_sourced": n_sourced,
        "n_inference": len(briefing.claims) - n_sourced,
        "sourced_fraction": briefing.sourced_fraction,
        "n_unverified_quotes": len(unverified),
        "quote_verification_rate": (
            1.0 - len(unverified) / n_sourced if n_sourced else float("nan")),
        "unverified": [{"text": c.text[:120], "source_id": c.source_id,
                        "quote": (c.quote or "")[:120]} for c in unverified],
        "passed": len(unverified) == 0,
    }
