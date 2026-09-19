"""EDGAR acquisition.

Two measured facts drive this design.

EXHIBIT-AWARE FETCHING (the important one)
An 8-K's primary document is a cover page -- measured at 0.5-1.1k tokens. The substance
(earnings releases, guidance, CFO commentary) lives in EX-99 exhibits, reachable only by
walking each filing's index.json. Measured across 6 issuers, narrative EX-99 exhibits
average ~4.1k tokens and appear roughly quarterly. A pipeline that fetches only
`primaryDocument` silently discards the most trading-relevant text in EDGAR.
This roughly triples the request count, and is why acquisition takes hours not minutes.

ENUMERATION VIA FULL-INDEX
The quarterly master.idx is ~33MB and lists every filing, so enumeration does not require
paginating per-company submissions. For a ~100-name universe the submissions API is still
simpler, so that is the default; the full-index path exists for whole-market sweeps.

Rate limit is 10 req/s, IP-throttled. We target 8/s with backoff.
"""

from __future__ import annotations

import os
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from argus.engine.corpus.catalog import Catalog, DocStatus, Document

log = logging.getLogger(__name__)

SEC_UA = os.environ.get("ARGUS_SEC_UA", "ARGUS Research contact@example.com")
RATE_LIMIT_PER_SEC = 8.0        # SEC allows 10; leave headroom
FORMS = ("10-K", "10-Q", "8-K")

# Exhibit names vary widely -- NVDA files `q1fy27cfocommentary.htm`, which an `ex.?99`
# filename filter misses entirely. Match on the exhibit TYPE from index.json where
# available, and fall back to a permissive filename pattern.
EXHIBIT_PATTERN = re.compile(r"(?i)(ex.?99|cfocommentary|prepared.?remark|commentary|"
                             r"press.?release|earnings)")

# 40-F exhibits: measured across 6 filers (SHOP plus 5 large, independent Canadian
# issuers -- CNI, ENB, TD, BCE, BMO) that most use the SAME ex99N convention EXHIBIT_
# PATTERN already catches (tm..._ex99-3.htm, ex991.htm, d86374dex991.htm, ...). SHOP is
# the outlier: `exhibit13mdaq42023.htm`, `exhibit11annualinformation.htm` -- descriptive
# names EXHIBIT_PATTERN does not match. This OR's in a bare "exhibit\d+" catch rather
# than editing EXHIBIT_PATTERN itself, so 8-K's already-proven matching is untouched.
FORTYF_EXHIBIT_PATTERN = re.compile(EXHIBIT_PATTERN.pattern + r"|exhibit\s*\d+")
# Full-submission text dumps duplicate every exhibit plus XBRL; excluding them prevents
# counting the same content many times over.
FULL_SUBMISSION = re.compile(r"^\d{10}-?\d{2}-?\d{6}\.txt$")
SKIP_FILES = re.compile(r"(?i)(_cal|_def|_lab|_pre|_htm\.xml|\.xsd|R\d+\.htm|FilingSummary)")


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self) -> None:
        dt = time.perf_counter() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.perf_counter()


class EdgarClient:
    def __init__(self, user_agent: str = SEC_UA, rate: float = RATE_LIMIT_PER_SEC) -> None:
        self.ua = user_agent
        self.limiter = RateLimiter(rate)

    def _get(self, url: str, retries: int = 3) -> bytes:
        for attempt in range(retries):
            self.limiter.wait()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.ua})
                return urllib.request.urlopen(req, timeout=30).read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Backoff on throttling rather than hammering; the SEC bans by IP.
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 404:
                    raise
                time.sleep(1 + attempt)
            except Exception:
                time.sleep(1 + attempt)
        raise RuntimeError(f"failed after {retries} attempts: {url}")

    def submissions(self, cik: str) -> dict:
        return json.loads(self._get(
            f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))

    def filing_index(self, cik: str, accession: str) -> dict:
        acc = accession.replace("-", "")
        return json.loads(self._get(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/index.json"))


@dataclass
class FilingRef:
    cik: str
    ticker: str
    form: str
    accession: str
    filing_date: date
    primary_doc: str


def discover(
    client: EdgarClient,
    catalog: Catalog,
    tickers: dict[str, str],          # ticker -> cik
    start: date,
    cutoff: date,
    forms: tuple[str, ...] = FORMS,
    include_exhibits: bool = True,
) -> int:
    """Enumerate filings and their exhibits into the catalog.

    Nothing is downloaded here beyond metadata; fetching is a separate resumable stage.
    Documents dated after `cutoff` are recorded as DROPPED rather than skipped, so the
    manifest can show what was excluded instead of silently omitting it.
    """
    discovered = 0

    for ticker, cik in tickers.items():
        try:
            meta = client.submissions(cik)
        except Exception as e:
            log.warning("%s: submissions lookup failed (%s)", ticker, type(e).__name__)
            continue

        recent = meta["filings"]["recent"]
        pages = [recent]
        # `recent` caps at 1000 filings; older ones live in paginated files.
        for f in meta["filings"].get("files", []):
            try:
                pages.append(json.loads(client._get(
                    "https://data.sec.gov/submissions/" + f["name"])))
            except Exception:
                log.warning("%s: could not page older filings", ticker)

        for page in pages:
            for form, acc, doc, dt in zip(page["form"], page["accessionNumber"],
                                          page["primaryDocument"], page["filingDate"]):
                if form not in forms:
                    continue
                fdate = date.fromisoformat(dt)
                if fdate < start:
                    continue

                base = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                        f"{acc.replace('-', '')}")
                post_cutoff = fdate > cutoff

                if doc:
                    catalog.upsert(Document(
                        doc_id=Document.make_id("edgar", f"{base}/{doc}"),
                        source="edgar", source_type=form, ticker=ticker, cik=cik,
                        url=f"{base}/{doc}", doc_date=fdate,
                        status=DocStatus.DROPPED if post_cutoff else DocStatus.DISCOVERED,
                        drop_reason="post_cutoff" if post_cutoff else None,
                        license_ok=True,   # EDGAR is public domain
                    ))
                    discovered += 1

                # Exhibits carry the substance; see module docstring (8-K) and
                # clean.py's SECTION_PATTERNS_40F docstring (40-F -- the primary
                # document there is pure XBRL cover-page metadata with no narrative
                # at all, an even starker version of 8-K's cover-page problem).
                exhibit_pattern = {"8-K": EXHIBIT_PATTERN,
                                   "40-F": FORTYF_EXHIBIT_PATTERN}.get(form)
                if include_exhibits and exhibit_pattern and not post_cutoff:
                    try:
                        idx = client.filing_index(cik, acc)
                    except Exception:
                        continue
                    for item in idx.get("directory", {}).get("item", []):
                        name = item.get("name", "")
                        if (not name.endswith((".htm", ".txt")) or name == doc
                                or SKIP_FILES.search(name) or FULL_SUBMISSION.match(name)
                                or not exhibit_pattern.search(name)):
                            continue
                        catalog.upsert(Document(
                            doc_id=Document.make_id("edgar", f"{base}/{name}"),
                            source="edgar", source_type=f"{form}/EX", ticker=ticker,
                            cik=cik, url=f"{base}/{name}", doc_date=fdate,
                            status=DocStatus.DISCOVERED, license_ok=True,
                        ))
                        discovered += 1

        log.info("%s: discovered filings (running total %d)", ticker, discovered)

    return discovered


def fetch_pending(
    client: EdgarClient,
    catalog: Catalog,
    raw_dir: Path = Path("data/corpus/raw/edgar"),
    limit: int | None = None,
) -> int:
    """Download discovered documents. Resumable: only DISCOVERED rows are fetched."""
    import hashlib

    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = catalog.pending(DocStatus.DISCOVERED, limit=limit)
    done = 0

    for doc in todo:
        if doc.source != "edgar":
            continue
        try:
            content = client._get(doc.url)
        except Exception as e:
            doc.status = DocStatus.DROPPED
            doc.drop_reason = f"fetch_failed:{type(e).__name__}"
            catalog.upsert(doc)
            continue

        path = raw_dir / f"{doc.doc_id}.html"
        path.write_bytes(content)
        doc.raw_path = str(path)
        doc.raw_sha256 = hashlib.sha256(content).hexdigest()
        doc.raw_bytes = len(content)
        doc.status = DocStatus.FETCHED
        catalog.upsert(doc)
        done += 1
        if done % 200 == 0:
            log.info("fetched %d/%d", done, len(todo))

    return done
