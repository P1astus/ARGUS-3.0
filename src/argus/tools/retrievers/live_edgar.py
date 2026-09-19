"""Live EDGAR retriever -- the `Retriever` Protocol implementation session 1 flagged as
missing ("no retrievers yet"), and every session since has left as a `CorpusRetriever`
that only covers the ~95 pre-indexed tickers.

WHY NOT A PAID NEWS API
`SourceType.NEWS`/`WEB` exist in the contracts for a reason, but every real news API
either costs money or needs an account -- and this project's own non-negotiable is to ask
before spending money, not to quietly wire up a paid dependency because the Protocol has
a slot for it. SEC's full-text and submissions APIs are free, unauthenticated, and public
domain (the same trust boundary the whole corpus already relies on), so they are the
retriever that can actually ship today.

WHAT THIS BUYS
`extract brief` currently only works for tickers already in a built corpus. This lets it
work for ANY SEC filer: resolve the ticker to a CIK, pull its most recent filings as of
the requested date directly from `data.sec.gov`, and run them through the same
`extract_narrative` used at corpus-build time (form-type-aware -- 20-F issuers get real
sections, not silent no-narrative-text drops, per this session's fix). No local index,
no pre-build step, at the cost of a live network round-trip per briefing instead of a
pre-computed one.

WHAT THIS IS NOT
Not a replacement for the corpus. It fetches ONLY the most recent few filings per call --
fine for "give me a quick read on a ticker outside the built universe", wrong for
anything needing historical depth or the retrieval quality measured in
docs/retrieval_and_extraction.md (BM25/dense/hybrid, all of which need an index this
retriever does not build).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from argus.contracts.briefing import SourceType
from argus.engine.corpus.build import resolve_ciks
from argus.engine.corpus.clean import extract_narrative
from argus.engine.corpus.sources.edgar import (EXHIBIT_PATTERN, FORTYF_EXHIBIT_PATTERN,
                                               FULL_SUBMISSION, SEC_UA, SKIP_FILES,
                                               EdgarClient)

log = logging.getLogger(__name__)

# Preference order when a filing yields multiple sections -- mirrors KEEP_SECTIONS'
# priority (handoff §4c(2): mdna and earnings releases are the high-value sections;
# risk_factors/business are the ones the section caps exist to limit).
_SECTION_PRIORITY = ("mdna", "full", "risk_factors", "business", "quantitative_qualitative")


@dataclass
class LiveEdgarRetriever:
    """Fetches a ticker's most recent SEC filings live, no local index required."""

    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K", "20-F", "40-F", "6-K")
    user_agent: str = SEC_UA
    client: EdgarClient = field(default_factory=lambda: EdgarClient(SEC_UA))
    name: str = "live_edgar"
    _cik_cache: dict[str, str | None] = field(default_factory=dict)

    def source_type(self) -> SourceType:
        return SourceType.SEC_FILING

    def license_flags(self) -> tuple[bool, bool]:
        # EDGAR is US government / public-domain data -- the one source in this project
        # where both flags are unambiguously true (same reasoning as CorpusRetriever).
        return (True, True)

    def _resolve(self, ticker: str) -> str | None:
        if ticker not in self._cik_cache:
            ciks = resolve_ciks([ticker], self.user_agent)
            self._cik_cache[ticker] = ciks.get(ticker)
        return self._cik_cache[ticker]

    def _recent_filings(self, cik: str, as_of: date, limit: int) -> list[dict]:
        sub = self.client.submissions(cik)
        recent = sub["filings"]["recent"]
        rows = []
        for form, acc, doc, dt in zip(recent["form"], recent["accessionNumber"],
                                      recent["primaryDocument"], recent["filingDate"]):
            if form not in self.forms or not doc:
                continue
            fdate = date.fromisoformat(dt)
            if fdate > as_of:
                continue    # the as_of leakage boundary -- see search.py's own filter
            rows.append({"form": form, "accession": acc, "doc": doc, "date": fdate})
        rows.sort(key=lambda r: r["date"], reverse=True)
        return rows[:limit]

    def _exhibit_names(self, cik: str, accession: str, primary_doc: str,
                      pattern: re.Pattern) -> list[str]:
        """Exhibits carrying the real content -- the primary document is a cover page
        for 8-K (handoff §6) and pure XBRL metadata with no narrative at all for 40-F
        (clean.py's SECTION_PATTERNS_40F docstring). Same filters `sources/edgar.py`'s
        corpus-build discovery already uses and has been running in production against,
        reused rather than re-derived so the live retriever and the corpus agree on what
        counts as an exhibit for a given form."""
        try:
            idx = self.client.filing_index(cik, accession)
        except Exception as e:
            log.warning("exhibit index lookup failed for %s (%s)", accession,
                       type(e).__name__)
            return []
        names = []
        for item in idx.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if (not name.endswith((".htm", ".txt")) or name == primary_doc
                    or SKIP_FILES.search(name) or FULL_SUBMISSION.match(name)
                    or not pattern.search(name)):
                continue
            names.append(name)
        return names

    def fetch(self, ticker: str, as_of: date, limit: int) -> list[tuple[str, dict]]:
        cik = self._resolve(ticker)
        if cik is None:
            log.info("%s: no SEC CIK (foreign filer under a different ticker, or not "
                     "SEC-registered)", ticker)
            return []

        try:
            filings = self._recent_filings(cik, as_of, limit)
        except Exception as e:
            log.warning("%s: submissions lookup failed (%s)", ticker, type(e).__name__)
            return []

        out: list[tuple[str, dict]] = []
        for f in filings:
            acc = f["accession"].replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}"

            docs_to_fetch = [(f["doc"], f["form"])]
            exhibit_pattern = {"8-K": EXHIBIT_PATTERN,
                               "40-F": FORTYF_EXHIBIT_PATTERN}.get(f["form"])
            if exhibit_pattern:
                # The primary document carries no narrative; exhibits carry the
                # substance. Both get fetched -- same "register the cover/metadata
                # document AND its exhibits separately" choice `sources/edgar.py`'s
                # discover() makes at corpus-build time, so a live briefing and a
                # pre-built one see the same shape of evidence.
                docs_to_fetch += [(name, f"{f['form']}/EX") for name in
                                  self._exhibit_names(cik, acc, f["doc"], exhibit_pattern)]

            for doc_name, form_type in docs_to_fetch:
                url = f"{base}/{doc_name}"
                try:
                    raw = self.client._get(url).decode("utf-8", errors="ignore")
                except Exception as e:
                    log.warning("%s: fetch failed for %s (%s)", ticker, url,
                               type(e).__name__)
                    continue

                # form_type is the primary form (e.g. "10-K", "40-F") for the primary
                # document and "<FORM>/EX" for an exhibit -- PATTERNS_BY_FORM has entries
                # for both 20-F (primary document) and 40-F/EX (exhibit, since 40-F's own
                # primary document has no narrative to speak of). Anything without a
                # registered pattern set correctly falls back to whole-document "full"
                # text rather than being mismatched.
                sections = extract_narrative(raw, form_type=form_type)
                if not sections:
                    continue
                name = next((s for s in _SECTION_PRIORITY if s in sections),
                           next(iter(sections)))
                out.append((sections[name], {
                    "url": url,
                    "title": f"{ticker} {form_type} {f['date'].isoformat()} {name}",
                    "published": f["date"],
                }))

        return out
