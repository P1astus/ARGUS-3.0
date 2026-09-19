"""HTML/XBRL extraction and narrative section selection.

The reference DAPT study (arXiv 2512.12384) used a conservative whitelist -- MD&A, Risk
Factors, Business Overview, CD&A -- and discarded ~9.6% of filings for insufficient text.
Against our measured raw sizes that implies ~24% retention.

We keep somewhat more than they did, because segment discussion and guidance language are
exactly the reasoning ARGUS needs, but the principle holds: financial statement tables and
exhibit boilerplate are mostly XBRL noise, and training on them teaches formatting rather
than sector dynamics.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Section headers as they appear in 10-K/10-Q filings. Matching is deliberately loose --
# issuers vary punctuation and capitalisation, and a missed section costs more than an
# occasional false positive.
SECTION_PATTERNS: dict[str, re.Pattern] = {
    "business": re.compile(r"(?i)item\s*1\s*[.\-:\s]*business"),
    "risk_factors": re.compile(r"(?i)item\s*1a\s*[.\-:\s]*risk\s*factors"),
    "mdna": re.compile(
        r"(?i)item\s*[27]\s*[.\-:\s]*management.{0,5}s\s*discussion"),
    "quantitative_qualitative": re.compile(
        r"(?i)item\s*[37]a\s*[.\-:\s]*quantitative\s*and\s*qualitative"),
    "properties": re.compile(r"(?i)item\s*2\s*[.\-:\s]*propert"),
    "legal": re.compile(r"(?i)item\s*3\s*[.\-:\s]*legal\s*proceedings"),
}

KEEP_SECTIONS = ("business", "risk_factors", "mdna", "quantitative_qualitative")

# 20-F (annual report for foreign private issuers -- TSM, UMC, ASML, STM and 8 more of
# the universe's semis file this and only this, never a 10-K, per SEC's own submissions
# API). Item numbering does not correspond to 10-K's at all: Item 5 is the MD&A
# equivalent, not Item 7; risk factors sit inside Item 3 ("Key Information") rather than
# their own item. VERIFIED against a real filing (TSM, accession 0001628280-26-025362)
# before trusting this at scale -- the item headers are genuine, well-formed "ITEM N.
# TITLE" text, so the same header-position extraction strategy applies, just with
# different anchors.
SECTION_PATTERNS_20F: dict[str, re.Pattern] = {
    # Item 3 also covers capitalization/indebtedness and reasons-for-the-offer, which a
    # 10-K's Item 1A does not -- broader than the 10-K risk_factors span, same tradeoff
    # this file already accepts elsewhere ("a missed section costs more than an
    # occasional false positive").
    "risk_factors": re.compile(r"(?i)item\s*3\s*[.\-:\s]*key\s*information"),
    "business": re.compile(r"(?i)item\s*4\s*[.\-:\s]*information\s*on\s*the\s*company"),
    "mdna": re.compile(
        r"(?i)item\s*5\s*[.\-:\s]*operating\s*and\s*financial\s*review"),
    "quantitative_qualitative": re.compile(
        r"(?i)item\s*11\s*[.\-:\s]*quantitative\s*and\s*qualitative"),
}

# 40-F (annual report for Canadian issuers using the US/Canada Multijurisdictional
# Disclosure System -- SHOP's real pre-2025 filing history, per the date-coverage audit).
# Unlike 10-K/20-F, a 40-F's own primary document is pure inline-XBRL cover-page metadata
# with essentially no narrative text -- VERIFIED against a real filing (SHOP, accession
# 0001594805-24-000007): the primary doc is 241KB and almost entirely XBRL tag soup. The
# actual content (MD&A, Annual Information Form) lives in separately-filed exhibits, the
# same "substance is in the exhibit, not the cover document" shape as 8-K, just with a
# different exhibit-naming convention and no SEC Item numbering inside the exhibit either
# -- Canadian filers use National Instrument 51-102's own heading conventions instead.
#
# VERIFIED against a large, established 40-F filer (Toronto-Dominion Bank, CIK 947263,
# not SHOP -- deliberately a second, independent filer, the same discipline applied to
# the 20-F caps-preference fix) to check these headings generalise rather than being
# SHOP-specific: "Risk Factors" and "Management's Discussion and Analysis" both appear
# multiple times (table of contents, cross-references) with the REAL section header as
# the LAST occurrence in the document -- same last-occurrence heuristic as 10-K/20-F,
# and unlike 20-F, headings here are NOT reliably ALL-CAPS (some are, some are Title
# Case), so `prefer_caps` is not used for this form.
SECTION_PATTERNS_40F: dict[str, re.Pattern] = {
    "business": re.compile(r"(?i)description\s+of\s+the\s+business"),
    "risk_factors": re.compile(r"(?i)\brisk\s+factors\b"),
    "mdna": re.compile(r"(?i)management.{0,5}s\s+discussion\s+and\s+analysis"),
}

# Dispatch by the form type recorded on the document, not a global default -- a 20-F run
# through the 10-K patterns matches nothing (safe: `extract_narrative` falls back to
# whole-document text, see below) and a 10-K run through the 20-F patterns is equally
# inert, but getting the wrong one silently would waste the whole document.
#
# "40-F/EX" (not "40-F") is the key on purpose: the primary 40-F document itself never
# has narrative content (see above), only its exhibits do, and exhibits are catalogued
# under a distinct source_type -- same convention 8-K/EX already established.
PATTERNS_BY_FORM: dict[str, dict[str, re.Pattern]] = {
    "20-F": SECTION_PATTERNS_20F,
    "40-F/EX": SECTION_PATTERNS_40F,
}

_SCRIPT = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAG = re.compile(r"(?s)<[^>]+>")
_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")

# Rows that are almost entirely digits/punctuation are financial-statement tables. In
# extracted text they become unreadable number soup that teaches the model nothing about
# sector dynamics while consuming a large share of the token budget.
_NUMERIC_ROW = re.compile(r"^[\s\d.,()%$\-+/]*$")


def strip_html(raw: str) -> str:
    t = _SCRIPT.sub(" ", raw)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", t)
    t = _TAG.sub(" ", t)
    t = _ENTITY.sub(" ", t)
    t = _WS.sub(" ", t)
    return _BLANKS.sub("\n\n", t).strip()


def drop_numeric_tables(text: str, min_alpha_ratio: float = 0.35) -> str:
    """Remove lines that are predominantly numbers.

    Keeps prose that happens to contain figures ("revenue grew 12% to $4.1 billion") while
    dropping extracted table rows, which carry the same information in a form the model
    cannot learn from.
    """
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if len(s) < 4:
            continue
        if _NUMERIC_ROW.match(s):
            continue
        alpha = sum(c.isalpha() for c in s)
        if alpha / max(len(s), 1) < min_alpha_ratio:
            continue
        kept.append(s)
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip()


def find_sections(text: str,
                  patterns: dict[str, re.Pattern] = SECTION_PATTERNS,
                  prefer_caps: bool = False) -> dict[str, tuple[int, int]]:
    """Locate narrative sections by header position.

    Filings repeat their item headers in the table of contents, so the LAST occurrence is
    used -- the table of contents comes first, the actual section later.

    `prefer_caps`: 20-F filings (VERIFIED against a real TSM filing) turn out to litter
    their narrative with mixed-case cross-references -- "see Item 4. Information on the
    Company for further discussion" -- that also match the header pattern and can sort
    after the genuine header, which is rendered ALL-CAPS. Picking the last ALL-CAPS match
    when any exist recovers the real header; 10-K filings do not show this problem in the
    existing corpus, so this stays opt-in rather than changing default behaviour that
    18,000+ already-processed documents depend on.
    """
    hits: list[tuple[int, str]] = []
    for name, pat in patterns.items():
        matches = list(pat.finditer(text))
        if not matches:
            continue
        if prefer_caps:
            caps = [m for m in matches if text[m.start():m.start() + 4].isupper()]
            matches = caps or matches
        hits.append((matches[-1].start(), name))

    hits.sort()
    spans: dict[str, tuple[int, int]] = {}
    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans[name] = (start, end)
    return spans


def extract_narrative(raw_html: str, keep: tuple[str, ...] = KEEP_SECTIONS,
                      min_chars: int = 500, form_type: str | None = None) -> dict[str, str]:
    """HTML -> {section_name: cleaned text}.

    Returns whole-document text under the key "full" when no section headers are found --
    true for 8-K exhibits and press releases, which have no item structure but are highly
    relevant. Also the safe fallback if `form_type` names a form ARGUS has no pattern set
    for: no spans match, so the document is not lost, just not section-split.

    `form_type` selects the header pattern set -- 20-F's Item 5 is not 10-K's Item 7, and
    running one form through the other's patterns finds nothing (§ see PATTERNS_BY_FORM).
    """
    text = strip_html(raw_html)
    if len(text) < min_chars:
        return {}

    patterns = PATTERNS_BY_FORM.get(form_type, SECTION_PATTERNS)
    spans = find_sections(text, patterns, prefer_caps=(form_type == "20-F"))
    if not spans:
        cleaned = drop_numeric_tables(text)
        return {"full": cleaned} if len(cleaned) >= min_chars else {}

    out: dict[str, str] = {}
    for name in keep:
        if name not in spans:
            continue
        start, end = spans[name]
        cleaned = drop_numeric_tables(text[start:end])
        if len(cleaned) >= min_chars:
            out[name] = cleaned
    return out
