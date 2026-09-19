"""Insider activity (Form 3/4/5) for a briefing.

WHY NOT THE CPT CORPUS PIPELINE
Form 4 is structured XML, not narrative prose -- there is no "section" to extract, no
Item numbering, nothing `clean.py`'s pattern-matching machinery is built for. This reuses
`EdgarClient`/`resolve_ciks` (the same primitives `sources/edgar.py` and `live_edgar.py`
use) but parses directly rather than routing through the corpus builder.

WHAT "measured, not assumed" CAUGHT HERE
Checked two real filers' raw XML before trusting the schema: NVIDIA's filing agent writes
boolean flags as "1"/"0"; Apple's writes "true"/"false". Same SEC schema, different filing
agents, different literal encoding. `_flag()` below normalises both rather than only
handling the one checked first.

OPEN-MARKET VS ROUTINE COMPENSATION
Transaction code P (open-market purchase) and S (open-market sale) reflect a discretionary
decision. A (grant/award), M (option exercise), F (tax withholding), G (gift), C
(conversion) are routine compensation mechanics that happen to insiders on a schedule, not
a signal about what they think of the stock. `InsiderTransactionCode.is_open_market` keeps
these distinguishable rather than reporting "N insider transactions" as if they were all
the same kind of information.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date

from argus.contracts.context import InsiderActivity, InsiderTransactionCode
from argus.engine.corpus.build import resolve_ciks
from argus.engine.corpus.sources.edgar import SEC_UA, EdgarClient

log = logging.getLogger(__name__)

OWNERSHIP_FORMS = ("3", "4", "5", "3/A", "4/A", "5/A")
OPEN_MARKET_CODES = {"P", "S"}


def _text(elem: ET.Element | None, path: str) -> str | None:
    """Reads `<path><value>X</value></path>` where present, falls back to direct text --
    SEC's own XML mixes both styles for different fields within the same document."""
    if elem is None:
        return None
    node = elem.find(path)
    if node is None:
        return None
    value = node.find("value")
    text = (value.text if value is not None else node.text) or ""
    return text.strip() or None


def _flag(elem: ET.Element | None, tag: str) -> bool:
    raw = _text(elem, tag)
    return raw is not None and raw.strip().lower() in ("1", "true")


def _float(elem: ET.Element | None, path: str) -> float | None:
    raw = _text(elem, path)
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _parse_filing(raw_xml: bytes, filing_date: date) -> list[InsiderTransactionCode]:
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        log.warning("insider filing: unparseable XML (%s)", e)
        return []

    owner = root.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    owner_name = _text(owner, "reportingOwnerId/rptOwnerName") or "unknown"
    officer_title = None
    if rel is not None:
        node = rel.find("officerTitle")
        officer_title = (node.text.strip() if node is not None and node.text else None)

    out = []
    table = root.find("nonDerivativeTable")
    if table is None:
        return out

    for txn in table.findall("nonDerivativeTransaction"):
        txn_date_raw = _text(txn, "transactionDate")
        if not txn_date_raw:
            continue
        code = _text(txn, "transactionCoding/transactionCode") or "?"
        acquired_raw = _text(txn, "transactionAmounts/transactionAcquiredDisposedCode")
        shares = _float(txn, "transactionAmounts/transactionShares")
        if shares is None:
            continue

        out.append(InsiderTransactionCode(
            filing_date=filing_date,
            transaction_date=date.fromisoformat(txn_date_raw),
            owner_name=owner_name,
            is_officer=_flag(rel, "isOfficer"),
            is_director=_flag(rel, "isDirector"),
            is_ten_pct_owner=_flag(rel, "isTenPercentOwner"),
            officer_title=officer_title,
            code=code,
            is_open_market=code in OPEN_MARKET_CODES,
            shares=shares,
            price_per_share=_float(txn, "transactionAmounts/transactionPricePerShare"),
            shares_owned_after=_float(txn, "postTransactionAmounts/"
                                      "sharesOwnedFollowingTransaction"),
            acquired=(acquired_raw == "A"),
        ))
    return out


def build_insider_activity(ticker: str, as_of: date, client: EdgarClient | None = None,
                           lookback_days: int = 180, limit: int = 20,
                           user_agent: str = SEC_UA) -> InsiderActivity | None:
    """Form 3/4/5 transactions filed in the `lookback_days` before `as_of`.

    `limit` bounds how many filings get fetched (one request per filing, on top of the
    submissions lookup) -- a name with unusually heavy insider filing activity should not
    turn one briefing into dozens of network round trips.
    """
    client = client or EdgarClient(user_agent)
    ciks = resolve_ciks([ticker], user_agent)
    cik = ciks.get(ticker)
    if cik is None:
        log.info("%s: no SEC CIK, cannot fetch insider activity", ticker)
        return None

    try:
        sub = client.submissions(cik)
    except Exception as e:
        log.warning("%s: submissions lookup failed for insider activity (%s)",
                   ticker, type(e).__name__)
        return None

    recent = sub["filings"]["recent"]
    from datetime import timedelta
    earliest = as_of - timedelta(days=lookback_days)

    rows = []
    for form, acc, doc, dt in zip(recent["form"], recent["accessionNumber"],
                                  recent["primaryDocument"], recent["filingDate"]):
        if form not in OWNERSHIP_FORMS or not doc:
            continue
        fdate = date.fromisoformat(dt)
        if fdate > as_of or fdate < earliest:
            continue
        rows.append((acc, doc, fdate))
    rows = rows[:limit]

    transactions: list[InsiderTransactionCode] = []
    for acc, doc, fdate in rows:
        acc_clean = acc.replace("-", "")
        # `doc` is the XSLT-viewer path (e.g. "xslF345X06/form4.xml"); the raw XML lives
        # at the filing root under the same basename, not the viewer subdirectory.
        basename = doc.rsplit("/", 1)[-1]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{basename}"
        try:
            raw = client._get(url)
        except Exception as e:
            log.warning("%s: insider filing fetch failed for %s (%s)",
                       ticker, url, type(e).__name__)
            continue
        transactions.extend(_parse_filing(raw, fdate))

    return InsiderActivity(ticker=ticker, as_of=as_of, transactions=transactions)


def render_insider_activity(a: InsiderActivity) -> str:
    if not a.transactions:
        return "No Form 3/4/5 filings in the lookback window."

    lines = [f"Open-market: {a.open_market_buys} buy(s), {a.open_market_sells} sell(s) "
             f"among {len(a.transactions)} total reported transaction(s)."]
    for t in sorted(a.transactions, key=lambda t: t.transaction_date, reverse=True)[:10]:
        role = t.officer_title or ("Director" if t.is_director else
                                   "10% Owner" if t.is_ten_pct_owner else "Insider")
        tag = "OPEN MARKET" if t.is_open_market else "routine"
        price = f" @ {t.price_per_share:.2f}" if t.price_per_share else ""
        verb = "acquired" if t.acquired else "disposed"
        lines.append(f"- {t.transaction_date} [{tag}] {t.owner_name} ({role}): "
                     f"{verb} {t.shares:,.0f} sh{price} (code {t.code})")
    return "\n".join(lines)
