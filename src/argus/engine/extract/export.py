"""Briefing -> the hand-off the user pastes into a frontier conversation.

DESIGNED AROUND WHAT THE READER CANNOT DO
The frontier model reading this has not seen the filings and cannot fetch them. So the
export has one job beyond being readable: make the evidential status of every line
impossible to miss, and make it impossible to mistake the local model's reasoning for
something a company said.

Hence the structure:
  * sourced claims and inference claims are in SEPARATE sections, not interleaved with
    footnote markers. Interleaving is how a reader ends up treating an inference as
    reported fact -- the two look identical when skimmed.
  * every sourced claim shows its quote, verbatim, indented. Not a citation *pointer* --
    the actual span. A pointer the reader cannot follow is decoration.
  * the audit block is included in the export rather than kept in a log, so the reader
    can see the fabrication and repair counts for the extraction they are reading.

Budget: this targets 2-5k tokens against 73-100k for the underlying filings (~50x
compression). `max_claims` exists to keep that true on a heavy quarter.
"""

from __future__ import annotations

from datetime import date

from argus.contracts.briefing import Briefing, ClaimKind
from argus.tools.analyst import render_analyst_consensus
from argus.tools.insider import render_insider_activity
from argus.tools.macro import render_macro_context
from argus.tools.price_context import render_price_snapshot
from argus.tools.taxonomy import PROFILES


def to_markdown(briefing: Briefing, audit: dict | None = None,
                max_claims: int = 40, quote_chars: int = 400) -> str:
    b = briefing
    p = PROFILES[b.sub_segment]
    sourced = [c for c in b.claims if c.kind == ClaimKind.SOURCED][:max_claims]
    inference = [c for c in b.claims if c.kind == ClaimKind.INFERENCE][:max_claims]

    by_id = {s.source_id: s for s in b.sources}
    # Short labels. A 24-hex id is unreadable in prose and invites the reader to skip
    # the citation entirely, which defeats the point of citing.
    short = {sid: f"S{i + 1}" for i, sid in enumerate(by_id)}

    out: list[str] = []
    out.append(f"# {b.ticker} briefing — as of {b.as_of.isoformat()}")
    out.append("")
    out.append(f"**Sub-segment:** {p.name}  ")
    out.append(f"**Cycle driver:** {p.cycle_driver}  ")
    out.append(f"**Timing:** {p.lead_lag}")
    out.append("")
    out.append(
        "> Assembled locally from SEC filings by retrieval + extraction. "
        "Every line under *Sourced* is backed by the verbatim span shown beneath it, "
        "located in the retained document by exact string match. Lines under "
        "*Inference* are the local model's reasoning and are **not** attributable to "
        "any source — treat them as hypotheses to test, not as evidence."
    )
    out.append("")

    # Structured appendices -- displayed directly, not extracted, so they carry no quote
    # to verify (contracts/context.py's docstring explains why). Each is independently
    # optional; a briefing built without price/insider/macro data just omits the section
    # rather than showing an empty placeholder.
    if b.price is not None:
        out.append("## Price context")
        out.append("")
        out.append(render_price_snapshot(b.price))
        out.append("")
    if b.insider_activity is not None:
        out.append("## Insider activity (Form 3/4/5)")
        out.append("")
        out.append(render_insider_activity(b.insider_activity))
        out.append("")
    if b.macro is not None:
        out.append("## Macro context")
        out.append("")
        out.append(render_macro_context(b.macro))
        out.append("")
    if b.analyst is not None:
        out.append("## Analyst consensus")
        out.append("")
        out.append(render_analyst_consensus(b.analyst))
        out.append("")

    out.append(f"## Sourced ({len(sourced)})")
    out.append("")
    if not sourced:
        out.append("_No claim survived quote verification. Treat this briefing as empty "
                   "rather than as an absence of news._")
        out.append("")
    for c in sourced:
        src = by_id.get(c.source_id or "")
        tag = short.get(c.source_id or "", "?")
        out.append(f"- **[{tag}]** {c.text}")
        q = (c.quote or "").strip()
        if len(q) > quote_chars:
            q = q[:quote_chars].rstrip() + " …"
        for line in q.splitlines():
            out.append(f"  > {line}")
        if src and src.published:
            out.append(f"  <sub>{src.title or ''} — {src.published.isoformat()}</sub>")
        out.append("")

    out.append(f"## Inference ({len(inference)}) — not sourced")
    out.append("")
    if not inference:
        out.append("_None._")
        out.append("")
    for c in inference:
        out.append(f"- {c.text}")
    out.append("")

    out.append("## Sources")
    out.append("")
    for sid, s in by_id.items():
        out.append(f"- **{short[sid]}** — {s.title or s.url or s.source_type} "
                   f"(`{sid[:12]}`, sha256 `{s.content_sha256[:12]}`)")
    out.append("")

    out.append("## Extraction audit")
    out.append("")
    out.append(f"- claims: {len(b.claims)} "
               f"({b.sourced_fraction:.0%} sourced)")
    if audit:
        ex = _pct(audit.get("exact_quote_rate"))
        fab = _pct(audit.get("fabrication_rate"))
        out.append(f"- quotes copied character-perfect: {ex}")
        out.append(f"- quotes repaired (whitespace only): "
                   f"{audit.get('quote_whitespace', 0)}")
        out.append(f"- claims dropped, quote not in any shown passage: "
                   f"{audit.get('quote_fabricated', 0)} ({fab} of attempts)")
        out.append(f"- malformed output lines dropped: "
                   f"{audit.get('malformed_json', 0)}")
        out.append(f"- extraction model: {audit.get('model', 'unknown')}")
    out.append("")
    out.append("## Suggested questions for this briefing")
    out.append("")
    for w in p.watch_items:
        out.append(f"- {w[0].upper()}{w[1:]}: is it visible in the sourced claims above, "
                   f"and if not, is the absence informative?")
    out.append("")
    return "\n".join(out)


def _pct(x) -> str:
    """Format a rate, tolerating None and the NaN that means 'no attempts to rate'."""
    if not isinstance(x, (int, float)) or x != x:
        return "n/a"
    return f"{x:.0%}"


def filename(briefing: Briefing, as_of: date | None = None) -> str:
    d = (as_of or briefing.as_of).isoformat()
    return f"{briefing.ticker}_{d}_briefing.md"
