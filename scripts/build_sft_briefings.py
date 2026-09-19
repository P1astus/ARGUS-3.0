"""Reconstruct real Briefing objects from the 20 existing baseline markdown artifacts.

WHY RECONSTRUCTION, NOT A FRESH EXTRACTION RUN
`baseline.py`'s run_arm() only ever persisted rendered markdown + raw model output for
each of the 20 DEFAULT_CASES/TECH_CASES cases -- never the structured Briefing object
sft_data.py's build_prompt() needs. A fresh extraction pass would reproduce it, but each
case measured ~150-300s in the foreground (handoff), so 20 cases is another ~1-1.5h of
compute to get back to data this repo already paid for once. The markdown is a faithful,
lossless-enough record of what sft_data.render_evidence() actually consumes: every
SOURCED claim's `text` line under "## Sourced" is copied verbatim from the real Briefing
that produced it. Reconstructing from that text is strictly cheaper than re-running
extraction to regenerate the same string.

WHAT IS AND ISN'T FAITHFUL ABOUT THE RECONSTRUCTION
Claim.text is exact -- byte-for-byte the same string the original extraction produced,
because to_markdown() emits it unmodified (export.py). Claim.quote is NOT reliable: the
export truncates quotes to 400 chars with a trailing "..." marker, whereas verified
quotes could run much longer. This is fine for THIS use, because sft_data.render_evidence
never shows the quote to the generation model, only claim.text -- but a reconstructed
Briefing must never be mistaken for one that would pass a fresh Phase 4 provenance audit.
Source objects are synthetic placeholders (content_sha256="reconstructed", not a real
hash) that exist only to satisfy Briefing's "every claim's source_id is known" validator.

Run: PYTHONPATH=src .venv/bin/python scripts/build_sft_briefings.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from argus.contracts.briefing import Briefing, Claim, ClaimKind, Source, SourceType
from argus.engine.extract.baseline import DEFAULT_CASES, TECH_CASES, BaselineCase

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "sft" / "briefings"

CLAIM_RE = re.compile(r"^- \*\*\[(?P<tag>S\d+)\]\*\* (?P<text>.+)$")
QUOTE_RE = re.compile(r"^  > (?P<line>.*)$")


def parse_sourced_claims(md_text: str) -> list[tuple[str, str, str]]:
    """(tag, text, quote) for every claim under "## Sourced" -- stops at the next "## "
    heading so the Inference/Sources/audit sections are never mistaken for claims."""
    lines = md_text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("## Sourced")]
    if not starts:
        return []
    start = starts[0]
    ends = [i for i, l in enumerate(lines[start + 1:], start + 1) if l.startswith("## ")]
    end = ends[0] if ends else len(lines)
    section = lines[start + 1:end]

    claims: list[tuple[str, str, str]] = []
    i = 0
    while i < len(section):
        m = CLAIM_RE.match(section[i])
        if not m:
            i += 1
            continue
        tag, text = m.group("tag"), m.group("text")
        i += 1
        quote_lines = []
        while i < len(section) and QUOTE_RE.match(section[i]):
            quote_lines.append(QUOTE_RE.match(section[i]).group("line"))
            i += 1
        claims.append((tag, text, "\n".join(quote_lines).strip()))
    return claims


def briefing_from_markdown(md_path: Path, case: BaselineCase) -> Briefing:
    parsed = parse_sourced_claims(md_path.read_text())
    tags = sorted({tag for tag, _, _ in parsed})

    now = datetime.now(timezone.utc)
    sources = [Source(source_id=f"reconstructed:{case.ticker}:{case.as_of.isoformat()}:{tag}",
                      source_type=SourceType.SEC_FILING, retrieved_at=now,
                      content_sha256="reconstructed") for tag in tags]
    sid_by_tag = {tag: s.source_id for tag, s in zip(tags, sources)}

    claims = [Claim(text=text, kind=ClaimKind.SOURCED, source_id=sid_by_tag[tag],
                    quote=quote or text)
             for tag, text, quote in parsed]
    return Briefing(ticker=case.ticker, sub_segment=case.segment, as_of=case.as_of,
                    claims=claims, sources=sources)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_dirs = [(ROOT / "artifacts" / "baseline" / "base", DEFAULT_CASES),
                (ROOT / "artifacts" / "baseline_tech" / "base", TECH_CASES)]

    written, missing = 0, []
    for base_dir, cases in case_dirs:
        for case in cases:
            md_path = base_dir / f"{case.ticker}_{case.as_of.isoformat()}.md"
            if not md_path.exists():
                missing.append(str(md_path))
                continue
            briefing = briefing_from_markdown(md_path, case)
            out_path = OUT_DIR / f"{case.ticker}_{case.as_of.isoformat()}.json"
            out_path.write_text(briefing.model_dump_json(indent=2))
            n_sourced = sum(1 for c in briefing.claims if c.kind == ClaimKind.SOURCED)
            print(f"{case.ticker} {case.as_of}: {n_sourced} sourced claims -> {out_path}")
            written += 1

    print(f"\n{written} briefings written to {OUT_DIR}")
    if missing:
        print(f"{len(missing)} case(s) had no markdown artifact:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
