"""Extraction prompts.

WHY JSON LINES AND NOT ONE JSON OBJECT
The extractor runs on `Ministral-3-14B-Base` -- a *base* checkpoint, chosen in Phase 0 for
domain BPC and for having no post-training to be degraded by CPT (handoff §4b). The cost
of that choice is that it has never been instruction-tuned, so it continues patterns
rather than obeying requests. Two consequences shape this file:

  1. The prompt is few-shot and completion-shaped, not an instruction. A base model shown
     three worked examples will continue the fourth; the same model told "extract the
     claims" will write an essay.
  2. Output is one JSON object per line. A single nested object is all-or-nothing: one
     malformed brace 900 tokens in and the entire extraction is lost. With JSON Lines a
     truncated or malformed line costs one claim, and the rest still parse. Since format
     validity is a measured quantity here, the format should not manufacture failures
     that are really just length.

The same prompt is used for the instruct variant in the baseline comparison, so that the
base-vs-instruct number measures the models and not two different prompts.
"""

from __future__ import annotations

import json
import re

from argus.contracts.quant import SubSegment
from argus.tools.taxonomy import PROFILES

STOP_SEQUENCES = ("\n\n###", "\nEND", "\n---")

_FEWSHOT = '''### SOURCES

[S1] AMAT 10-Q (2019-08-15) mdna
Semiconductor Systems segment net sales decreased 22 percent in the third quarter of
fiscal 2019 compared to the corresponding period of fiscal 2018, primarily due to lower
customer investments in foundry and memory. Backlog for the segment was $2.61 billion at
the end of the quarter, compared to $3.05 billion a year earlier.

[S2] AMAT 8-K/EX (2019-08-15) full
"Applied delivered solid results in a softer market environment," said Gary Dickerson,
president and CEO. Management expects fourth quarter net sales to be approximately $3.68
billion, plus or minus $150 million.

### CLAIMS
{"kind":"sourced","source":"S1","quote":"Semiconductor Systems segment net sales decreased 22 percent in the third quarter of\\nfiscal 2019","text":"Semiconductor Systems net sales fell 22% year over year in fiscal Q3 2019."}
{"kind":"sourced","source":"S1","quote":"Backlog for the segment was $2.61 billion at\\nthe end of the quarter, compared to $3.05 billion a year earlier","text":"Segment backlog declined to $2.61bn from $3.05bn a year earlier."}
{"kind":"sourced","source":"S2","quote":"Management expects fourth quarter net sales to be approximately $3.68\\nbillion, plus or minus $150 million","text":"Guidance for Q4 net sales is $3.68bn +/- $150m."}
{"kind":"inference","text":"Backlog falling faster than revenue suggests the correction has further to run, since equipment backlog cushions reported revenue before it corrects."}

### SOURCES

[S1] MU 10-Q (2023-01-05) mdna
Revenue for the first quarter of 2023 decreased 39% as compared to the fourth quarter of
2022 primarily due to decreases of approximately 25% in average selling prices per bit
and decreases in bit shipments. Days of inventory increased to 214 days.

### CLAIMS
{"kind":"sourced","source":"S1","quote":"Revenue for the first quarter of 2023 decreased 39% as compared to the fourth quarter of\\n2022","text":"Revenue fell 39% sequentially in fiscal Q1 2023."}
{"kind":"sourced","source":"S1","quote":"decreases of approximately 25% in average selling prices per bit","text":"ASP per bit fell about 25% sequentially."}
{"kind":"sourced","source":"S1","quote":"Days of inventory increased to 214 days","text":"Days of inventory rose to 214."}
{"kind":"inference","text":"With most of the revenue decline coming from price rather than volume, this is a pricing correction rather than a demand collapse."}

### SOURCES

[S1] XSAS 10-Q (2024-05-02) mdna
Net revenue retention was 108% at the end of the first quarter, down from 118% a year
ago. Remaining performance obligations were $1.42 billion, up 14% year over year but
decelerating from 22% growth in the prior quarter.

### CLAIMS
{"kind":"sourced","source":"S1","quote":"Net revenue retention was 108% at the end of the first quarter, down from 118% a\\nyear ago","text":"Net revenue retention fell to 108% from 118% a year ago."}
{"kind":"sourced","source":"S1","quote":"Remaining performance obligations were $1.42 billion, up 14% year over year but\\ndecelerating from 22% growth in the prior quarter","text":"RPO grew 14% year over year, down from 22% growth last quarter."}
{"kind":"inference","text":"RPO growth decelerating well ahead of the retention drop suggests the slowdown is broadening from expansion revenue into new bookings, not just existing-customer downgrades."}
'''

# `XSAS` above is a fictional ticker, deliberately -- the other two examples are real
# tickers, and a third real one would risk being read as a verified fact rather than a
# demonstration of the JSON format and quoting discipline.

# The demonstrated inference sentences, extracted from the block above rather than
# duplicated as separate string literals, so this can never silently drift out of sync
# with `_FEWSHOT`. Exists for claims.py's own-demo-copy check: a base model shown three
# worked examples occasionally continues the PATTERN by reproducing one of these verbatim
# instead of reasoning about the passages actually in front of it -- measured live (a
# GOOGL extraction whose only inference line was a character-for-character copy of the
# MU/memory demo below, applied to a company with the opposite revenue trend from the one
# the demo describes). That is not evidence about GOOGL; it is the model echoing its own
# prompt, and claims.py drops it rather than presenting it as the model's reasoning.
FEWSHOT_INFERENCE_TEXTS: frozenset[str] = frozenset(
    json.loads(line)["text"] for line in re.findall(r'\{"kind":"inference".*?\}', _FEWSHOT))

_RULES = """You are extracting claims from company SEC filings.

Emit one JSON object per line and nothing else.

A "sourced" line needs: source (the bracketed id), quote, text.
  - `quote` must be copied CHARACTER FOR CHARACTER from that source, including line
    breaks (write them as \\n). If you cannot copy it exactly, do not make the claim.
  - `text` is your own one-sentence restatement of what the quote shows.
A "inference" line needs only: text. Use it for anything you concluded rather than read.

Do not write a sourced line for something you are inferring. Separating the two is the
entire purpose of this step -- a reader downstream cannot see these documents and has no
way to check a quote that was never in them.
Prefer specific numbers and guidance over boilerplate. Follow the metrics listed under
"Metrics that matter here" below -- they vary by company; a chip company's inventory
commentary and a software company's retention commentary are the same kind of thing.
"""


def segment_framing(seg: SubSegment) -> str:
    p = PROFILES[seg]
    return (
        f"SUB-SEGMENT: {p.name}\n"
        f"Cycle driver: {p.cycle_driver}\n"
        f"Timing: {p.lead_lag}\n"
        f"Metrics that matter here: {', '.join(p.key_metrics)}\n"
        f"Watch: {', '.join(p.watch_items)}"
    )


def render_sources(passages: list[tuple[str, str, str]]) -> str:
    """passages = [(sid, header, text)].

    `text` must be verbatim from the retained document -- it is what the quote is later
    resolved against, so any reformatting here shows up as a fabricated quote there.
    """
    return "\n\n".join(f"[{sid}] {header}\n{text}" for sid, header, text in passages)


def extraction_prompt(ticker: str, seg: SubSegment,
                      passages: list[tuple[str, str, str]]) -> str:
    """Few-shot completion prompt ending exactly at the point the model must continue."""
    return (
        f"{_RULES}\n"
        f"{_FEWSHOT}\n"
        f"### SOURCES\n\n"
        f"TICKER: {ticker}\n"
        f"{segment_framing(seg)}\n\n"
        f"{render_sources(passages)}\n\n"
        f"### CLAIMS\n"
    )
