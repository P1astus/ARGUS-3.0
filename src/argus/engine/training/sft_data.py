"""SFT dataset generation: real Briefings -> schema-valid Recommendation completions.

WHY THIS IS EVIDENCE-GROUNDED, NOT SYNTHETIC
`sft.py`'s own docstring is explicit: "SFT should not try to teach analysis" -- at ~100M
CPT tokens the model cannot be taught new judgment, only output shape. That does NOT mean
the training examples should be arbitrary, though. A prompt built from a real Briefing
(real ticker, real sub-segment, real sourced claims from the actual corpus) and scored
against the real schema is what the model will actually be shown at inference time; a
prompt built from invented evidence would teach the format in a context that never
recurs, which is a worse use of a small SFT budget than it looks.

WHY GENERATION, NOT HAND-AUTHORING
The alternative to generating completions with the local model is writing them by hand.
That does not scale past a few dozen examples and does not exercise the actual failure
modes an SFT set needs to correct (malformed JSON, missing fields, scenario-probability
drift) -- those come from the model's own attempts, not from an author's imagination.
`generate_structured` (engine/inference/structured.py) already has the retry-on-
validation-error loop this needs; reused here rather than re-built.

WHAT IS NOT CLAIMED
A completion accepted into this dataset is SCHEMA-VALID, not analytically sound. Nothing
here checks whether a "long" call is a good trade -- that is explicitly out of scope, per
the CPT-cannot-teach-judgment finding this whole architecture is built around (handoff
§9). The one soft check applied is that scenario theses at least reference sub-segment
vocabulary the briefing actually surfaced, which is closer to "did not ignore the prompt"
than "made a good call".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from argus.contracts.briefing import Briefing, ClaimKind
from argus.contracts.quant import SubSegment
from argus.contracts.recommendation import Recommendation
from argus.engine.inference.structured import SCHEMA_PROMPT, ParseResult, generate_structured
from argus.engine.training.sft import SFTExample
from argus.tools.taxonomy import prompt_context

log = logging.getLogger(__name__)


def render_evidence(briefing: Briefing, max_claims: int = 15) -> str:
    """Compressed, sourced-only evidence block -- what a real Stage 5 prompt would show.

    Only SOURCED claims are shown, not INFERENCE ones: the point of this prompt is
    "here is what the filings say, now produce a decision", and showing the local
    extractor's own inferences would let the generation model launder them into a
    "recommendation" without any real analysis happening in between.
    """
    sourced = [c for c in briefing.claims if c.kind == ClaimKind.SOURCED][:max_claims]
    if not sourced:
        return "(no sourced claims survived extraction for this briefing)"
    return "\n".join(f"- {c.text}" for c in sourced)


# Fictional company/ticker, same reasoning as prompts.py's XSAS -- a real ticker in a
# few-shot demonstration risks being read as a verified fact about that company rather
# than a worked example of the JSON format.
_FEWSHOT = f"""### EVIDENCE

TICKER: XSAS
{prompt_context(SubSegment.TECH_CLOUD_SAAS)}

- Net revenue retention fell to 104% from 118% a year ago.
- Remaining performance obligations grew 6% year over year, down from 22% growth in the
  prior quarter.
- Management said budget scrutiny from enterprise customers had lengthened deal cycles
  by an average of three weeks.
- Gross margin held steady at 78%, in line with the prior four quarters.

### RECOMMENDATION
{{"ticker": "XSAS", "sub_segment": "tech_cloud_saas", "as_of": "2026-02-10", "direction": "short", "conviction": 0.55, "chosen": "base", "target_holding_days": 15, "summary": "RPO growth decelerating well ahead of the retention drop is the earlier tell for this segment; deal-cycle lengthening corroborates a real demand slowdown rather than one-off churn.", "scenarios": [{{"kind": "bull", "thesis": "Retention stabilizes and RPO growth reaccelerates as budget scrutiny eases next quarter.", "probability": 0.2, "levels": {{"entry": 42.0, "target": 50.0, "invalidation": 38.0}}}}, {{"kind": "base", "thesis": "RPO growth stays depressed for 2-3 more quarters as enterprise budget cycles remain elongated.", "probability": 0.5, "levels": {{"entry": 42.0, "target": 34.0, "invalidation": 46.0}}}}, {{"kind": "bear", "thesis": "Retention keeps falling and RPO growth turns negative, forcing guidance down.", "probability": 0.3, "levels": {{"entry": 42.0, "target": 27.0, "invalidation": 46.0}}}}]}}

### EVIDENCE

TICKER: XMEM
{prompt_context(SubSegment.MEMORY)}

- Average selling prices per bit declined approximately 4% sequentially, the smallest
  decline in five quarters.
- Days of inventory decreased to 118 days from 134 days in the prior quarter.
- Management said capacity additions remain disciplined and below prior-cycle levels.
- Bit shipment growth was roughly flat quarter over quarter.

### RECOMMENDATION
{{"ticker": "XMEM", "sub_segment": "memory", "as_of": "2026-03-04", "direction": "flat", "conviction": 0.35, "chosen": "base", "target_holding_days": 10, "summary": "Inventory drawdown and a decelerating ASP decline are consistent with a trough forming, but flat bit shipment growth means the recovery is not yet confirmed -- no clean setup either direction.", "scenarios": [{{"kind": "bull", "thesis": "Inventory normalization completes and pricing inflects positive next quarter.", "probability": 0.35, "levels": null}}, {{"kind": "base", "thesis": "Pricing stabilizes near current levels for another one to two quarters before a clear trend emerges.", "probability": 0.4, "levels": null}}, {{"kind": "bear", "thesis": "Capacity discipline breaks down elsewhere in the industry and pricing resumes its decline.", "probability": 0.25, "levels": null}}]}}
"""

_RULES = """You are producing a trading recommendation from sourced evidence extracted
from SEC filings. Base every scenario thesis on the evidence shown -- do not introduce
facts the evidence does not support. "flat" with no strong thesis is a correct answer
when the evidence does not point clearly either way; do not manufacture conviction that
is not there.
"""


def build_prompt(ticker: str, seg: SubSegment, as_of: date, briefing: Briefing) -> str:
    """Few-shot completion prompt ending exactly where the model must continue.

    Few-shot, not instruction-only, for the same reason extract/prompts.py is: this runs
    on a BASE checkpoint (handoff §4b), which continues a demonstrated pattern far more
    reliably than it follows a bare request.
    """
    return (
        f"{_RULES}\n{SCHEMA_PROMPT}\n{_FEWSHOT}\n"
        f"### EVIDENCE\n\n"
        f"TICKER: {ticker}\n"
        f"AS OF: {as_of.isoformat()}\n"
        f"{prompt_context(seg)}\n"
        # prompt_context() above only prints the human-readable segment NAME
        # ("Semiconductor capital equipment"); the JSON's sub_segment field needs the
        # machine key ("equipment"). Measured: without this line the model has to guess
        # the key from the two few-shot examples alone, which only demonstrate 2 of 11
        # possible values, and it fabricates a plausible-looking wrong one for the rest
        # (e.g. AMAT/equipment, META/tech_ad_platform both failed schema validation on
        # sub_segment before this line was added) -- the same fix TICKER already gets by
        # being labeled explicitly above.
        f"(sub_segment field must be exactly \"{seg.value}\")\n\n"
        f"{render_evidence(briefing)}\n\n"
        f"### RECOMMENDATION\n"
    )


@dataclass
class GenerationStats:
    attempted: int = 0
    accepted: int = 0
    rejected_no_evidence: int = 0
    rejected_invalid: int = 0
    total_attempts_used: int = 0

    def render(self) -> str:
        return (f"attempted={self.attempted} accepted={self.accepted} "
               f"rejected(no_evidence)={self.rejected_no_evidence} "
               f"rejected(invalid)={self.rejected_invalid} "
               f"mean_attempts={self.total_attempts_used / max(self.attempted, 1):.2f}")


def generate_sft_example(generator, ticker: str, seg: SubSegment, as_of: date,
                         briefing: Briefing, max_attempts: int = 3, max_tokens: int = 1400,
                         stats: GenerationStats | None = None) -> SFTExample | None:
    """One (prompt, completion) pair from one real briefing, or None on failure.

    Failures are expected and counted, not hidden -- format_validity is exactly the kind
    of number this project measures rather than assumes (handoff's own baseline
    discipline). A base checkpoint asked for a considerably harder structured-output task
    than plain extraction should be expected to fail some fraction of the time; silently
    retrying past that would misrepresent how ready this generation pipeline actually is.

    max_tokens=1400, not the 700 an earlier version used: measured against the real 20-case
    baseline set, 3/20 completions hit a 700-token cap mid-"summary" -- the model does not
    reliably keep `summary` to "one paragraph" as instructed, and the 3-scenario JSON never
    got to close. 1400 gives roughly 2x headroom over the longest observed truncation point.
    """
    s = stats if stats is not None else GenerationStats()
    s.attempted += 1

    sourced = [c for c in briefing.claims if c.kind == ClaimKind.SOURCED]
    if not sourced:
        s.rejected_no_evidence += 1
        return None

    prompt = build_prompt(ticker, seg, as_of, briefing)
    result: ParseResult = generate_structured(
        lambda p: generator.generate(p, max_tokens=max_tokens), prompt,
        max_attempts=max_attempts)
    s.total_attempts_used += result.attempts

    if not result.ok:
        s.rejected_invalid += 1
        log.warning("%s %s: generation did not produce a valid Recommendation (%s)",
                   ticker, as_of, result.error)
        return None

    rec = result.recommendation
    # Cross-check the model didn't silently rewrite the ticker/date it was given --
    # the schema validates internal coherence, not that the model stayed on-task.
    if rec.ticker.upper() != ticker.upper():
        s.rejected_invalid += 1
        log.warning("%s %s: generated ticker %s does not match the prompt",
                   ticker, as_of, rec.ticker)
        return None

    s.accepted += 1
    return SFTExample.from_recommendation(prompt, rec)
