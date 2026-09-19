"""Lookahead Propensity (LAP) -- a per-setup contamination score.

From "Detecting Lookahead Bias in LLM Forecasts" (arXiv 2512.23847), flagged in
`docs/literature_review.md` §2 as the single most actionable finding in the review: LAP
is a date-only recall query for a (ticker, as_of) pair that estimates the probability the
model has already internalised the realised outcome from pretraining, rather than
reasoning about it. The paper's headline result -- LAP is materially positive in-sample
and collapses to ~zero right after the training cutoff, and forecast "skill" concentrates
almost entirely on high-LAP pairs -- is the concrete mechanism behind KTD-Fin's finding
(handoff §4e) that properly-anonymised agents post negative selection alpha.

WHY THIS IS A SCORE, NOT A GATE CRITERION
The literature review's own recommendation is to "report results conditioned on it rather
than merely asserting our post-cutoff window is clean" -- i.e. a diagnostic every setup
carries, not a pass/fail threshold. A high-LAP setup is not disqualified; a RESULT that
depends on high-LAP setups to look good is the thing to distrust. `gate.py` treats it the
same way it already treats the quant-skill guard: reported, and used to flag when a result
is uninformative, not folded silently into a single pass/fail number.

THE PROBE DESIGN
The probe asks the model to recall the realised outcome for a (ticker, as_of, horizon)
triple using ONLY the ticker and date -- no retrieved passages, no briefing, nothing this
architecture would show a real recommendation. If the model answers correctly at a rate
well above chance, that is evidence the base model's own pretraining already contains the
answer for that specific setup, which is exactly the contamination path this project's
entire retrieval-first design exists to route around (handoff §1: "the local model is not
allowed to remember what a filing said -- it is shown the passage and asked to copy from
it"). LAP asks the same question of the market outcome itself.

WHAT THIS FILE DOES NOT CLAIM
No trained CPT model exists yet (handoff §8), so nothing here has been run against real
setups -- there is no LAP number to report. This is readiness: the scoring machinery
tested and ready so that when the pilot CPT run produces recommendations, conditioning on
contamination is a function call, not a research project started from zero under time
pressure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from argus.engine.inference.local import Generator

# Forced-choice, not open-ended: an open "what happened to TICKER" invites the model to
# hedge or refuse, which would be scored as "no recall" and understate true LAP. Forcing
# a direction means a model with no real signal should split ~50/50, which is exactly the
# comparison point the probe needs.
_PROMPT = """You are recalling market history from memory. Do not explain your reasoning.

Between {as_of} and {horizon_days} trading days later, did {ticker} stock go UP or DOWN
relative to a broad market/sector benchmark?

Answer with exactly one word: UP or DOWN."""

_ANSWER = re.compile(r"\b(UP|DOWN)\b", re.I)


@dataclass(frozen=True)
class LAPCase:
    """One (ticker, as_of, horizon) probe, with the outcome it will be checked against.

    `realized_up` is the ground truth -- computed the same way the quant labels are
    (handoff §5: equal-weight leave-one-out relative return), supplied by the caller
    rather than looked up here, so this module has no data-fetching dependency of its own.
    """

    ticker: str
    as_of: date
    horizon_days: int
    realized_up: bool


@dataclass
class LAPResult:
    case: LAPCase
    raw_response: str
    predicted_up: bool | None   # None if the model's answer didn't parse
    correct: bool | None        # None when predicted_up is None -- not a wrong guess

    @property
    def parsed(self) -> bool:
        return self.predicted_up is not None


def probe_prompt(case: LAPCase) -> str:
    return _PROMPT.format(as_of=case.as_of.isoformat(), horizon_days=case.horizon_days,
                          ticker=case.ticker)


def run_probe(generator: Generator, case: LAPCase, max_tokens: int = 10) -> LAPResult:
    """Ask the model to recall one setup's outcome from memory alone."""
    raw = generator.generate(probe_prompt(case), max_tokens=max_tokens)
    m = _ANSWER.search(raw)
    if m is None:
        return LAPResult(case, raw, predicted_up=None, correct=None)
    predicted_up = m.group(1).upper() == "UP"
    return LAPResult(case, raw, predicted_up, predicted_up == case.realized_up)


def run_probes(generator: Generator, cases: list[LAPCase]) -> list[LAPResult]:
    return [run_probe(generator, c) for c in cases]


@dataclass
class LAPScore:
    """Aggregate LAP over a set of probed setups."""

    n: int
    n_parsed: int
    recall_accuracy: float | None   # None if nothing parsed
    excess_over_chance: float | None  # recall_accuracy - 0.5; the contamination signal
    contaminated_keys: frozenset[tuple[str, str]] = field(default_factory=frozenset)


def score(results: list[LAPResult], contamination_threshold: float = 0.65) -> LAPScore:
    """Aggregate probe results into a single LAP reading.

    `contamination_threshold`: a setup's own probe is binary (right/wrong), so
    per-setup contamination is better read as "was this one of the setups where the
    model's recall-only guess matched" -- flagged individually below, not from this
    aggregate threshold, which instead answers the population-level question the
    literature review asks: is recall accuracy on this sample distinguishable from
    chance at all.
    """
    parsed = [r for r in results if r.parsed]
    n = len(results)
    n_parsed = len(parsed)
    if not parsed:
        return LAPScore(n, 0, None, None)

    acc = sum(1 for r in parsed if r.correct) / n_parsed
    contaminated = frozenset(
        (r.case.ticker, r.case.as_of.isoformat()) for r in parsed if r.correct)
    return LAPScore(n, n_parsed, acc, acc - 0.5, contaminated)


def split_by_lap(setups: list, lap: LAPScore,
                 key_fn=lambda s: (s.ticker, s.as_of.isoformat())) -> tuple[list, list]:
    """Partition arbitrary setup objects (e.g. journal trades, Recommendations) into
    (high-LAP, low-LAP) using the contaminated-keys set from `score()`.

    This is the shape `gate.py` needs: report the arm's accuracy on each half
    separately. A result that only holds on the high-LAP half is the exact failure
    mode the literature review found in 9 of 10 published agents once memorisation
    was controlled for (handoff §4e) -- reporting it splits, rather than averaging it
    away in one aggregate number.
    """
    high, low = [], []
    for s in setups:
        (high if key_fn(s) in lap.contaminated_keys else low).append(s)
    return high, low
