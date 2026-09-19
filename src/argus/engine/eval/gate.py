"""Phase 3 gate: does the trained engine's reasoning add value?

A DESIGN CHANGE FORCED BY THE PHASE 1 RESULT
The gate was originally specified as "does the trained model add value over the raw quant
score alone". That framing assumed Stage 1 works. It does not, yet: measured out-of-sample
IC excluding the survivorship-contaminated fold was -0.0007, i.e. indistinguishable from
zero, and the model did not beat a plain 12-1 momentum sort.

Against a zero-skill baseline, "beats the quant score" is a trivial bar that a coin flip
clears half the time. Left unchanged, a broken Stage 1 would quietly make the Stage 5 gate
meaningless -- and it would LOOK like a pass.

So the gate is restructured around an ABSOLUTE standard plus a relative one:

  ABSOLUTE   the engine must beat a no-skill baseline by a pre-registered margin, with a
             confidence interval that excludes zero. This is the criterion that has teeth
             when the quant score is weak.
  RELATIVE   the engine must beat Arm B (same base model, prompt and tools, no CPT).
             Without this arm the measurement is of prompt engineering, not training.
  GUARD      if Stage 1 has no skill, beating it is reported as UNINFORMATIVE rather than
             as a pass.

All thresholds are pre-registered: at multi-day cost per run, deciding what counts as
success after seeing results is not a gate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from argus.contracts.provenance import Arm
from argus.engine.eval.lap import LAPScore

log = logging.getLogger(__name__)


@dataclass
class GateThresholds:
    """Pre-registered. Fix before running, never after."""

    min_directional_accuracy: float = 0.55      # vs 0.50 no-skill
    min_accuracy_ci_lower: float = 0.50         # CI must exclude coin-flip
    min_margin_over_base_prompt: float = 0.03   # Arm C - Arm B, absolute
    min_format_validity: float = 0.95           # parses to a valid Recommendation
    max_calibration_error: float = 0.15         # |confidence - realised| , binned
    min_setups: int = 150                       # below this the CI swallows any effect
    # A quant arm below this |accuracy - 0.5| is treated as no-skill, making comparisons
    # against it uninformative rather than favourable.
    quant_skill_epsilon: float = 0.02


@dataclass
class ArmResult:
    arm: str
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    mean_rel_return: float | None = None
    format_validity: float = 1.0
    calibration_error: float | None = None
    detail: dict = field(default_factory=dict)
    # Optional: LAP over this arm's setups (engine/eval/lap.py). None means "not probed",
    # not "clean" -- an absent probe is a missing measurement, and `decide()` treats it
    # that way rather than silently assuming zero contamination.
    lap: LAPScore | None = None
    accuracy_low_lap: float | None = None   # accuracy restricted to LAP-uncontaminated setups


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over the normal approximation because at n~150 with p near 0.5 the normal
    interval is noticeably wrong, and the whole point here is whether the interval
    excludes 0.5.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_error(confidences: list[float], outcomes: list[bool],
                      n_bins: int = 5) -> float:
    """Expected calibration error.

    A model that says 80% and is right 80% of the time is far more useful for position
    sizing than one that is merely accurate, which is why this is a gate criterion rather
    than a diagnostic.
    """
    if not confidences:
        return float("nan")
    total, n = 0.0, len(confidences)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi or (b == n_bins - 1 and c == 1.0))]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_acc = sum(1 for i in idx if outcomes[i]) / len(idx)
        total += (len(idx) / n) * abs(avg_conf - avg_acc)
    return total


@dataclass
class GateDecision:
    passed: bool
    criteria: dict[str, bool]
    warnings: list[str]
    arms: dict[str, ArmResult]
    thresholds: GateThresholds

    def render(self) -> str:
        out = ["=" * 72, "PHASE 3 GATE: " + ("PASS" if self.passed else "FAIL"), "=" * 72, ""]
        out.append(f"{'arm':16s} {'n':>6} {'accuracy':>10} {'95% CI':>18} {'format':>8}")
        out.append("-" * 66)
        for name, r in self.arms.items():
            ci = f"[{r.accuracy_ci[0]:.3f}, {r.accuracy_ci[1]:.3f}]"
            out.append(f"{name:16s} {r.n:>6} {r.accuracy:>10.3f} {ci:>18} "
                       f"{r.format_validity:>8.1%}")
        out.append("")
        for name, ok in self.criteria.items():
            out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if self.warnings:
            out.append("")
            for w in self.warnings:
                out.append(f"  WARNING: {w}")
        return "\n".join(out)


def decide(arms: dict[str, ArmResult],
           thresholds: GateThresholds = GateThresholds()) -> GateDecision:
    """Apply the pre-registered gate to measured arm results."""
    t = thresholds
    warnings: list[str] = []

    cpt = arms.get(str(Arm.CPT_SFT))
    base = arms.get(str(Arm.BASE_PROMPT))
    quant = arms.get(str(Arm.QUANT_ONLY))

    if cpt is None:
        return GateDecision(False, {"CPT arm present": False},
                            ["no CPT_SFT arm was evaluated"], arms, t)

    criteria: dict[str, bool] = {}

    criteria[f"n >= {t.min_setups} setups"] = cpt.n >= t.min_setups
    if cpt.n < t.min_setups:
        warnings.append(
            f"only {cpt.n} setups; below ~{t.min_setups} the confidence interval is wider "
            "than any plausible effect, so a pass here would not be meaningful")

    criteria[f"accuracy >= {t.min_directional_accuracy}"] = (
        cpt.accuracy >= t.min_directional_accuracy)

    # The absolute criterion -- the one that still has teeth when Stage 1 is weak.
    criteria["accuracy CI excludes no-skill (0.50)"] = (
        cpt.accuracy_ci[0] > t.min_accuracy_ci_lower)

    criteria[f"format validity >= {t.min_format_validity:.0%}"] = (
        cpt.format_validity >= t.min_format_validity)

    if cpt.calibration_error is not None:
        criteria[f"calibration error <= {t.max_calibration_error}"] = (
            cpt.calibration_error <= t.max_calibration_error)

    # Relative criterion: without Arm B this measures the prompt, not the training.
    if base is not None:
        margin = cpt.accuracy - base.accuracy
        criteria[f"beats base+prompt by >= {t.min_margin_over_base_prompt}"] = (
            margin >= t.min_margin_over_base_prompt)
        if 0 < margin < t.min_margin_over_base_prompt:
            warnings.append(
                f"CPT beats base+prompt by only {margin:+.3f}; the pre-registered margin "
                f"is {t.min_margin_over_base_prompt}. A small positive gap at this sample "
                "size is not distinguishable from noise")
    else:
        criteria["base+prompt arm present"] = False
        warnings.append(
            "no BASE_PROMPT arm: without it, any result measures prompt engineering "
            "rather than the effect of continued pretraining")

    # The guard. A comparison against a no-skill baseline is uninformative, and must not
    # be allowed to read as a pass.
    if quant is not None:
        quant_skill = abs(quant.accuracy - 0.5)
        if quant_skill < t.quant_skill_epsilon:
            warnings.append(
                f"quant arm accuracy {quant.accuracy:.3f} is within "
                f"{t.quant_skill_epsilon} of chance -- Stage 1 has no measurable skill on "
                "this sample, so 'beats the quant score' is UNINFORMATIVE and is excluded "
                "from the gate. This matches the Phase 1 finding (IC ~= 0 once the "
                "survivorship-contaminated fold is removed)")
        else:
            criteria["beats quant-only arm"] = cpt.accuracy > quant.accuracy

    # LAP conditioning (docs/literature_review.md §2): a result driven by high-LAP setups
    # is exactly the pattern behind KTD-Fin's finding that 9/10 published agents post
    # negative selection alpha once memorisation is controlled for. Reported as a warning,
    # not folded into `criteria`, because the literature review's own recommendation is
    # to condition results on it rather than gate on it -- a single LAP threshold would
    # just move the p-hacking surface, which is the same trap §4d's pre-registration
    # discipline exists to close off.
    if cpt.lap is None:
        warnings.append(
            "no LAP probe was run for the CPT_SFT arm -- contamination is UNMEASURED, "
            "not absent. Run engine.eval.lap against this arm's setups before trusting "
            "a pass at face value")
    elif cpt.lap.excess_over_chance is not None and cpt.lap.excess_over_chance > 0.15:
        detail = (f"excess {cpt.lap.excess_over_chance:+.2f} over chance recall on "
                  f"{cpt.lap.n_parsed}/{cpt.lap.n} probed setups")
        if cpt.accuracy_low_lap is not None:
            warnings.append(
                f"LAP recall is elevated ({detail}). Accuracy restricted to "
                f"LAP-uncontaminated setups is {cpt.accuracy_low_lap:.3f} vs "
                f"{cpt.accuracy:.3f} pooled -- trust the restricted number over the "
                "pooled one")
        else:
            warnings.append(
                f"LAP recall is elevated ({detail}), and accuracy was not re-measured on "
                "the LAP-uncontaminated subset -- the pooled accuracy above may be "
                "substantially inflated by setups the model already knew the answer to")

    # Standing caveat, always emitted: the base model's own pretraining covers the eval
    # window, so even a post-cutoff eval leaks.
    warnings.append(
        "retrospective eval remains partly contaminated by the base model's pretraining; "
        "this gate can fail a run out, but cannot fully clear it. Prospective journal "
        "evidence is the uncontaminated signal")

    return GateDecision(all(criteria.values()), criteria, warnings, arms, t)


def save(decision: GateDecision, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "passed": decision.passed,
        "criteria": decision.criteria,
        "warnings": decision.warnings,
        "thresholds": asdict(decision.thresholds),
        "arms": {k: asdict(v) for k, v in decision.arms.items()},
    }, indent=2, default=str))
