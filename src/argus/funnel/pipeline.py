"""Phase 5: six-stage funnel assembly.

Orchestration only. Every stage is reached through `argus.contracts`, never through a
sibling's internals -- that boundary is what let Stages 1, 5 and 6 be built and gated
independently, and it is why assembly is wiring rather than integration surgery.

If real logic starts accumulating in this file, a stage boundary was drawn wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from argus.contracts.briefing import Briefing
from argus.contracts.provenance import Arm, RunProvenance
from argus.contracts.quant import RankedUniverse, QuantSignal
from argus.contracts.recommendation import Direction, Recommendation

log = logging.getLogger(__name__)


@dataclass
class FunnelConfig:
    as_of: date
    top_n: int = 10                 # candidates passed from Stage 1 to Stages 2-5
    arm: Arm = Arm.CPT_SFT
    enable_tools: bool = True       # Stages 2-4
    record_to_journal: bool = True  # Stage 6
    paper: bool = True


@dataclass
class FunnelResult:
    as_of: date
    ranked: RankedUniverse | None = None
    briefings: dict[str, Briefing] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    journal_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Funnel:
    """Wires Stage 1 -> Stages 2-4 -> Stage 5 -> Stage 6.

    Components are injected rather than constructed here, so the funnel can be exercised
    with any subset present -- which is how it stays testable while Stage 5 is untrained.
    """

    def __init__(self, ranker=None, briefer=None, engine=None, journal=None,
                 provenance: RunProvenance | None = None) -> None:
        self.ranker = ranker          # Stage 1
        self.briefer = briefer        # Stages 2-4
        self.engine = engine          # Stage 5
        self.journal = journal        # Stage 6
        self.provenance = provenance or RunProvenance(arm=Arm.QUANT_ONLY)

    def run(self, config: FunnelConfig) -> FunnelResult:
        result = FunnelResult(as_of=config.as_of)

        # ---- Stage 1: quantitative ranking ----
        if self.ranker is None:
            result.warnings.append("no Stage 1 ranker; funnel produces no candidates")
            return result

        result.ranked = self.ranker.rank(config.as_of)
        candidates = result.ranked.top(config.top_n)

        # Surfaced rather than assumed: the Phase 1 gate has not been cleared, so the
        # quant score is a weak prior, not a validated signal. Anything consuming this
        # result should know that.
        if result.ranked.coverage < 0.8:
            result.warnings.append(
                f"universe coverage {result.ranked.coverage:.1%} -- ranking is computed "
                "on an incomplete cross-section (missing delisted names)")

        log.info("stage 1: %d candidates from %d scored (coverage %.1%%)",
                 len(candidates), result.ranked.n_scored, result.ranked.coverage * 100)

        # ---- Stages 2-4: tool-sourced briefings ----
        if config.enable_tools and self.briefer is not None:
            for sig in candidates:
                try:
                    result.briefings[sig.ticker] = self.briefer.build(
                        sig.ticker, sig.sub_segment, config.as_of)
                except Exception as e:
                    result.warnings.append(
                        f"briefing failed for {sig.ticker}: {type(e).__name__}")
        elif config.enable_tools:
            result.warnings.append("tools enabled but no briefer wired")

        # ---- Stage 5: reasoning ----
        if self.engine is None:
            result.warnings.append("no Stage 5 engine; stopping after briefings")
            return result

        for sig in candidates:
            try:
                rec = self.engine.recommend(
                    signal=sig,
                    briefing=result.briefings.get(sig.ticker),
                    ranked=result.ranked,
                )
                result.recommendations.append(rec)
            except Exception as e:
                result.warnings.append(
                    f"recommendation failed for {sig.ticker}: {type(e).__name__}: {e}")

        # ---- Stage 6: journal ----
        if config.record_to_journal and self.journal is not None:
            for rec in result.recommendations:
                b = result.briefings.get(rec.ticker)
                rid = self.journal.record_recommendation(
                    rec, self.provenance,
                    briefing_json=b.model_dump_json() if b else None,
                    sourced_fraction=b.sourced_fraction if b else None,
                )
                result.journal_ids.append(rid)

                # FLAT recommendations are recorded but not opened. Keeping the
                # non-trades is what makes calibration measurable -- a model judged only
                # on the trades it chose to take looks better than it is.
                if rec.direction != Direction.FLAT and config.paper:
                    entry = self._entry_price(rec, result.ranked)
                    if entry:
                        self.journal.open_trade(rec, self.provenance, entry,
                                                is_paper=config.paper)

        return result

    @staticmethod
    def _entry_price(rec: Recommendation, ranked: RankedUniverse | None) -> float | None:
        s = rec.chosen_scenario
        return s.levels.entry if s.levels else None
