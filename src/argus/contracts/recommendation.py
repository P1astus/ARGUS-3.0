"""Stage 5 output schema: the trade recommendation.

This is what the SFT pass shapes the model to produce, and what Stage 6 records. Making it
a validated schema rather than free text means a malformed generation is caught and
retried at inference, not discovered later in the journal.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from argus.contracts.quant import SubSegment


class ScenarioKind(StrEnum):
    BULL = "bull"
    BASE = "base"
    BEAR = "bear"


class Direction(StrEnum):
    LONG = "long"
    FLAT = "flat"          # an explicit "no trade" -- see Recommendation below
    SHORT = "short"


class Levels(BaseModel):
    """Entry, target, invalidation. All prices, all required together."""

    entry: float = Field(gt=0)
    target: float = Field(gt=0)
    invalidation: float = Field(gt=0, description="Thesis is wrong below/above this")

    @model_validator(mode="after")
    def _coherent(self) -> Levels:
        # Enforced because a writeup with target below entry on a long is a formatting
        # failure that reads as a real recommendation.
        if self.target == self.entry or self.invalidation == self.entry:
            raise ValueError("target and invalidation must differ from entry")
        return self

    def reward_risk(self, direction: Direction) -> float:
        reward = abs(self.target - self.entry)
        risk = abs(self.entry - self.invalidation)
        return reward / risk if risk > 0 else float("inf")


class Scenario(BaseModel):
    kind: ScenarioKind
    thesis: str
    probability: float = Field(ge=0.0, le=1.0)
    levels: Levels | None = None


class Recommendation(BaseModel):
    """The Stage 5 deliverable for one ticker on one date."""

    ticker: str
    sub_segment: SubSegment
    as_of: date
    direction: Direction
    conviction: float = Field(ge=0.0, le=1.0)

    scenarios: list[Scenario]
    chosen: ScenarioKind
    target_holding_days: int = Field(gt=0)

    quant_score: float | None = Field(
        default=None, description="Stage 1 score at recommendation time, carried for attribution")
    quant_percentile: float | None = Field(default=None, ge=0.0, le=1.0)

    summary: str

    @model_validator(mode="after")
    def _validate(self) -> Recommendation:
        kinds = {s.kind for s in self.scenarios}
        missing = {ScenarioKind.BULL, ScenarioKind.BASE, ScenarioKind.BEAR} - kinds
        if missing:
            raise ValueError(f"all three scenarios required; missing {sorted(missing)}")
        if self.chosen not in kinds:
            raise ValueError(f"chosen scenario {self.chosen} is not among the scenarios")

        total = sum(s.probability for s in self.scenarios)
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"scenario probabilities sum to {total:.3f}, expected ~1.0")

        # A tradeable recommendation must carry levels. FLAT deliberately need not --
        # "no trade" is a legitimate and useful output, and forcing invented levels on it
        # would train the model to always find a trade.
        if self.direction != Direction.FLAT:
            chosen = next(s for s in self.scenarios if s.kind == self.chosen)
            if chosen.levels is None:
                raise ValueError(
                    f"direction={self.direction} requires levels on the chosen scenario")
        return self

    @property
    def chosen_scenario(self) -> Scenario:
        return next(s for s in self.scenarios if s.kind == self.chosen)
