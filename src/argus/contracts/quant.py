"""Stage 1 output schema.

This is the boundary between the quant core and everything downstream. `engine/` and
`tools/` consume `QuantSignal` and must never import from `argus.quant` -- that rule is
what lets Stages 1, 5 and 6 be built and gated months apart and still assemble cleanly in
Phase 5.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class SubSegment(StrEnum):
    """Business-model sub-segment: what cycle logic applies to this ticker.

    Cycle dynamics differ sharply between these -- memory is famously capacity-driven and
    cyclical, equipment lags capex decisions, analog is comparatively stable. Stage 5
    prompts key off this so the model reasons about the right cycle, and the failure mode
    it exists to prevent is applying one segment's logic to a company that doesn't have
    that cycle -- inventory-cycle reasoning on a subscription business is the same class
    of mistake as memory-cycle reasoning on an analog name.

    Originally five semiconductor sub-segments (still the only ones with a validated
    quant universe, §5 of the handoff). The six TECH_* members extend the same mechanism
    to software/internet business models the user also trades -- deliberately still
    business-model categories, not an industry taxonomy, because that is the axis the
    extraction prompt actually needs.
    """

    EQUIPMENT = "equipment"
    FABLESS = "fabless"
    FOUNDRY = "foundry"
    MEMORY = "memory"
    ANALOG = "analog"

    TECH_AD_PLATFORM = "tech_ad_platform"        # GOOGL, META
    TECH_DEVICE_ECOSYSTEM = "tech_device_ecosystem"  # AAPL
    TECH_CLOUD_SAAS = "tech_cloud_saas"           # MSFT, ORCL, CRM, SNOW, ...
    TECH_CYBERSECURITY = "tech_cybersecurity"     # PANW, CRWD, ZS
    TECH_COMMERCE_PLATFORM = "tech_commerce_platform"  # AMZN, SHOP, PYPL, UBER
    TECH_STREAMING_MEDIA = "tech_streaming_media"  # NFLX


class QuantSignal(BaseModel):
    """One ticker's ranking output on one rebalance date."""

    ticker: str
    as_of: date
    score: float = Field(description="Model output; higher = better expected relative return")
    rank: int = Field(ge=1, description="1 = most attractive within the date's cross-section")
    percentile: float = Field(ge=0.0, le=1.0)
    sub_segment: SubSegment
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Cross-sectionally normalised feature values behind the score. "
                    "Carried through so Stage 5 can reason about *why*, and so the "
                    "journal can record what the model saw at entry.",
    )


class RankedUniverse(BaseModel):
    """The full cross-section for one rebalance date.

    Stage 5 receives this whole object, not a pre-filtered top-N: relative position within
    the cross-section is the signal, and a model shown only winners cannot judge whether
    the best available setup is actually good.
    """

    as_of: date
    signals: list[QuantSignal]
    benchmark: str = Field(description="Benchmark used for relative-return labelling")
    model_version: str
    n_universe: int = Field(description="Names that existed on this date")
    n_scored: int = Field(description="Names with complete data (see coverage caveat)")

    @model_validator(mode="after")
    def _check_coverage(self) -> RankedUniverse:
        if self.n_scored > self.n_universe:
            raise ValueError("n_scored cannot exceed n_universe")
        if len(self.signals) != self.n_scored:
            raise ValueError(
                f"signals ({len(self.signals)}) must match n_scored ({self.n_scored})")
        return self

    @property
    def coverage(self) -> float:
        """Fraction of the point-in-time universe actually scored.

        Surfaced deliberately: with yfinance there is no delisted history, so this number
        is the survivorship-bias disclosure. It belongs in every validation report.
        """
        return self.n_scored / max(self.n_universe, 1)

    def top(self, n: int) -> list[QuantSignal]:
        return sorted(self.signals, key=lambda s: s.rank)[:n]
