"""Structured evidence appendices for a Briefing: price, insider activity, macro context.

WHY THESE DO NOT GO THROUGH THE Claim/SOURCED MECHANISM
`briefing.py`'s Claim schema exists to police PROSE: a model summarising or paraphrasing
free text can quietly drift or fabricate, so SOURCED claims require a verbatim quote to
audit against. Numbers pulled directly from a provider API carry no such risk -- there is
nothing to paraphrase, and forcing "close: $131.24" through a quote-verification step would
add ceremony without adding safety. These three types are therefore displayed directly
rather than run through extraction.

WHY THEY ARE SEPARATE FROM Source/SourceType
`SourceType.PRICE_DATA` already exists in briefing.py but was never wired to anything. It
stays there for provenance labelling (which of a Briefing's sources are price vs filing vs
news), while the actual payload shape lives here since it is structured data, not retrieved
prose text with a byte-identical retained copy.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PriceSnapshot(BaseModel):
    """Price context as of a decision date. Never includes a bar dated after `as_of` --
    the same leakage boundary every other evidence source in this project enforces."""

    ticker: str
    as_of: date
    close: float
    volume: float

    return_20d: float | None = Field(default=None, description="pct return, ~1 month")
    return_90d: float | None = Field(default=None, description="pct return, ~1 quarter")
    return_252d: float | None = Field(default=None, description="pct return, ~1 year")

    avg_volume_20d: float | None = None
    high_252d: float | None = None
    low_252d: float | None = None

    bars_used: int = Field(description="how many trailing bars were available; low counts "
                           "(new listing, thin history) mean the longer-window returns are "
                           "unreliable and should be read with that in mind")


class InsiderTransactionCode(BaseModel):
    """One row from a Form 4 (or 3/5) filing's non-derivative transaction table."""

    filing_date: date
    transaction_date: date
    owner_name: str
    is_officer: bool
    is_director: bool
    is_ten_pct_owner: bool
    officer_title: str | None = None

    code: str = Field(description="SEC transaction code, e.g. P=open-market purchase, "
                      "S=open-market sale, A=award/grant, M=option exercise, F=tax "
                      "withholding, G=gift, C=conversion, D=disposition to issuer")
    is_open_market: bool = Field(description="True for P/S only -- the codes that reflect "
                                 "a discretionary decision to buy or sell, as opposed to "
                                 "routine compensation mechanics (grants, tax withholding, "
                                 "option exercises). Conflating the two overstates how much "
                                 "'insider activity' data actually says about conviction.")
    shares: float
    price_per_share: float | None = None
    shares_owned_after: float | None = None
    acquired: bool = Field(description="True if shares were acquired, False if disposed")


class InsiderActivity(BaseModel):
    """Form 3/4/5 filings for one issuer, up to an as_of date."""

    ticker: str
    as_of: date
    transactions: list[InsiderTransactionCode]

    @property
    def open_market_buys(self) -> int:
        return sum(1 for t in self.transactions if t.is_open_market and t.acquired)

    @property
    def open_market_sells(self) -> int:
        return sum(1 for t in self.transactions if t.is_open_market and not t.acquired)


class MacroSeries(BaseModel):
    """One FRED series' most recent point as of a decision date."""

    series_id: str
    label: str
    value: float
    value_date: date
    revision_safe: bool = Field(description="False for series subject to later revision "
                                "(CPI, employment, industrial production benchmark/"
                                "seasonal updates) -- the value shown is the CURRENT "
                                "vintage, not necessarily what was known on `as_of`. True "
                                "only for series like policy rates that are final at "
                                "publication. This is a real leakage risk for non-"
                                "revision-safe series and is not hidden.")


class MacroContext(BaseModel):
    as_of: date
    series: list[MacroSeries]


class AnalystRating(BaseModel):
    """One firm's most recent rating/target action on or before `as_of`.

    Deliberately NOT "this firm's target as of today" -- that field exists in yfinance's
    API too (Ticker.info's targetMeanPrice) but carries no date, so there is no way to
    tell whether it reflects today or six months ago. `tools/analyst.py` reconstructs
    this from yfinance's dated upgrades_downgrades history instead, specifically so a
    historical `as_of` briefing shows what was known then, not today's live consensus.
    """

    firm: str
    grade_date: date
    to_grade: str
    action: str
    price_target: float | None = None


class AnalystConsensus(BaseModel):
    ticker: str
    as_of: date
    ratings: list[AnalystRating] = Field(description="most recent action per firm, "
                                         "on or before as_of -- not a full history")

    @property
    def targets(self) -> list[float]:
        return [r.price_target for r in self.ratings if r.price_target is not None]

    @property
    def mean_target(self) -> float | None:
        t = self.targets
        return sum(t) / len(t) if t else None

    @property
    def high_target(self) -> float | None:
        t = self.targets
        return max(t) if t else None

    @property
    def low_target(self) -> float | None:
        t = self.targets
        return min(t) if t else None
