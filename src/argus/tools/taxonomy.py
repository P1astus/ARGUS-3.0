"""Sub-segment cycle context.

Stage 5 prompts key off this so the model reasons about the right cycle dynamics. The five
sub-segments behave differently enough that applying memory logic to an analog name, or
equipment logic to a fabless one, produces confident and wrong analysis.

This is domain framing, not data -- it shapes what the model is asked to consider, and gives
the CPT'd model vocabulary anchors it should have internalised from the filings corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.contracts.quant import SubSegment


@dataclass(frozen=True)
class SegmentProfile:
    name: str
    cycle_driver: str
    lead_lag: str
    key_metrics: tuple[str, ...]
    inventory_dynamics: str
    typical_cycle_months: tuple[int, int]
    watch_items: tuple[str, ...]


PROFILES: dict[SubSegment, SegmentProfile] = {
    SubSegment.EQUIPMENT: SegmentProfile(
        name="Semiconductor capital equipment",
        cycle_driver="Customer capex budgets and fab construction schedules",
        lead_lag="LAGS chip demand by 2-4 quarters; orders are committed well ahead of "
                 "the demand they serve, so equipment can be strong into a chip downturn",
        key_metrics=("bookings", "book-to-bill", "backlog", "WFE spend",
                     "service/installed-base revenue"),
        inventory_dynamics="Backlog cushions near-term revenue, which delays and then "
                           "amplifies the eventual correction",
        typical_cycle_months=(24, 42),
        watch_items=("customer capex guidance", "China export-control exposure",
                     "fab construction announcements", "service revenue share as a floor"),
    ),
    SubSegment.FABLESS: SegmentProfile(
        name="Fabless chip designers",
        cycle_driver="End-market demand and design-win cycles",
        lead_lag="COINCIDENT to slightly leading; channel inventory distorts the signal",
        key_metrics=("channel inventory weeks", "design wins", "ASP trend",
                     "gross margin", "foundry allocation"),
        inventory_dynamics="Sell-in vs sell-through divergence is the key tell -- revenue "
                           "can hold up while distributors absorb inventory, then falls "
                           "abruptly when they stop",
        typical_cycle_months=(18, 30),
        watch_items=("distributor inventory commentary", "foundry wafer pricing",
                     "end-market mix", "hyperscaler capex for AI-exposed names"),
    ),
    SubSegment.FOUNDRY: SegmentProfile(
        name="Foundries",
        cycle_driver="Utilisation rates and leading-edge node transitions",
        lead_lag="COINCIDENT; utilisation is close to a real-time demand read",
        key_metrics=("utilisation %", "node mix", "capex intensity", "wafer ASP"),
        inventory_dynamics="Utilisation below ~80% compresses margins sharply because the "
                           "cost base is overwhelmingly fixed",
        typical_cycle_months=(18, 36),
        watch_items=("monthly revenue disclosures", "advanced-node capacity additions",
                     "customer concentration", "geopolitical/location risk"),
    ),
    SubSegment.MEMORY: SegmentProfile(
        name="Memory (DRAM/NAND)",
        cycle_driver="Commodity supply/demand balance; near-pure price cycle",
        lead_lag="LEADS the broader semi cycle -- memory pricing typically inflects first",
        # MEASURED against the corpus, not assumed. "days of inventory" was listed here
        # and appears in ZERO chunks across MU/WDC/STX -- it is analog/MCU vocabulary
        # (336 chunks there, the highest of any segment). Memory issuers write about
        # inventory as write-downs and carrying value: 315 chunks. Since these strings
        # generate the retrieval queries in `extract.loop.segment_queries`, a metric the
        # issuers never use sends the extraction loop hunting for language that is not
        # there. Counts across 21,303 memory chunks: ASP 983, capex 1,555, write-downs
        # 315, bit growth 96 (memory-exclusive: 3/0/0 in the other segments),
        # contract/spot pricing 43.
        key_metrics=("contract/spot pricing", "bit growth", "inventory write-downs",
                     "average selling price per bit", "capex discipline"),
        inventory_dynamics="The most violent cycle in semis. Pricing can fall 40-60% "
                           "peak-to-trough; supplier capex discipline is the main "
                           "determinant of cycle length",
        typical_cycle_months=(12, 24),
        watch_items=("DRAM/NAND contract pricing", "supplier capex cuts",
                     "hyperscaler order patterns", "days-of-inventory trend"),
    ),
    SubSegment.ANALOG: SegmentProfile(
        name="Analog and mixed-signal",
        cycle_driver="Broad industrial and automotive demand",
        lead_lag="LAGS; long product lifecycles and sticky sockets smooth the cycle",
        # Corpus counts across 58,110 analog chunks: ASP 2,237, design wins 1,110,
        # backlog 1,074, utilisation 1,288, lead times 691, book-to-bill 523,
        # distributor inventory 398, days of inventory 336. Note book-to-bill is NOT
        # equipment-exclusive -- analog and MCU issuers report it nearly twice as often
        # as the equipment names do (523 vs 286), because their distributors do.
        key_metrics=("distributor inventory", "days of inventory", "book-to-bill",
                     "auto/industrial mix", "internal fab utilisation", "lead times"),
        inventory_dynamics="Slowest-moving of the five. Corrections are shallower but "
                           "much longer, and recoveries are correspondingly gradual",
        typical_cycle_months=(24, 48),
        watch_items=("auto production forecasts", "industrial PMI",
                     "distributor restocking commentary", "pricing discipline"),
    ),
}


# ---------------------------------------------------------------------------------
# TECH_* profiles: informed priors, NOT YET corpus-measured.
#
# The five semiconductor profiles above went through a correction pass once retrieval
# was measured against the real corpus -- MEMORY's "days of inventory" turned out to
# appear in ZERO memory-issuer chunks (it's analog vocabulary). These six are written
# from general knowledge of each business model, before any tech filing has been
# indexed. Treat every key_metric string here as a hypothesis about vocabulary, not a
# fact -- the same correction pass is expected once `docs/retrieval_and_extraction.md`
# has tech-corpus counts to check them against. Do not skip that step.
PROFILES_TECH: dict[SubSegment, SegmentProfile] = {
    SubSegment.TECH_AD_PLATFORM: SegmentProfile(
        name="Advertising platforms",
        cycle_driver="Advertiser budgets, which track the broader economy, plus "
                     "engagement/reach for the audience side",
        lead_lag="COINCIDENT to slightly LEADING; ad budgets are among the first "
                 "corporate costs cut in a slowdown and among the first restored",
        # MEASURED across 9,083 GOOGL/META chunks: "advertising revenue" 1,077,
        # "platform" 897 (too generic to anchor on), monthly active users 247 --
        # narrowly ahead of daily active users at 197 -- ARPU 223. CPM, the metric
        # originally listed here, appears only 5 times: it is earnings-call and
        # investor-deck language, almost never disclosed with a figure in the filing
        # itself. Swapped for MAU, which has 236 gold (concept + digit) chunks.
        key_metrics=("ad revenue growth", "monthly active users", "daily active users",
                     "ARPU", "AI infrastructure capex"),
        inventory_dynamics="No physical inventory. Impression VOLUME can hold up while "
                           "PRICE (CPM) softens -- price is the earlier tell, same "
                           "sell-in/sell-through logic as fabless chips, different asset "
                           "-- though CPM itself is rarely the figure the filing states; "
                           "MAU/ARPU trend is what is actually disclosed",
        typical_cycle_months=(6, 18),
        watch_items=("advertiser vertical mix (retail, auto, travel)", "engagement trend",
                     "regulatory/antitrust overhang", "AI capex intensity vs revenue"),
    ),
    SubSegment.TECH_DEVICE_ECOSYSTEM: SegmentProfile(
        name="Consumer device + services ecosystem",
        cycle_driver="Hardware upgrade cycles, increasingly cushioned by a stickier "
                     "services attach",
        lead_lag="LAGS consumer discretionary spending; upgrade cycles elongate in a "
                 "downturn rather than reversing sharply",
        # MEASURED across 5,048 AAPL chunks: the generic anchor "services revenue" hit
        # only 16 times. The real phrasing is "Services net sales" (136 hits) -- Apple's
        # filings use "net sales" throughout, not "revenue", the same way MU's filings
        # never say "days of inventory" (taxonomy note above). Also strong: deferred
        # revenue 185, installed base 103, channel inventory 91.
        key_metrics=("unit sell-through", "channel inventory", "Services net sales growth",
                     "gross margin mix (services vs hardware)", "installed base growth"),
        inventory_dynamics="Channel sell-in vs sell-through divergence is the tell -- "
                           "identical mechanism to fabless chip designers: revenue can "
                           "hold while channel partners absorb inventory, then falls "
                           "abruptly when they stop reordering",
        typical_cycle_months=(12, 24),
        watch_items=("unit trends by region", "China exposure", "services attach rate",
                     "gross margin trajectory"),
    ),
    SubSegment.TECH_CLOUD_SAAS: SegmentProfile(
        name="Enterprise / cloud software",
        cycle_driver="Enterprise IT budgets and cloud migration/consumption growth",
        lead_lag="LAGS corporate capex sentiment; multi-year contracts smooth the cycle, "
                 "but renewal season concentrates the pain into discrete quarters",
        # MEASURED across 85,504 chunks (15 tickers): deferred revenue 3,841 and services
        # revenue 2,330 are the two most common figures, well ahead of the four already
        # listed here -- added both. billings 1,252, RPO 1,152, renewal rate 1,007,
        # customer count 945, ARR 856, net revenue retention only 318 (still usable, but
        # the least common of the group -- many issuers report NRR narratively rather
        # than as a clean quotable figure).
        key_metrics=("net revenue retention", "remaining performance obligations (RPO)",
                     "billings growth", "deferred revenue", "customer count",
                     "free cash flow margin", "seat/consumption growth"),
        inventory_dynamics="No physical inventory. RPO and deferred revenue are the "
                           "software equivalent of backlog -- a slowing RPO growth rate "
                           "is the earliest visible sign of deceleration, well before it "
                           "shows up in reported revenue",
        typical_cycle_months=(18, 36),
        watch_items=("net retention trend", "budget-scrutiny commentary in earnings calls",
                     "multi-year deal duration", "seat growth vs price increases"),
    ),
    SubSegment.TECH_CYBERSECURITY: SegmentProfile(
        name="Cybersecurity software",
        cycle_driver="Largely non-discretionary IT spend; more resilient than general "
                     "enterprise software, though deal TIMING still responds to budget "
                     "scrutiny even when underlying demand does not",
        lead_lag="LAGGING; security budgets are among the last enterprise software "
                 "categories to be cut in a downturn",
        # MEASURED across 13,711 PANW/CRWD/ZS chunks: "new logo", the term originally
        # used here, appears ONCE in the whole set -- effectively dead vocabulary. These
        # filings talk about retention, not new-customer acquisition: deferred revenue
        # 1,085, billings 548, ARR 359, net revenue retention 204, renewal rate 159,
        # "logo retention" (94) but not "new logo". Re-anchored on billings and renewal
        # rate, which are both real and well ahead of net-new-logo framing.
        key_metrics=("ARR growth", "billings growth", "net revenue retention",
                     "renewal rate", "platformization / module attach rate",
                     "free cash flow margin"),
        inventory_dynamics="No physical inventory; deferred revenue and billings play "
                           "the same early-warning role as in cloud SaaS. A renewal-rate "
                           "or net-retention deceleration is the specific tell for this "
                           "segment -- security budgets get cut by non-renewal before "
                           "they get cut by cancellation mid-contract",
        typical_cycle_months=(12, 24),
        watch_items=("renewal rate and net revenue retention trend",
                     "billings growth vs ARR growth (a leading/lagging pair)",
                     "platform consolidation", "macro-driven deal elongation"),
    ),
    SubSegment.TECH_COMMERCE_PLATFORM: SegmentProfile(
        name="Commerce and payments platforms",
        cycle_driver="Consumer discretionary spending and transaction/GMV volume; take "
                     "rate is a second, largely independent lever on revenue",
        lead_lag="COINCIDENT with consumer spending -- GMV is close to a real-time read "
                 "on discretionary demand",
        key_metrics=("GMV growth", "take rate", "active buyers/users",
                     "fulfillment/logistics cost as % of revenue"),
        inventory_dynamics="Mixed: AMZN alone carries real physical (retail) inventory; "
                           "the others are asset-light marketplaces where the leading "
                           "indicator is GMV growth decelerating ahead of take-rate or "
                           "margin pressure, not an inventory metric at all",
        typical_cycle_months=(6, 18),
        watch_items=("consumer spending proxies", "take-rate trend",
                     "competitive discounting", "logistics/delivery cost trend"),
    ),
    SubSegment.TECH_STREAMING_MEDIA: SegmentProfile(
        name="Subscription streaming media",
        cycle_driver="Subscriber growth and content spend discipline; increasingly an "
                     "advertising business as well",
        lead_lag="LAGS discretionary spending mildly; subscriptions are sticky but not "
                 "immune to cancellation waves in a squeeze",
        key_metrics=("subscriber net adds", "churn rate", "ARPU",
                     "content spend as % of revenue", "ad-tier adoption"),
        inventory_dynamics="No physical inventory; content amortization is the analogous "
                           "leading cost indicator -- spend committed years ahead of a "
                           "title's release shows up on the balance sheet before it hits "
                           "the P&L",
        typical_cycle_months=(12, 24),
        watch_items=("net-add trend", "churn rate", "password-sharing crackdown effects",
                     "ad-tier mix", "content slate strength"),
    ),
}
PROFILES.update(PROFILES_TECH)


def profile(seg: SubSegment) -> SegmentProfile:
    return PROFILES[seg]


def prompt_context(seg: SubSegment) -> str:
    """Render a sub-segment's cycle framing for inclusion in a Stage 5 prompt."""
    p = PROFILES[seg]
    lo, hi = p.typical_cycle_months
    return (
        f"SUB-SEGMENT: {p.name}\n"
        f"Cycle driver: {p.cycle_driver}\n"
        f"Cycle timing: {p.lead_lag}\n"
        f"Typical cycle length: {lo}-{hi} months\n"
        f"Inventory dynamics: {p.inventory_dynamics}\n"
        f"Key metrics: {', '.join(p.key_metrics)}\n"
        f"Watch: {', '.join(p.watch_items)}"
    )
