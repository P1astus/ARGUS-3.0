"""Retrieval probes: a gold set defined by rule, not by hand.

THE PROBLEM WITH HAND-LABELLING HERE
Judging "is this passage relevant?" over 300k chunks needs either a lot of human hours or
a model, and a model-labelled gold set scored against model-based retrieval is circular:
the embedding model and the judge share the same notion of similarity, so dense wins by
construction. So relevance is defined by a rule that neither retriever has access to --
an anchor pattern plus a number -- and the queries are written against the *concept*.

TWO FAMILIES, BECAUSE ONE WOULD RIG THE RESULT
  direct     -- the query contains the anchor term itself ("book-to-bill ratio").
                BM25 should win these; a dense-only system that loses them badly is
                broken regardless of how good its paraphrase handling is.
  paraphrase -- the query deliberately avoids every anchor word and describes the concept
                instead ("ratio of orders received to products shipped"). Lexical match is
                unavailable by construction. This is the only family where dense can
                justify its 35 minutes of index build.

Reporting them separately is the point. A single blended recall number would let a gain
on one family hide a loss on the other, which is exactly the mistake §4d of the handoff
was written about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from argus.contracts.quant import SubSegment

# A gold chunk must carry a figure, not just the vocabulary. "We monitor book-to-bill
# closely" is boilerplate; "book-to-bill was 1.05" is the thing a briefing needs.
HAS_NUMBER = re.compile(r"\d")


@dataclass(frozen=True)
class Probe:
    key: str
    family: str                 # direct | paraphrase
    query: str
    anchor: re.Pattern          # defines the gold set, independently of any retriever
    segment: SubSegment
    tickers: tuple[str, ...]


def _p(key, family, query, anchor, segment, tickers):
    return Probe(key, family, query, re.compile(anchor, re.I), segment, tickers)


EQUIPMENT = ("AMAT", "LRCX", "KLAC", "ACLS", "ONTO", "VECO", "UCTT", "ICHR")
FABLESS = ("NVDA", "AMD", "AVGO", "QCOM", "MRVL", "SWKS", "QRVO", "CRUS", "SYNA", "LSCC")
MEMORY = ("MU", "WDC", "STX")
ANALOG = ("ADI", "TXN", "ON", "MPWR", "DIOD", "ALGM", "POWI", "SLAB", "MCHP", "LSCC")

# MEASURED: none of these has a single chunk in the corpus. Foundries are foreign private
# issuers filing 20-F, and the EDGAR builder collects 10-K/10-Q/8-K only. The FOUNDRY
# sub-segment therefore has NO evidence base -- `PROFILES[FOUNDRY]` will frame a briefing
# for which retrieval can return nothing. The probes are kept so the gap keeps showing up
# in the eval as `skipped` rather than being quietly forgotten.
FOUNDRY = ("TSM", "UMC", "GFS")

# tech_v1.yaml universe. All 26 resolved to CIKs cleanly (unlike FOUNDRY above), so no
# equivalent structural gap is expected here.
TECH_AD_PLATFORM = ("GOOGL", "META")
TECH_DEVICE_ECOSYSTEM = ("AAPL",)
TECH_CLOUD_SAAS = ("MSFT", "ORCL", "CRM", "ADBE", "INTU", "NOW", "WDAY", "SNOW", "DDOG",
                   "NET", "MDB", "TEAM", "IBM", "CSCO", "PLTR")
TECH_CYBERSECURITY = ("PANW", "CRWD", "ZS")
TECH_COMMERCE_PLATFORM = ("AMZN", "SHOP", "PYPL", "UBER")
TECH_STREAMING_MEDIA = ("NFLX",)

PROBES: tuple[Probe, ...] = (
    # ------------------------------------------------------------ equipment
    _p("bookings_direct", "direct",
       "book-to-bill ratio and bookings for the quarter",
       r"book[\s\-]?to[\s\-]?bill", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("bookings_para", "paraphrase",
       "how many new customer orders were received compared with how much was shipped",
       r"book[\s\-]?to[\s\-]?bill", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("backlog_direct", "direct",
       "systems backlog at the end of the quarter",
       r"backlog", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("backlog_para", "paraphrase",
       "value of orders already committed but not yet recognised as revenue",
       r"backlog", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("wfe_direct", "direct",
       "wafer fab equipment spending outlook for the year",
       r"wafer fab(rication)? equipment|\bWFE\b", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("wfe_para", "paraphrase",
       "how much customers plan to invest in new production tools this year",
       r"wafer fab(rication)? equipment|\bWFE\b", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("service_direct", "direct",
       "service and installed base revenue growth",
       r"installed base|service revenue", SubSegment.EQUIPMENT, EQUIPMENT),

    # -------------------------------------------------------------- memory
    #
    # The first version of these two probes anchored on "days of inventory" and returned
    # NO gold chunks -- MU, WDC and STX never use the phrase. It turns out to be analog /
    # microcontroller vocabulary (MCHP 181 chunks, SLAB 77, TXN 76, LSCC 35), while memory
    # issuers write about inventory as write-downs and carrying value. That is a defect in
    # `PROFILES[MEMORY].key_metrics`, which still lists "days of inventory" and therefore
    # sends the extraction loop looking for language its issuers do not use. Re-anchored
    # here on what the filings actually say; the analog version of the probe is below.
    _p("inventory_direct", "direct",
       "inventory write-down and carrying value of inventories",
       r"inventor\w*", SubSegment.MEMORY, MEMORY),
    _p("inventory_para", "paraphrase",
       "unsold stock that had to be marked down below what it cost to make",
       r"write[\s\-]?down|write[\s\-]?downs|lower of cost", SubSegment.MEMORY, MEMORY),
    _p("bitgrowth_direct", "direct",
       "bit shipments and bit growth for DRAM and NAND",
       r"bit (shipment|growth)", SubSegment.MEMORY, MEMORY),
    _p("bitgrowth_para", "paraphrase",
       "change in the total volume of storage capacity sold",
       r"bit (shipment|growth)", SubSegment.MEMORY, MEMORY),
    _p("asp_direct", "direct",
       "average selling prices per bit declined",
       r"average selling price", SubSegment.MEMORY, MEMORY),
    _p("asp_para", "paraphrase",
       "the amount customers paid per unit fell during the period",
       r"average selling price", SubSegment.MEMORY, MEMORY),

    # ------------------------------------------------------------- fabless
    _p("channel_direct", "direct",
       "channel inventory and distributor inventory weeks",
       r"channel inventory|distributor inventor", SubSegment.FABLESS, FABLESS),
    _p("channel_para", "paraphrase",
       "how much unsold product resellers are still holding",
       r"channel inventory|distributor inventor", SubSegment.FABLESS, FABLESS),
    _p("designwin_direct", "direct",
       "design wins with major customers",
       r"design win", SubSegment.FABLESS, FABLESS),
    _p("designwin_para", "paraphrase",
       "products selected for inclusion in a customer's future platform",
       r"design win", SubSegment.FABLESS, FABLESS),
    _p("grossmargin_direct", "direct",
       "gross margin percentage and the drivers of the change",
       r"gross margin", SubSegment.FABLESS, FABLESS),
    _p("grossmargin_para", "paraphrase",
       "share of sales left after the direct cost of making the product",
       r"gross margin", SubSegment.FABLESS, FABLESS),
    _p("foundryalloc_direct", "direct",
       "foundry wafer supply and capacity allocation",
       r"foundry (capacity|allocation|partner)|wafer supply", SubSegment.FABLESS, FABLESS),
    _p("concentration_direct", "direct",
       "customer concentration percentage of total revenue",
       r"(customer|customers) accounted for", SubSegment.FABLESS, FABLESS),
    _p("concentration_para", "paraphrase",
       "how dependent the business is on a small number of buyers",
       r"(customer|customers) accounted for", SubSegment.FABLESS, FABLESS),

    # -------------------------------------------------------------- analog
    _p("leadtime_direct", "direct",
       "lead times for our products extended",
       r"lead[\s\-]?time", SubSegment.ANALOG, ANALOG),
    _p("leadtime_para", "paraphrase",
       "how long customers now wait between ordering and delivery",
       r"lead[\s\-]?time", SubSegment.ANALOG, ANALOG),
    _p("utilisation_direct", "direct",
       "factory utilization rates during the quarter",
       r"utili[sz]ation", SubSegment.ANALOG, ANALOG),
    _p("utilisation_para", "paraphrase",
       "what proportion of manufacturing capacity was actually in use",
       r"utili[sz]ation", SubSegment.ANALOG, ANALOG),
    _p("autoindustrial_direct", "direct",
       "automotive and industrial end market revenue",
       r"automotive", SubSegment.ANALOG, ANALOG),
    # Where "days of inventory" actually lives in this corpus. See the memory note above.
    _p("daysinv_direct", "direct",
       "days of inventory at the end of the period",
       r"days of inventory", SubSegment.ANALOG, ANALOG),
    _p("daysinv_para", "paraphrase",
       "how long current stock on hand would last at the present rate of sale",
       r"days of inventory", SubSegment.ANALOG, ANALOG),

    # ------------------------------------------------------------- foundry
    _p("nodemix_direct", "direct",
       "advanced node revenue mix by technology",
       r"\b\d+\s?(nm|nanometer)", SubSegment.FOUNDRY, FOUNDRY),
    _p("capex_direct", "direct",
       "capital expenditure plans for the coming year",
       r"capital expenditure", SubSegment.FOUNDRY, FOUNDRY),
    _p("capex_para", "paraphrase",
       "how much money is being committed to new plant and machinery",
       r"capital expenditure", SubSegment.FOUNDRY, FOUNDRY),

    # ------------------------------------------- cross-segment, guidance language
    _p("guidance_direct", "direct",
       "guidance for next quarter revenue and gross margin",
       r"we expect|expects? .{0,40}(revenue|net sales)", SubSegment.FABLESS, FABLESS),
    _p("guidance_para", "paraphrase",
       "what management said the coming three months would look like",
       r"we expect|expects? .{0,40}(revenue|net sales)", SubSegment.FABLESS, FABLESS),
    _p("chinaexport_direct", "direct",
       "China export control restrictions impact on revenue",
       r"export (control|license|restriction)", SubSegment.EQUIPMENT, EQUIPMENT),
    _p("chinaexport_para", "paraphrase",
       "government limits on selling our products to certain overseas customers",
       r"export (control|license|restriction)", SubSegment.EQUIPMENT, EQUIPMENT),

    # ------------------------------------------------------- TECH: ad platform
    #
    # UNMEASURED, like PROFILES_TECH in taxonomy.py -- written before any tech filing was
    # indexed. Expect a correction pass here identical to the MEMORY "days of inventory"
    # fix above once `retrieve evaluate` has real tech-corpus counts to check these
    # against. Do not treat these anchors as validated vocabulary yet.
    # CPM was the original anchor here and returned 5 hits across 9,083 GOOGL/META
    # chunks -- essentially unusable. MAU has 236 gold (concept + digit) chunks; see the
    # taxonomy.py note on TECH_AD_PLATFORM for the measurement.
    _p("mau_direct", "direct",
       "monthly active users growth",
       r"monthly active users|\bMAU\b", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),
    _p("mau_para", "paraphrase",
       "how many people used the product at least once in the past month",
       r"monthly active users|\bMAU\b", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),
    _p("dau_direct", "direct",
       "daily active users growth",
       r"daily active users|\bDAU\b", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),
    _p("dau_para", "paraphrase",
       "how many people use the product on a typical day",
       r"daily active users|\bDAU\b", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),
    _p("adrev_direct", "direct",
       "advertising revenue growth this quarter",
       r"advertising revenue", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),
    _p("adrev_para", "paraphrase",
       "how much money came from selling ads",
       r"advertising revenue", SubSegment.TECH_AD_PLATFORM, TECH_AD_PLATFORM),

    # --------------------------------------------------- TECH: device ecosystem
    # "services revenue" alone hit only 16/5,048 AAPL chunks; "Services net sales" (the
    # phrasing Apple's own filings use) hit 136. Anchor covers both spellings.
    _p("svcrev_direct", "direct",
       "Services net sales growth",
       r"services net sales|services revenue", SubSegment.TECH_DEVICE_ECOSYSTEM,
       TECH_DEVICE_ECOSYSTEM),
    _p("svcrev_para", "paraphrase",
       "money made from subscriptions and services rather than hardware sales",
       r"services net sales|services revenue", SubSegment.TECH_DEVICE_ECOSYSTEM,
       TECH_DEVICE_ECOSYSTEM),
    _p("devchannel_direct", "direct",
       "channel inventory levels for our products",
       r"channel inventory", SubSegment.TECH_DEVICE_ECOSYSTEM, TECH_DEVICE_ECOSYSTEM),
    _p("devchannel_para", "paraphrase",
       "how much unsold product resellers are still holding",
       r"channel inventory", SubSegment.TECH_DEVICE_ECOSYSTEM, TECH_DEVICE_ECOSYSTEM),
    _p("installedbase_direct", "direct",
       "installed base of active devices",
       r"installed base", SubSegment.TECH_DEVICE_ECOSYSTEM, TECH_DEVICE_ECOSYSTEM),
    _p("installedbase_para", "paraphrase",
       "the total number of devices still in active use by customers",
       r"installed base", SubSegment.TECH_DEVICE_ECOSYSTEM, TECH_DEVICE_ECOSYSTEM),

    # ------------------------------------------------------- TECH: cloud SaaS
    _p("nrr_direct", "direct",
       "net revenue retention rate",
       r"net (revenue |dollar )?retention", SubSegment.TECH_CLOUD_SAAS, TECH_CLOUD_SAAS),
    _p("nrr_para", "paraphrase",
       "how much more or less existing customers spent compared with a year ago",
       r"net (revenue |dollar )?retention", SubSegment.TECH_CLOUD_SAAS, TECH_CLOUD_SAAS),
    _p("rpo_direct", "direct",
       "remaining performance obligations at the end of the period",
       r"remaining performance obligation|\bRPO\b", SubSegment.TECH_CLOUD_SAAS,
       TECH_CLOUD_SAAS),
    _p("rpo_para", "paraphrase",
       "value of contracted revenue not yet recognised",
       r"remaining performance obligation|\bRPO\b", SubSegment.TECH_CLOUD_SAAS,
       TECH_CLOUD_SAAS),
    _p("billings_direct", "direct",
       "billings growth for the quarter",
       r"\bbillings\b", SubSegment.TECH_CLOUD_SAAS, TECH_CLOUD_SAAS),
    _p("billings_para", "paraphrase",
       "the amount invoiced to customers during the period",
       r"\bbillings\b", SubSegment.TECH_CLOUD_SAAS, TECH_CLOUD_SAAS),

    # ---------------------------------------------------- TECH: cybersecurity
    _p("arr_direct", "direct",
       "annual recurring revenue growth",
       r"annual recurring revenue|\bARR\b", SubSegment.TECH_CYBERSECURITY,
       TECH_CYBERSECURITY),
    _p("arr_para", "paraphrase",
       "the yearly value of all active subscription contracts",
       r"annual recurring revenue|\bARR\b", SubSegment.TECH_CYBERSECURITY,
       TECH_CYBERSECURITY),
    # "new logo" hit ONCE across all of PANW/CRWD/ZS -- essentially dead vocabulary; these
    # filings talk in terms of renewal rate and net retention, not new-customer framing.
    # See the taxonomy.py note on TECH_CYBERSECURITY for the full measurement.
    _p("cyberrenewal_direct", "direct",
       "renewal rate for existing customers",
       r"renewal rate", SubSegment.TECH_CYBERSECURITY, TECH_CYBERSECURITY),
    _p("cyberrenewal_para", "paraphrase",
       "what share of customers chose to continue their subscription at expiration",
       r"renewal rate", SubSegment.TECH_CYBERSECURITY, TECH_CYBERSECURITY),
    _p("attach_direct", "direct",
       "attach rate of additional security modules",
       r"attach rate|cross-sell", SubSegment.TECH_CYBERSECURITY, TECH_CYBERSECURITY),
    _p("attach_para", "paraphrase",
       "how many extra products existing customers bought on top of their first purchase",
       r"attach rate|cross-sell", SubSegment.TECH_CYBERSECURITY, TECH_CYBERSECURITY),

    # ------------------------------------------------- TECH: commerce platform
    _p("gmv_direct", "direct",
       "gross merchandise value growth",
       r"gross merchandise (value|volume)|\bGMV\b", SubSegment.TECH_COMMERCE_PLATFORM,
       TECH_COMMERCE_PLATFORM),
    _p("gmv_para", "paraphrase",
       "the total dollar value of everything sold through the platform",
       r"gross merchandise (value|volume)|\bGMV\b", SubSegment.TECH_COMMERCE_PLATFORM,
       TECH_COMMERCE_PLATFORM),
    _p("takerate_direct", "direct",
       "take rate as a percentage of GMV",
       r"take rate", SubSegment.TECH_COMMERCE_PLATFORM, TECH_COMMERCE_PLATFORM),
    _p("takerate_para", "paraphrase",
       "what share of each transaction the platform keeps as revenue",
       r"take rate", SubSegment.TECH_COMMERCE_PLATFORM, TECH_COMMERCE_PLATFORM),
    _p("activebuyers_direct", "direct",
       "active buyers on the platform",
       r"active (buyers|customers|users)", SubSegment.TECH_COMMERCE_PLATFORM,
       TECH_COMMERCE_PLATFORM),
    _p("activebuyers_para", "paraphrase",
       "how many distinct customers made a purchase in the period",
       r"active (buyers|customers|users)", SubSegment.TECH_COMMERCE_PLATFORM,
       TECH_COMMERCE_PLATFORM),

    # -------------------------------------------------- TECH: streaming media
    _p("netadds_direct", "direct",
       "subscriber net additions for the quarter",
       r"(subscriber|member)s? (net )?add", SubSegment.TECH_STREAMING_MEDIA,
       TECH_STREAMING_MEDIA),
    _p("netadds_para", "paraphrase",
       "how many new paying members joined minus those who left",
       r"(subscriber|member)s? (net )?add", SubSegment.TECH_STREAMING_MEDIA,
       TECH_STREAMING_MEDIA),
    _p("churn_direct", "direct",
       "churn rate for the period",
       r"\bchurn\b", SubSegment.TECH_STREAMING_MEDIA, TECH_STREAMING_MEDIA),
    _p("churn_para", "paraphrase",
       "the share of subscribers who cancelled",
       r"\bchurn\b", SubSegment.TECH_STREAMING_MEDIA, TECH_STREAMING_MEDIA),
    _p("arpu_direct", "direct",
       "average revenue per membership",
       r"average revenue per (member|user|subscriber)|\bARPU\b",
       SubSegment.TECH_STREAMING_MEDIA, TECH_STREAMING_MEDIA),
    _p("arpu_para", "paraphrase",
       "average amount each subscriber pays",
       r"average revenue per (member|user|subscriber)|\bARPU\b",
       SubSegment.TECH_STREAMING_MEDIA, TECH_STREAMING_MEDIA),
)


def is_gold(text: str, probe: Probe) -> bool:
    """A chunk is gold if it carries the anchor concept AND a figure."""
    return bool(probe.anchor.search(text)) and bool(HAS_NUMBER.search(text))
