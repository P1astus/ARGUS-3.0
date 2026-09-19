"""Baseline extraction quality on the untrained model.

WHY THIS RUNS BEFORE ANY CPT
Handoff §8 is explicit: a pilot CPT run before this exists produces a loss curve with
nothing to compare against. Cross-entropy on filings going down tells you the model got
better at predicting filing text; it does not tell you whether the *extraction* improved,
which is the only thing the new architecture asks CPT to deliver. These numbers are the
denominator for that question.

WHAT IS AND IS NOT MEASURED
Measured, mechanically and without labels:
  * exact_quote_rate     -- share of quotes copied character-perfect. The honesty metric.
  * fabrication_rate     -- quotes present in no shown passage. The dangerous failure.
  * claim_yield          -- claims accepted per passage; a proxy for whether the model
                            finds anything at all.
  * sourced_fraction     -- how much of the briefing is evidence vs the model talking.
  * format validity      -- malformed lines, missing fields.

NOT measured here: whether the claims are the *important* ones. That is salience, it needs
judgment, and the honest place for it is the frontier model reading the briefings -- so
this harness writes the briefings out for exactly that. Do not let a good number here be
read as "extraction is good"; it means "extraction is not lying", which is a lower bar
and the one that can be automated.

ARMS
Base and Instruct are both run. Handoff §4b measured a 3.8% instruct BPC penalty on domain
text, but instruction-following is a different axis from domain fit, and this loop asks
for a format a base model has never been trained to produce. Running both separates "the
extraction task is hard" from "the base checkpoint cannot follow the format" -- and if the
gap is large, that is direct evidence for the Llama-Fin finding in §4e that CPT and
instruction-following interact, which changes how the CPT run should be configured.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from argus.contracts.quant import SubSegment
from argus.engine.extract.claims import ExtractionStats
from argus.engine.extract.export import to_markdown
from argus.engine.extract.loop import ExtractConfig, ExtractionLoop
from argus.engine.inference.local import BASE_MODEL, INSTRUCT_MODEL, GenConfig, MLXGenerator
from argus.engine.extract.prompts import STOP_SEQUENCES
from argus.engine.retrieval.search import Searcher
from argus.tools.provenance import ProvenanceStore, audit_briefing

log = logging.getLogger(__name__)


@dataclass
class BaselineCase:
    ticker: str
    segment: SubSegment
    as_of: date


# Deliberately spread across all five sub-segments and across the cycle. A baseline
# measured only on NVDA in 2024 would be a measurement of one unusually well-covered
# issuer in one unusually eventful year.
DEFAULT_CASES: tuple[BaselineCase, ...] = (
    BaselineCase("AMAT", SubSegment.EQUIPMENT, date(2019, 11, 1)),   # equipment downturn
    BaselineCase("LRCX", SubSegment.EQUIPMENT, date(2023, 5, 1)),    # export controls
    BaselineCase("MU",   SubSegment.MEMORY,    date(2023, 3, 1)),    # memory trough
    BaselineCase("WDC",  SubSegment.MEMORY,    date(2022, 8, 1)),
    BaselineCase("NVDA", SubSegment.FABLESS,   date(2024, 5, 1)),    # AI upcycle
    BaselineCase("AMD",  SubSegment.FABLESS,   date(2022, 11, 1)),   # PC correction
    BaselineCase("ADI",  SubSegment.ANALOG,    date(2024, 2, 1)),    # analog destock
    BaselineCase("MPWR", SubSegment.ANALOG,    date(2023, 8, 1)),
    BaselineCase("ON",   SubSegment.ANALOG,    date(2025, 5, 1)),
    BaselineCase("CRUS", SubSegment.FABLESS,   date(2021, 8, 1)),    # shortage era
)

# tech_v1.yaml universe. Same spread principle as DEFAULT_CASES: touch every sub-segment,
# not just the largest one, and vary the period rather than clustering on one moment.
TECH_CASES: tuple[BaselineCase, ...] = (
    BaselineCase("GOOGL", SubSegment.TECH_AD_PLATFORM,    date(2023, 2, 1)),  # ad slowdown
    BaselineCase("META",  SubSegment.TECH_AD_PLATFORM,    date(2023, 5, 1)),  # recovery
    BaselineCase("AAPL",  SubSegment.TECH_DEVICE_ECOSYSTEM, date(2023, 11, 1)),
    BaselineCase("CRM",   SubSegment.TECH_CLOUD_SAAS,     date(2024, 5, 1)),
    BaselineCase("NOW",   SubSegment.TECH_CLOUD_SAAS,     date(2023, 8, 1)),
    BaselineCase("PANW",  SubSegment.TECH_CYBERSECURITY,  date(2023, 8, 1)),
    BaselineCase("CRWD",  SubSegment.TECH_CYBERSECURITY,  date(2024, 3, 1)),
    BaselineCase("AMZN",  SubSegment.TECH_COMMERCE_PLATFORM, date(2023, 2, 1)),
    # NOT SHOP: measured after the fact that SHOP's indexed corpus only covers
    # 2025-02-11 .. 2025-12-02 (Shopify is a Canadian filer; most of its EDGAR history is
    # under a different filing convention -- the same class of gap as the FOUNDRY 20-F
    # issue). SHOP@2023-05-01 correctly returned zero passages under the as_of leakage
    # filter, which is the filter doing its job, not a bug -- but it made a bad case
    # choice. PYPL has clean 10-K/10-Q coverage from its 2015 spinoff onward.
    BaselineCase("PYPL",  SubSegment.TECH_COMMERCE_PLATFORM, date(2023, 5, 1)),
    BaselineCase("NFLX",  SubSegment.TECH_STREAMING_MEDIA, date(2023, 1, 1)),  # password crackdown
)


@dataclass
class ArmResult:
    model: str
    stats: ExtractionStats = field(default_factory=ExtractionStats)
    per_case: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    n_cases: int = 0

    def summary(self) -> dict:
        """Aggregate from the per-case records, not from in-memory stats.

        Resumed cases are read back from `cases.jsonl` and never touch `self.stats`, so
        summing the records is the only way a partially-resumed arm reports the whole
        arm rather than just the slice this process happened to run.
        """
        rows = [c for c in self.per_case if "error" not in c]

        def total(key: str) -> int:
            return sum(int(c.get(key) or 0) for c in rows)

        passages = total("n_sources")
        sourced = total("sourced_claims")
        inference = total("inference_claims")
        accepted = sourced + inference
        exact = total("quote_exact")
        whitespace = total("quote_whitespace")
        recited = total("quote_recited")
        too_short = total("quote_too_short")
        fabricated = total("quote_fabricated")
        attempts = exact + whitespace + recited + too_short + fabricated
        seconds = sum(float(c.get("seconds") or 0) for c in rows)
        n = max(len(rows), 1)

        return {
            "model": self.model,
            "cases": len(rows),
            "cases_failed": len(self.per_case) - len(rows),
            "seconds": round(seconds, 1),
            "seconds_per_case": round(seconds / n, 1),
            "passages_shown": passages,
            "claims_accepted": accepted,
            "claim_yield_per_passage": round(accepted / max(passages, 1), 2),
            "sourced_claims": sourced,
            "inference_claims": inference,
            "sourced_fraction": round(sourced / max(accepted, 1), 3),
            "quote_attempts": attempts,
            "exact_quote_rate": round(exact / attempts, 3) if attempts else None,
            "whitespace_repaired": whitespace,
            "fabrication_rate": round(fabricated / attempts, 3) if attempts else None,
            "quote_fabricated": fabricated,
            "quote_too_short": too_short,
            "miscited_recovered": recited,
            "unknown_source_ids": total("unknown_source"),
            "malformed_json_lines": total("malformed_json"),
            "missing_field_lines": total("missing_fields"),
            "inference_with_citation": total("inference_with_citation"),
        }


def _load_done(arm_dir: Path) -> dict[str, dict]:
    """Per-case results already on disk, keyed by ticker@as_of."""
    path = arm_dir / "cases.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[f"{rec.get('ticker')}@{rec.get('as_of')}"] = rec
    return out


def run_arm(model_name: str, searcher: Searcher, store: ProvenanceStore,
            cases: tuple[BaselineCase, ...], out_dir: Path,
            config: ExtractConfig | None = None,
            budget_seconds: float | None = None) -> ArmResult:
    """Run one arm, resumably.

    RESUMABILITY IS NOT A CONVENIENCE HERE. A case costs ~312s in the foreground and ~18x
    that when the process is backgrounded (§6 of the handoff), so the full run cannot be
    done in one detached job and does not fit in one interactive window either. Completed
    cases are appended to `cases.jsonl` and the arm summary is aggregated from that file,
    so the baseline can be built up across as many short foreground runs as it takes.
    """
    gen = MLXGenerator(GenConfig(model=model_name, stop=STOP_SEQUENCES))
    loop = ExtractionLoop(searcher, gen, store, config or ExtractConfig())
    arm = ArmResult(model=model_name)
    t0 = time.time()

    tag = "base" if model_name == BASE_MODEL else "instruct"
    arm_dir = out_dir / tag
    arm_dir.mkdir(parents=True, exist_ok=True)
    done = _load_done(arm_dir)
    if done:
        log.info("%s: resuming, %d cases already complete", tag, len(done))

    for case in cases:
        key = f"{case.ticker}@{case.as_of.isoformat()}"
        if key in done:
            arm.per_case.append(done[key])
            continue
        if budget_seconds and (time.time() - t0) > budget_seconds:
            log.info("%s: budget reached after %d new cases; rerun to resume",
                     tag, arm.n_cases)
            break
        try:
            res = loop.run(case.ticker, case.segment, case.as_of)
        except Exception as e:
            log.exception("extraction failed for %s: %s", case.ticker, e)
            arm.per_case.append({"ticker": case.ticker, "error": f"{type(e).__name__}: {e}",
                                 "n_sources": 0})
            continue

        # The audit is run against the retained bytes, not against the loop's own
        # bookkeeping -- a self-reported pass would be worth nothing.
        audit = audit_briefing(res.briefing, store)
        arm.n_cases += 1
        record = {**res.audit, "gate_passed": audit["passed"],
                  "n_unverified": audit["n_unverified_quotes"]}
        arm.per_case.append(record)
        # Appended immediately, before the next case starts: a run interrupted at case 7
        # must not lose the six that already cost half an hour each.
        with (arm_dir / "cases.jsonl").open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        md = to_markdown(res.briefing, {**res.audit, "model": model_name})
        (arm_dir / f"{case.ticker}_{case.as_of.isoformat()}.md").write_text(md)
        (arm_dir / f"{case.ticker}_{case.as_of.isoformat()}.raw.txt").write_text(
            "\n\n===== NEXT GENERATION =====\n\n".join(res.raw_outputs))
        log.info("%s %s %s: %d claims (%.0f%% sourced), exact quotes %.0f%%, "
                 "fabricated %d, %.0fs", tag, case.ticker, case.as_of,
                 len(res.briefing.claims), 100 * res.briefing.sourced_fraction,
                 100 * (res.stats.exact_quote_rate if res.stats.quote_attempts else 0),
                 res.stats.quote_fabricated, res.seconds)

    arm.seconds = time.time() - t0
    return arm


def run(searcher: Searcher, store: ProvenanceStore, out_dir: Path,
        arms: tuple[str, ...] = (BASE_MODEL, INSTRUCT_MODEL),
        cases: tuple[BaselineCase, ...] = DEFAULT_CASES,
        config: ExtractConfig | None = None,
        budget_seconds: float | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": date.today().isoformat(),
              "n_cases": len(cases),
              "cases": [{"ticker": c.ticker, "segment": str(c.segment),
                         "as_of": c.as_of.isoformat()} for c in cases],
              "arms": {}}

    for model_name in arms:
        arm = run_arm(model_name, searcher, store, cases, out_dir, config,
                      budget_seconds=budget_seconds)
        summary = arm.summary()
        summary["complete"] = summary["cases"] == len(cases)
        report["arms"][model_name] = {"summary": summary, "per_case": arm.per_case}
        (out_dir / "baseline.json").write_text(json.dumps(report, indent=2, default=str))

    (out_dir / "baseline.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def render(report: dict) -> str:
    rows = [(m, a["summary"]) for m, a in report["arms"].items()]
    lines = ["BASELINE EXTRACTION QUALITY (untrained)", "=" * 78, "",
             f"{len(report['cases'])} cases across five sub-segments, "
             f"{report['generated_at']}", ""]

    fields = [
        ("cases complete", "cases", "{}"),
        ("claims accepted", "claims_accepted", "{}"),
        ("claim yield / passage", "claim_yield_per_passage", "{}"),
        ("sourced fraction", "sourced_fraction", "{}"),
        ("quote attempts", "quote_attempts", "{}"),
        ("EXACT quote rate", "exact_quote_rate", "{}"),
        ("whitespace-repaired", "whitespace_repaired", "{}"),
        ("FABRICATION rate", "fabrication_rate", "{}"),
        ("  fabricated (dropped)", "quote_fabricated", "{}"),
        ("quote too short", "quote_too_short", "{}"),
        ("miscited, recovered", "miscited_recovered", "{}"),
        ("malformed JSON lines", "malformed_json_lines", "{}"),
        ("missing-field lines", "missing_field_lines", "{}"),
        ("inference w/ citation", "inference_with_citation", "{}"),
        ("seconds / case", "seconds_per_case", "{}"),
    ]

    names = [m.split("/")[-1] for m, _ in rows]
    lines.append(f"{'metric':26s}" + "".join(f"{n[:26]:>28s}" for n in names))
    lines.append("-" * (26 + 28 * len(names)))
    for label, key, fmt in fields:
        cells = "".join(f"{str(s.get(key)):>28s}" for _, s in rows)
        lines.append(f"{label:26s}{cells}")

    lines += ["", "EXACT quote rate is the headline: repairs are excluded from it on "
                  "purpose.", "FABRICATION rate is the number that must stay near zero -- "
                  "those claims were dropped,", "but a model that produces them at scale "
                  "cannot be trusted to extract unsupervised.", "",
              "Salience (are these the RIGHT claims?) is not measured here. Read the "
              "briefings in", "artifacts/baseline/*/ for that."]
    return "\n".join(lines)
