"""Walk-forward evaluation and the Phase 1 gate.

The gate has five criteria, all of which must hold. Four were added to the original
"mean IC consistently positive" because that alone would pass models that are useless:

  1. mean rank IC >= 0.03
  2. IC t-statistic >= 2.0            <- the criterion with actual teeth
  3. positive mean IC in >= 4 of 5 folds
  4. beats momentum_12_1 AND relative_strength_20d on IC and quintile spread
  5. Q1-Q5 spread positive net of costs

Criterion 4 is the important one. Without it a GBDT that has merely rediscovered momentum
clears every other bar while adding nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from argus.quant.dataset import Panel
from argus.quant.model import FoldResult, ModelConfig, QuantRanker
from argus.quant.universe import summarize_coverage
from argus.quant.validation.baselines import BASELINES, GATE_BASELINES
from argus.quant.validation.metrics import (
    ICResult,
    quantile_spread,
    rank_ic,
    shuffle_test,
    summarize_ic,
)
from argus.quant.validation.splitter import PurgedWalkForward

log = logging.getLogger(__name__)

GATE_MIN_IC = 0.03
GATE_MIN_T_STAT = 2.0
GATE_MIN_POSITIVE_FOLDS = 4


@dataclass
class GateResult:
    passed: bool
    criteria: dict[str, bool]
    detail: dict[str, object] = field(default_factory=dict)

    def render(self) -> str:
        lines = ["PHASE 1 GATE: " + ("PASS" if self.passed else "FAIL"), ""]
        for name, ok in self.criteria.items():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        return "\n".join(lines)


def run_walk_forward(
    panel: Panel,
    cv: PurgedWalkForward,
    config: ModelConfig | None = None,
    model_factory=None,
) -> list[FoldResult]:
    """Fit and predict fold by fold, refitting from scratch each time."""
    results: list[FoldResult] = []
    wide_y = panel.wide_labels()

    for fold in cv.split(panel.dates):
        X_tr, y_tr = panel.slice_dates(fold.train_dates)
        X_te, _ = panel.slice_dates(fold.test_dates)
        if X_tr.empty or X_te.empty:
            log.warning("fold %d: empty split, skipping", fold.fold_id)
            continue

        model = (model_factory() if model_factory else QuantRanker(config)).fit(
            X_tr.fillna(0.5), y_tr)
        preds = model.predict_wide(X_te.fillna(0.5))

        results.append(FoldResult(
            fold_id=fold.fold_id,
            predictions=preds,
            labels=wide_y.reindex(index=preds.index, columns=preds.columns),
            train_dates=(str(fold.train_dates[0].date()), str(fold.train_dates[-1].date())),
            test_dates=(str(fold.test_dates[0].date()), str(fold.test_dates[-1].date())),
            n_train=len(X_tr),
            n_test=len(X_te),
            importance=model.feature_importance(),
        ))
        log.info("fold %d: train %d rows, test %d rows", fold.fold_id, len(X_tr), len(X_te))

    return results


def evaluate(
    panel: Panel,
    folds: list[FoldResult],
    cost_bps: float = 10.0,
) -> dict:
    """Aggregate fold results, baselines, and the gate decision."""
    if not folds:
        raise RuntimeError("no folds produced results")

    per_fold: dict[int, ICResult] = {}
    fold_coverage: dict[int, dict] = {}
    for f in folds:
        per_fold[f.fold_id] = summarize_ic(rank_ic(f.predictions, f.labels))
        cov_slice = panel.coverage.reindex(f.predictions.index).dropna()
        if not cov_slice.empty:
            fold_coverage[f.fold_id] = {
                "period": f"{f.test_dates[0]} .. {f.test_dates[1]}",
                "coverage": float(cov_slice["coverage"].mean()),
                "n_available": float(cov_slice["n_available"].mean()),
            }

    all_preds = pd.concat([f.predictions for f in folds]).sort_index()
    all_labels = pd.concat([f.labels for f in folds]).sort_index()
    pooled = summarize_ic(rank_ic(all_preds, all_labels))
    spread = quantile_spread(all_preds, all_labels, cost_bps=cost_bps)

    # Baselines on the same out-of-sample dates only -- comparing an in-sample baseline
    # against an out-of-sample model would flatter the model.
    oos_dates = all_preds.index
    baseline_stats: dict[str, dict] = {}
    for name, fn in BASELINES.items():
        bp = fn(panel.close).reindex(index=oos_dates, columns=all_preds.columns)
        b_ic = summarize_ic(rank_ic(bp, all_labels))
        b_spread = quantile_spread(bp, all_labels, cost_bps=cost_bps)
        baseline_stats[name] = {
            "mean_ic": b_ic.mean_ic, "t_stat": b_ic.t_stat, "ir": b_ic.ir,
            "spread_net": b_spread["mean_spread_net"],
        }

    shuffled = shuffle_test(all_preds, all_labels, n_trials=20)

    positive_folds = sum(1 for r in per_fold.values() if r.mean_ic > 0)
    beats = {
        b: (pooled.mean_ic > baseline_stats[b]["mean_ic"]
            and spread["mean_spread_net"] > baseline_stats[b]["spread_net"])
        for b in GATE_BASELINES
    }

    criteria = {
        f"mean rank IC >= {GATE_MIN_IC}": pooled.mean_ic >= GATE_MIN_IC,
        f"IC t-stat >= {GATE_MIN_T_STAT}": pooled.t_stat >= GATE_MIN_T_STAT,
        f"positive IC in >= {GATE_MIN_POSITIVE_FOLDS}/{len(per_fold)} folds":
            positive_folds >= GATE_MIN_POSITIVE_FOLDS,
        "beats momentum_12_1 (IC and net spread)": beats.get("momentum_12_1", False),
        "beats relative_strength_20d (IC and net spread)":
            beats.get("relative_strength_20d", False),
        "Q1-Q5 spread positive net of costs": spread["mean_spread_net"] > 0,
    }

    gate = GateResult(passed=all(criteria.values()), criteria=criteria)

    return {
        "pooled_ic": pooled,
        "per_fold_ic": per_fold,
        "spread": spread,
        "baselines": baseline_stats,
        "shuffle": shuffled,
        "coverage": summarize_coverage(panel.coverage),
        "fold_coverage": fold_coverage,
        "gate": gate,
        "importance": pd.concat([f.importance for f in folds], axis=1).mean(axis=1)
                        .sort_values(ascending=False),
    }


def render(results: dict, panel: Panel) -> str:
    """Human-readable validation report. This output IS the Phase 1 gate."""
    p: ICResult = results["pooled_ic"]
    sp = results["spread"]
    cov = results["coverage"]
    out: list[str] = []

    out.append("=" * 74)
    out.append("ARGUS PHASE 1 -- QUANT CORE VALIDATION")
    out.append("=" * 74)

    out.append("\nUNIVERSE COVERAGE (survivorship-bias disclosure)")
    out.append(f"  mean coverage      {cov['mean_coverage']:.1%}")
    out.append(f"  worst year         {cov['worst_year']} at {cov['worst_year_coverage']:.1%}")
    out.append("  NOTE: yfinance serves no delisted history. Semiconductor exits since")
    out.append("        2010 skew towards acquisitions, so the IC below is an UPPER BOUND.")

    out.append("\nOUT-OF-SAMPLE INFORMATION COEFFICIENT")
    out.append(f"  {p.summary()}")

    out.append("\nPER-FOLD IC vs COVERAGE")
    out.append(f"  {'fold':4s} {'test period':26s} {'mean IC':>8} {'t':>7} {'coverage':>9} {'names':>7}")
    fold_cov = results.get("fold_coverage", {})
    for fid, r in sorted(results["per_fold_ic"].items()):
        fc = fold_cov.get(fid, {})
        period = fc.get("period", "")
        cov_s = f"{fc['coverage']:.1%}" if "coverage" in fc else "n/a"
        names_s = f"{fc['n_available']:.1f}" if "n_available" in fc else "n/a"
        out.append(f"  {fid:<4d} {period:26s} {r.mean_ic:>8.4f} {r.t_stat:>7.2f} "
                   f"{cov_s:>9} {names_s:>7}")

    # A fold whose IC is high precisely where coverage is low is the signature of a
    # survivorship artifact, not alpha: a thin cross-section is dominated by names that
    # survived, and ranking survivors is easier than ranking the real universe.
    if len(results["per_fold_ic"]) > 2 and fold_cov:
        ics, covs = [], []
        for fid, r in results["per_fold_ic"].items():
            if fid in fold_cov and "coverage" in fold_cov[fid] and r.mean_ic == r.mean_ic:
                ics.append(r.mean_ic)
                covs.append(fold_cov[fid]["coverage"])
        if len(ics) > 2:
            corr = float(np.corrcoef(ics, covs)[0, 1])
            out.append(f"\n  IC-vs-coverage correlation: {corr:+.2f}")
            if corr < -0.5:
                worst = min(results["per_fold_ic"].items(),
                            key=lambda kv: fold_cov.get(kv[0], {}).get("coverage", 1.0))
                rest = [r.mean_ic for f, r in results["per_fold_ic"].items() if f != worst[0]]
                out.append("  WARNING: IC is concentrated in the WORST-COVERED folds.")
                out.append("           This is the signature of a survivorship artifact.")
                out.append(f"           Mean IC excluding fold {worst[0]}: "
                           f"{np.mean(rest):+.4f} (vs pooled {p.mean_ic:+.4f})")

    out.append(f"\nQUINTILE SPREAD (Q1-Q5, {sp['cost_bps']:.0f}bps round trip)")
    out.append(f"  gross {sp['mean_spread_gross']:+.4f}   net {sp['mean_spread_net']:+.4f}"
               f"   t={sp['t_stat']:.2f}   n={sp['n_dates']}")

    out.append("\nBASELINES (same out-of-sample dates)")
    out.append(f"  {'predictor':26s} {'mean IC':>9} {'t-stat':>8} {'net spread':>11}")
    out.append(f"  {'MODEL':26s} {p.mean_ic:>9.4f} {p.t_stat:>8.2f} {sp['mean_spread_net']:>11.4f}")
    for name, b in results["baselines"].items():
        mark = "  <- gate" if name in GATE_BASELINES else ""
        out.append(f"  {name:26s} {b['mean_ic']:>9.4f} {b['t_stat']:>8.2f} "
                   f"{b['spread_net']:>11.4f}{mark}")

    sh = results["shuffle"]
    out.append("\nSHUFFLE TEST (labels permuted within date; IC must collapse to ~0)")
    out.append(f"  mean {sh['mean_shuffled_ic']:+.5f}   max|IC| {sh['max_abs_shuffled_ic']:.5f}"
               f"   over {sh['n_trials']} trials")
    if sh["max_abs_shuffled_ic"] > 0.02:
        out.append("  WARNING: shuffled IC is not ~0 -- suspect leakage in the pipeline.")

    out.append("\nFEATURE IMPORTANCE (mean across folds)")
    for name, v in results["importance"].head(8).items():
        out.append(f"  {name:28s} {v:8.1f}")

    out.append("\n" + "=" * 74)
    out.append(results["gate"].render())
    out.append("=" * 74)
    return "\n".join(out)
