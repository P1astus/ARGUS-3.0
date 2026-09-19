"""Score BM25, dense and hybrid retrieval against the rule-defined probes.

Reports recall@k and MRR per family. The number that decides whether the dense index was
worth building is `paraphrase` recall, because `direct` recall is a lexical task that
BM25 should already win -- and if a mode loses badly on `direct` it is disqualified no
matter what it does on paraphrase, since real analyst queries contain the domain terms.

`skipped` is reported explicitly: a probe whose filter yields no gold chunk at all is not
a retrieval failure and must not be averaged in as one. Silent exclusion is how a harness
ends up reporting a number computed over a different question than the one asked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from argus.engine.retrieval.embed import Embedder
from argus.engine.retrieval.probes import PROBES, Probe, is_gold
from argus.engine.retrieval.search import SearchConfig, Searcher
from argus.engine.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    key: str
    family: str
    mode: str
    n_gold: int
    n_candidates: int
    hit_ranks: list[int] = field(default_factory=list)   # 1-based ranks of gold hits
    seconds: float = 0.0

    def recall_at(self, k: int) -> float:
        return 1.0 if any(r <= k for r in self.hit_ranks) else 0.0

    @property
    def rr(self) -> float:
        return 1.0 / min(self.hit_ranks) if self.hit_ranks else 0.0


def gold_ids(store: ChunkStore, probe: Probe, limit_scan: int = 40000) -> set[int]:
    """Chunk ids satisfying the probe's rule, within the probe's ticker filter.

    Computed by scanning the filtered candidate set rather than by querying the FTS
    index: using the lexical index to define the gold set would hand BM25 the answer key.
    """
    cand = store.candidate_ids(tickers=list(probe.tickers))
    if len(cand) > limit_scan:
        # Deterministic thinning, so the gold set is reproducible across runs.
        step = len(cand) // limit_scan + 1
        cand = cand[::step]
    gold = set()
    for i in range(0, len(cand), 2000):
        for ref in store.get(cand[i:i + 2000]):
            if is_gold(ref.text(), probe):
                gold.add(ref.chunk_id)
    return gold


def run(store: ChunkStore, embedder: Embedder | None,
        modes: tuple[str, ...] = ("bm25", "dense", "hybrid"),
        k: int = 20, probes: tuple[Probe, ...] = PROBES) -> dict:
    searcher = Searcher(store, embedder)
    results: list[ProbeResult] = []
    skipped: list[str] = []

    for probe in probes:
        gold = gold_ids(store, probe)
        if not gold:
            skipped.append(probe.key)
            log.warning("probe %s has no gold chunks under its filter; skipping",
                        probe.key)
            continue

        for mode in modes:
            if mode in ("dense", "hybrid") and (embedder is None or store.vectors is None):
                continue
            cfg = SearchConfig(top_k=k, candidate_k=max(60, k * 3), mode=mode,
                               max_per_doc=k)   # dedup would distort recall accounting
            t = time.time()
            hits = searcher.search(probe.query, tickers=list(probe.tickers), config=cfg)
            dt = time.time() - t
            ranks = [i for i, h in enumerate(hits, start=1) if h.ref.chunk_id in gold]
            results.append(ProbeResult(probe.key, probe.family, mode, len(gold),
                                       0, ranks, dt))

    return summarise(results, k=k, skipped=skipped)


def summarise(results: list[ProbeResult], k: int, skipped: list[str]) -> dict:
    out: dict = {"k": k, "skipped_probes": skipped, "by_mode": {}, "probes": []}
    modes = sorted({r.mode for r in results})
    families = sorted({r.family for r in results})

    for mode in modes:
        rows = [r for r in results if r.mode == mode]
        entry = {"n_probes": len(rows),
                 "recall_at_k": _mean(r.recall_at(k) for r in rows),
                 "recall_at_5": _mean(r.recall_at(5) for r in rows),
                 "mrr": _mean(r.rr for r in rows),
                 "mean_latency_ms": 1000 * _mean(r.seconds for r in rows)}
        for fam in families:
            frows = [r for r in rows if r.family == fam]
            entry[fam] = {"n": len(frows),
                          "recall_at_k": _mean(r.recall_at(k) for r in frows),
                          "recall_at_5": _mean(r.recall_at(5) for r in frows),
                          "mrr": _mean(r.rr for r in frows)}
        out["by_mode"][mode] = entry

    for r in results:
        out["probes"].append({"key": r.key, "family": r.family, "mode": r.mode,
                              "n_gold": r.n_gold, "first_rank": min(r.hit_ranks, default=None),
                              "recall_at_k": r.recall_at(k)})
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def render(report: dict) -> str:
    k = report["k"]
    lines = ["RETRIEVAL EVALUATION", "=" * 72, ""]
    lines.append(f"{'mode':8s} {'n':>4s} {'R@5':>7s} {f'R@{k}':>7s} {'MRR':>7s} "
                 f"{'ms':>7s}   {'direct R@'+str(k):>14s} {'para R@'+str(k):>14s}")
    for mode, e in report["by_mode"].items():
        d = e.get("direct", {}).get("recall_at_k", float("nan"))
        p = e.get("paraphrase", {}).get("recall_at_k", float("nan"))
        lines.append(f"{mode:8s} {e['n_probes']:>4d} {e['recall_at_5']:>7.3f} "
                     f"{e['recall_at_k']:>7.3f} {e['mrr']:>7.3f} "
                     f"{e['mean_latency_ms']:>7.1f}   {d:>14.3f} {p:>14.3f}")
    if report["skipped_probes"]:
        lines += ["", f"skipped (no gold chunk under filter): "
                      f"{', '.join(report['skipped_probes'])}"]
    lines += ["", "direct = query contains the domain term (BM25 should win these)",
              "para   = query avoids the term entirely (the case dense exists for)"]
    return "\n".join(lines)
