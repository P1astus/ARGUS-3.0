"""Generate the SFT dataset: run every reconstructed baseline Briefing through
sft_data.generate_sft_example on the real local model, write accepted (prompt, completion)
pairs to data/sft/examples.jsonl -- the path sft.SFTConfig.data_path defaults to.

RESUMABLE, same reasoning as baseline.py's run_arm(): appended immediately per case, and a
rerun skips tickers already present in the output file, so an interrupted run loses at most
the one case in flight.

Run: PYTHONPATH=src .venv/bin/python scripts/generate_sft_dataset.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from argus.contracts.briefing import Briefing
from argus.engine.extract.baseline import DEFAULT_CASES, TECH_CASES
from argus.engine.extract.prompts import STOP_SEQUENCES
from argus.engine.inference.local import BASE_MODEL, GenConfig, MLXGenerator
from argus.engine.training.sft_data import GenerationStats, generate_sft_example

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BRIEF_DIR = ROOT / "data" / "sft" / "briefings"
OUT_PATH = ROOT / "data" / "sft" / "examples.jsonl"

CASES = (*DEFAULT_CASES, *TECH_CASES)


def _done_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        done.add(obj["_key"])
    return done


def main(budget_seconds: float | None = None) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _done_tickers(OUT_PATH)
    if done:
        log.info("resuming: %d cases already complete", len(done))

    gen = MLXGenerator(GenConfig(model=BASE_MODEL, max_tokens=700, stop=STOP_SEQUENCES))
    stats = GenerationStats()
    t0 = time.time()

    for case in CASES:
        key = f"{case.ticker}_{case.as_of.isoformat()}"
        if key in done:
            continue
        if budget_seconds and (time.time() - t0) > budget_seconds:
            log.info("budget reached; rerun to resume the remaining cases")
            break

        brief_path = BRIEF_DIR / f"{key}.json"
        briefing = Briefing.model_validate_json(brief_path.read_text())

        t_case = time.time()
        ex = generate_sft_example(gen, case.ticker, case.segment, case.as_of, briefing,
                                  stats=stats)
        dt = time.time() - t_case

        if ex is None:
            log.info("%s: REJECTED (%.1fs)", key, dt)
            continue

        with OUT_PATH.open("a") as fh:
            fh.write(json.dumps({"_key": key, "prompt": ex.prompt,
                                 "completion": ex.completion}) + "\n")
        log.info("%s: accepted (%.1fs)", key, dt)

    log.info("")
    log.info(stats.render())
    log.info("wall_seconds=%.1f", time.time() - t0)
    log.info("output: %s", OUT_PATH)


if __name__ == "__main__":
    main()
