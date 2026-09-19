# ARGUS 3.0

A local semiconductor/tech research assistant. A local model (Ministral-3-14B-Base, running on
Apple Silicon via MLX) does the volume work: retrieving from a corpus of SEC EDGAR filings,
extracting claims with verbatim quotes, and assembling short briefings. A frontier model
(Claude) then does the judgment work conversationally on those briefings.

This is **decision support, not automation**: the user stays in the loop by design, and the
local model does not produce trade recommendations.

```
local, free                                    interactive
┌──────────────────────────────┐   briefing   ┌────────────────────┐
│ Ministral-14B (MLX)          │  (2-5k tok)  │ Claude             │
│  retrieve from EDGAR corpus  │ ───────────► │  analyses briefing │
│  extract claims + quotes     │              │  talks it through  │
│  diff filings, rank, shortlist│             │  with the user     │
└──────────────────────────────┘              └────────────────────┘
```

A 10-K is roughly 73-100k tokens; a briefing is 2-5k.

## Status

Personal research project, not a polished product. The design decisions, every measured
benchmark, failed experiments and known gotchas are recorded in [`docs/HANDOFF.md`](docs/HANDOFF.md).
Retrieval/extraction measurements are in [`docs/retrieval_and_extraction.md`](docs/retrieval_and_extraction.md),
and the literature the design rests on is in [`docs/literature_review.md`](docs/literature_review.md).

The repo does not include the filing corpus, indexes or model weights (`data/` is gitignored,
~13 GB). You need to build the corpus yourself with the CLI.

## Requirements

- Python 3.12+
- Apple Silicon Mac for the local-model engine (MLX). The quant core does not need MLX.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # quant core + tests
.venv/bin/pip install -e ".[engine]"     # adds MLX for the local model / retrieval

# The package is not installed in a way that adds src/ to the path; always set:
export PYTHONPATH=src

# SEC requires a contact address in the User-Agent for EDGAR requests:
export ARGUS_SEC_UA="ARGUS Research you@example.com"
```

## Usage

```bash
PYTHONPATH=src .venv/bin/python -m argus.cli --help
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Ticker universes live in `configs/universe/`. Scripts in `scripts/` are benchmarks and dataset
builders used during development.

## Disclaimer

This is research software, not financial advice. Nothing here recommends buying or selling any
security. Use at your own risk.

## License

[MIT](LICENSE)
