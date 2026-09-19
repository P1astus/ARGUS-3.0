# ARGUS 3.0

Local semiconductor/tech research assistant: a local model (Ministral-3-14B-Base, MLX) does
volume work — retrieve from the EDGAR corpus, extract verbatim-quoted claims, assemble
briefings — and a frontier model (Claude) does the judgment work conversationally. Decision
support, not automation; the user stays in the loop by design.

**Before doing anything, read `docs/HANDOFF.md` in full.** It is the living record of
architecture, every measured benchmark, decisions already made (don't relitigate them),
failed experiments with diagnoses, and gotchas that already cost hours. It is kept current
session-by-session — treat it as the source of truth over anything below, and over
`docs/NEXT_SESSION_PROMPT.md`, which predates it and is no longer updated.

## Non-negotiables (stable across sessions — see HANDOFF for the reasoning)

- **`PYTHONPATH=src` on every invocation.** The package is not pip-installed. venv is
  `.venv/` (Python 3.12).
- **Claude Pro, not Max** — do not design a nightly automated frontier batch. The frontier
  side is conversational by design.
- **Do not tune the Phase 1 quant gate to make it pass.** It failed three times; the cause
  is a diagnosed survivorship artifact, not a hyperparameter. Pre-registration discipline is
  agreed — further tuning is p-hacking. HANDOFF §4d.
- **Measure, don't assume.** This project has repeatedly found estimates wrong by an order
  of magnitude. Report failures plainly with the actual output; say so when a prediction
  turns out wrong rather than quietly moving on.
- **Ask before spending money or committing to anything multi-hour.**

## Other docs in `docs/`

- `retrieval_and_extraction.md` — full retrieval/extraction measurements behind HANDOFF §10-12
- `literature_review.md` — the academic literature this project's design is grounded in
- `phase0_model_selection.md` — model selection benchmarking detail behind HANDOFF §4a-4b
