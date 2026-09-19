# ARGUS 3.0 — Handoff

**Written 2026-08-02**, updated **2026-08-03** (session 2), updated again **2026-08-15**
(session 3). Everything below is either measured on this machine or a decision the user
made explicitly. Numbers are real, not estimated, unless flagged otherwise.

**Read §1 and §2 first.** The architecture changed materially during the first session,
and the original plan in `~/.claude/plans/i-m-starting-a-new-robust-backus.md` describes a
design that has since been superseded in one important respect.

**Session 2 built the retrieval + extraction loop** (semis only). Its design and
measurements live in `docs/retrieval_and_extraction.md` §1-7; §10 below is the summary.

**Session 3 extended it to 26 software/internet tickers** the user also trades — same
question as session 2 asked of itself: is this architecture actually general, or does it
just look general? Measured, not assumed: `docs/retrieval_and_extraction.md` §8, summary
in §11 below. Two real bugs were caught by the same measure-don't-assume discipline that
found MEMORY's "days of inventory" defect in session 2 — see §11c and the new gotchas in
§6.

---

## 1. What ARGUS is now

**A semiconductor research assistant with memory, provenance, and a track record.**

The pipeline splits by *volume vs judgment*:

```
overnight, local, free                        interactive, frontier
┌────────────────────────────────┐            ┌─────────────────────┐
│ Ministral-14B (local, MLX)     │            │ Claude (Pro)        │
│  • retrieve from 103M-token    │  Briefing  │  • analyses the     │
│    corpus                      │ ─────────► │    briefing         │
│  • extract Claims + verbatim   │  (2-5k     │  • converses with   │
│    quotes                      │   tokens)  │    the user         │
│  • diff filings, rank, shortlist│           │  • judgment calls   │
└────────────────────────────────┘            └─────────────────────┘
        ~50x compression: a 10-K is 73-100k tokens; a briefing is 2-5k
```

The local model does what machines are good at — reading everything, never tiring, never
skipping a filing. The frontier model does what needs judgment, on the handful that survive
the filter. **The user is in the loop by design** — this is decision *support*, not
automation.

### The pivot, and why it happened

The original design had a locally CPT-trained model *produce trade recommendations*
directly. Over the session the evidence against that accumulated (§4, §5), and the user
reframed it. The reframe is better-supported and the user articulated it themselves.

**Critically, the pivot re-targets CPT at something it can plausibly deliver.** CPT at
~100M tokens cannot install reliable factual recall or deep reasoning — those were what the
old design needed. What it *can* plausibly do is exactly the new local-model job: knowing
that book-to-bill matters for equipment and sell-through for fabless; spotting the one
inventory sentence that changed in 40 pages that didn't; not applying memory-cycle logic to
an analog name. That is extraction quality.

**Do not revert to "local model writes the recommendation."** The user considered and
rejected it on evidence.

### User context that shapes design

- **Claude Pro subscription** (not Max). Pro's usage limits are the binding constraint on
  the frontier side — which is *why* the local model does the batch work and the frontier
  model is used conversationally. Do not design a nightly automated frontier batch.
- Original spec said "local"; the user cares about that. Local-by-default, frontier for
  judgment.

---

## 2. Current state — what is built and working

**81 `.py` files** under `src/argus/` (recounted directly via `find`, not carried forward --
§12j's own note above about stale module counts applies to this line too, so it is
re-verified here rather than repeated from memory), all importing cleanly, **232 tests
passing** (13 leakage + 219 across retrieval, extraction, journal, corpus cleaning,
eval/LAP, training/sft+interleave, and the price/insider/macro/analyst context tools +
live briefing path, §12j-§12k).

| Area | Status | Key files |
|---|---|---|
| **Corpus** | ✅ **164,541,463 tokens built** (semis + tech + 20-F) | `engine/corpus/{build,catalog,clean,dedupe,pack}.py`, `sources/edgar.py` |
| **Retrieval index** | ✅ **488,400 chunks, lexical + dense, incremental** | `engine/retrieval/{chunk,store,embed,index,search}.py` |
| **Retrieval eval** | ✅ measured, hybrid wins, **0 probes skipped** (72/72) | `engine/retrieval/{probes,evaluate}.py` |
| **Extraction loop** | ✅ built, tested, domain-neutral prompt, sentence-boundary passage trim | `engine/extract/{quotes,claims,prompts,loop}.py` |
| **Briefing export** | ✅ built, tested | `engine/extract/export.py` |
| **Extraction baseline** | ✅ **RUN — semis §10d, tech §11e (v2 post-fix)** | `engine/extract/baseline.py` |
| **Contracts** | ✅ tested, 11 sub-segments (5 semis + 6 tech) | `contracts/{quant,briefing,recommendation,provenance}.py` |
| **Journal (Stage 6)** | ✅ **now actually tested** (12 tests) + CLI | `journal/{repository.py,schema.sql}`, `cli.py` |
| **Quant core (Stage 1)** | ⚠️ built, **gate FAILED** (semis only) | `quant/**` |
| **Training (Stage 5)** | ✅ built, never run. `SFTTrainer.train()` complete (§12i); SFT/CPT interleaving ready (untested end-to-end) | `engine/training/{cpt,checkpoint,data_iter,sft,sft_data,preflight,interleave}.py` |
| **Eval gate (Phase 3)** | ✅ built, tested. **LAP conditioning wired** (unmeasured — no CPT model) | `engine/eval/{gate,lap}.py` |
| **Tools (Stages 2-4)** | ✅ corpus retriever **+ live index-free retriever + price/insider/macro context** (§12j) | `tools/{provenance,taxonomy,briefing,price_context,insider,macro}.py`, `tools/retrievers/{corpus,live_edgar}.py` |
| **Funnel** | ✅ smoke-tested | `funnel/pipeline.py` |

**Not built yet:** a real CPT run of any kind, or any real SFT training pass (SFT is
architecturally meant to run after CPT — §12i — and no CPT pilot exists yet either). Quant
Stage 1 (`quant validate`) still runs only against `semis_v1.yaml` — `tech_v1.yaml` and
`foreign_filers_20f_v1.yaml` are corpus-only by design (§11a). SFT dataset (18 examples,
§12i) and 40-F/6-K support (§12h) both closed this session.

### Environment (all installed and verified)

- macOS 26.4, **Apple M5 Pro, 64GB**. Metal reports **55.7GB usable working set** (not 64).
- `uv` 0.12.1, Python 3.12.13 in `.venv/`, MLX 0.32.0, mlx-lm 0.31.3
- pandas 3.0.5, lightgbm 4.7.0 (needs `brew install libomp`), yfinance 1.5.2, scikit-learn
- Models downloaded: `Ministral-3-14B-Base-2512-4bit` (selected), plus 8B/14B/24B/32B/35B
  benchmark models and `phi-4`
- **Run everything with `PYTHONPATH=src`** — the package is not pip-installed.
- Postgres NOT installed; journal runs on SQLite (`Journal(dsn)` accepts either).

### CLI

```bash
PYTHONPATH=src .venv/bin/python -m argus.cli quant build-dataset --end 2026-07-31
PYTHONPATH=src .venv/bin/python -m argus.cli quant validate --features v2 --model gbdt
PYTHONPATH=src .venv/bin/python -m argus.cli engine corpus-build --skip-discovery
PYTHONPATH=src .venv/bin/python -m argus.cli engine corpus-status

# retrieval (index is already built; rebuilding is 11s lexical + ~35min dense)
PYTHONPATH=src .venv/bin/python -m argus.cli retrieve index --no-dense
PYTHONPATH=src .venv/bin/python -m argus.cli retrieve search "book-to-bill" --ticker AMAT --as-of 2024-01-01
PYTHONPATH=src .venv/bin/python -m argus.cli retrieve evaluate

# extraction
PYTHONPATH=src .venv/bin/python -m argus.cli extract brief NVDA --as-of 2024-05-01 --model base
# baseline is resumable; run FOREGROUND in slices (~140-155s/case)
PYTHONPATH=src .venv/bin/python -m argus.cli extract baseline --arms base --budget-seconds 300 -v
PYTHONPATH=src .venv/bin/python -m argus.cli extract baseline --arms base --cases tech --budget-seconds 300 -v

# tech universe (corpus-only; NOT wired into `quant`)
PYTHONPATH=src .venv/bin/python -m argus.cli extract brief CRM --as-of 2024-05-01 --universe configs/universe/tech_v1.yaml --model base
```

---

## 3. The corpus (built, ready)

**Historical record of the session-1 build. Current totals are §2's table
(164,541,463 tokens / 54,392 documents) — tech tickers (§11) and 12 twenty-F foreign
filers (§12a) were added later, additively, on top of everything below.**

**103,460,398 tokens · 18,009 documents · 2010-01-04 → 2025-12-30**

- `data/corpus/shards/` — 393MB, memmap `.npy` shards, packed at seq 4096
- `data/corpus/catalog.sqlite` — 21MB, the resumability backbone; every stage queries it
- `data/corpus/manifest.json` — token counts, dedup rate, hashes, cutoff

**Cutoff is 2025-12-31, enforced by `Catalog.assert_cutoff()`.** Chosen to match the base
model's own knowledge cutoff — an earlier cutoff would sacrifice training tokens for eval
cleanliness we would not actually gain, because the base model already read everything
before its own cutoff. **The genuinely clean eval window is 2026-01 onward regardless.**

**Dedup came in at 18.4% within-issuer** (4,050 near-duplicates). Predicted 25-40% — the
direction was right (an order of magnitude above the reference study's 1.9%, because our
corpus is narrow-and-deep rather than wide-and-shallow) but the magnitude was overestimated.

**Rebuild cost if ever needed:** ~2.5h (discovery ~1h, fetch ~1.8h at 8 req/s, clean/dedupe/
pack ~45min). Use `--skip-discovery` to reuse the catalog.

---

## 4. Measured findings — do not re-derive these

### 4a. Hardware / model selection (Phase 0)

**MoE trains dense.** The plan estimated 250-600 tok/s for Qwen3.6-35B-A3B from its *3B
active parameters*. Wrong by ~10× — backward propagates through the expert stack, so MoE
sparsity buys cheap inference, not cheap training.

| Model | Arch | seq | tok/s | Peak mem |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | MoE | 4096 | **OOM** | — |
| Qwen3.6-35B-A3B | MoE | 512 | 55.5 | 28.8 GB |
| DeepSeek-R1-Distill-8B | dense | 4096 | 208.1 | 16.0 GB |
| **Ministral-3-14B-Base** | dense | 4096 | **137.0** | 23.7 GB |
| Qwen3-14B | dense | 4096 | 118.4 | 22.6 GB |
| Mistral-Small-3.1-Text-24B | dense | 4096 | 85.2 | 25.9 GB |
| Qwen3-32B | dense | 4096 | 50.4 | 34.9 GB |

**Scaling rule (use this instead of re-benchmarking a new model):**
```
tok/s     ≈ 1660 / params_in_billions     (±20% for architecture; Mistral beats trend)
peak_GB   ≈ params_B / 2 + 12…19          (4-bit, seq 4096)
```
Predicted vs measured: 8B 208/208.1, 14B 119/118.4, 32B 52/50.4. **Practical ceiling ~48B
dense at seq 4096; ~32B at seq 8192; 70B does not fit.**

**Gradient checkpointing is not optional** — without it, 84.6GB peak (swaps) and 3.9 tok/s.
With it, 28.8GB. Already forced on in `cpt.py`.

**8192 context is affordable** — costs only ~18% wall-clock, 39GB. Relevant to chunking:
at 8k a full Risk Factors section usually fits intact. `ChunkSpec.window` is a config field.

**Thermal: −0.1% over 15 minutes.** The plan's multi-day throttling warning is unsupported
at that timescale; re-check during a long run.

### 4b. Model choice: Ministral-3-14B-Base

Measured **bits-per-character on 100k chars of held-out 2026 10-Qs** (NVDA/AMAT/MU/ADI —
fabless/equipment/memory/analog). BPC not perplexity, because tokenizers differ by up to
23% on filing text and perplexity is per-token.

| Model | Type | BPC | chars/token |
|---|---|---|---|
| **Ministral-3-14B-Base** | base | **0.3394** | 3.168 |
| Ministral-3-14B-Instruct | instruct | 0.3524 | 3.168 |
| Mistral-Small-24B | instruct | 0.3307 | 3.168 |
| phi-4 | post-trained | 0.3691 | 3.912 |
| Qwen3-14B | instruct | 0.3749 | 3.249 |
| DeepSeek-R1-Distill-8B | distill | **0.7656** | 3.847 |

- **Instruct penalty measured at 3.8%** by running Ministral in both variants — so the
  base/instruct confound is real but small, and doesn't explain Qwen's 10.5% deficit.
- **The 8B is disqualified as a domain model** — 126% worse, token ppl 7.71 vs 2.11. That is
  distributional (R1 distillation optimises for reasoning traces), not a size effect. Fine
  for plumbing tests; do not use its loss curves to infer anything about domain behaviour.
- **Selected Ministral-3-14B-Base** on three grounds that agree: best domain BPC, the only
  true *base* checkpoint at 14B (no CPT-on-post-trained degradation risk), fastest 14B
  measured. Apache-2.0, 262k native context.
- `gemma-4-12B` is **blocked, not judged** — mlx-lm 0.31.3 rejects `gemma4_unified`.

### 4b-addendum. Qwen3.8-27B re-examined 2026-08-16 — selection stands

Released 2026-08-14 (two days before this evaluation), user-proposed as a candidate. Given
the full Phase 0 sequence rather than judged on release-day benchmarks, because SWE-Bench
and GPQA scores say nothing about domain BPC or whether this project's training pipeline
can actually run it.

**Domain BPC: measured best in this project**, same corpus, same methodology, no new
fetch:

| Model | Type | BPC | vs. Ministral-Base |
|---|---|---|---|
| **Qwen3.8-27B** | instruct-only | **0.3111** | **−8.3%** |
| Mistral-Small-24B | instruct | 0.3307 | −2.6% |
| Ministral-3-14B-Base | base | 0.3394 | — |

The gap is 4x past this project's own ~2% quantisation-noise floor, so it is real. It is
also a stronger result than the number alone suggests: this project's own measured
instruct-tuning penalty is 3.8% (Ministral-Base 0.3394 → Ministral-Instruct 0.3524), and
Qwen3.8-27B is instruct-only, yet still beats every base checkpoint tested by a wider
margin than that penalty. `artifacts/bpc_qwen38_27b.json`.

**Architecture compatibility: passes, and the initial caution based on the gemma-4-12B
precedent was wrong.** Qwen3.8-27B is a hybrid architecture (64 layers: 48 Gated
DeltaNet linear-attention + 16 standard grouped-query attention, 3:1 ratio, model_type
`qwen3_5`) bundled as a vision-language checkpoint. Verified directly against the
installed mlx-lm 0.31.3, not assumed from documentation:
- `mlx_lm.utils.load()` loads the VLM-packaged checkpoint correctly (extracts the text
  tower; no vision-loading errors) — 180s cold, 1.2s cached.
- `linear_to_lora_layers()` is fully generic (confirmed by reading its source, not just
  running it) — it walks `named_modules()` and swaps in `nn.Linear`/`nn.QuantizedLinear`
  instances by dotted key, with no architecture-specific logic. Both layer types expose
  real `nn.Linear` submodules (`GatedDeltaNet.{in_proj_qkv,in_proj_z,in_proj_b,
  in_proj_a,out_proj}`, `Qwen3NextAttention.{q,k,v,o}_proj`), so both are LoRA-attachable
  with **explicit keys covering both** — the default `ATTENTION_ONLY_KEYS` in
  `bench_mlx.py` only targets the 16 standard-attention layers and would have silently
  left the 48 Gated DeltaNet layers unadapted.
- Full forward + backward + optimizer step succeeds, including real autodiff through the
  custom recurrent SSM computation (the genuine remaining risk after module-wrapping
  succeeds — this is where a custom architecture most often breaks even after loading
  fine).

**Throughput and memory at real training config: FAILS, decisively.** This is where the
candidate is disqualified.

| config | result |
|---|---|
| seq_len 4096 (this project's real CPT config), rank 16, grad checkpoint, `--no-compile` | **crashes**: `RuntimeError: [metal::malloc] Resource limit (499000) exceeded` |
| seq_len 512, same config | runs: **18.1 tok/s**, 26.5GB peak |
| seq_len 512, `mx.compile` enabled | runs: **22.4 tok/s**, 29.0GB peak (compile helps only +24%) |

Confirmed the crash is seq_len-dependent (512 works, 4096 does not) — the graph this
hybrid architecture generates for gradient checkpointing exceeds a Metal *resource-count*
ceiling at production sequence length, a different failure mode from every memory-capacity
issue found earlier in this project (§4a's 84.6GB OOM, session 2's 49GB buffer-cache
stall). `mx.compile` — which every other benchmark in this project disabled, because it
hung on a *different* model's MoE routing (Qwen3.6-35B-A3B, §6) — was worth re-testing
here since a recurrent/SSM architecture depends on graph fusion far more than standard
attention does. It helped, marginally, and did not remotely change the verdict.

Even the *best* measured configuration (22.4 tok/s, and not even at the sequence length
actually needed) is **13x under this project's own exit criterion** (<300 tok/s → stop and
revisit, §4a) and **6x slower than Ministral-14B-Base's measured 137 tok/s at the seq_len
that actually matters**. Projected epoch time: 51.8 days at the best case, versus
Ministral's measured 8.45 days. `artifacts/bench_qwen38_27b_seq512.json`,
`artifacts/bench_qwen38_27b_seq512_compiled.json` — the seq_len=4096 crash produced no
artifact (the script fails before writing output); the traceback is reproduced above in
full because it is the record.

**Read plainly:** this is very likely a "too new" problem, not a "wrong model" problem.
mlx-lm added real architecture support within two days of release (notable on its own),
but Metal kernel optimisation for a genuinely novel recurrent op type lags architecture
support, and it shows. **Worth re-checking in a few months if MLX's Gated DeltaNet kernels
mature — not worth pursuing for the pilot now.**

**Selection stands: Ministral-3-14B-Base.** Its BPC is no longer the best measured in this
project, but it is the only candidate that is simultaneously fast, fits the real training
config, and has a genuine base checkpoint. Re-litigating this again requires a new
candidate to clear the *throughput* bar, not just the BPC one — BPC alone was never
sufficient, which is exactly why Phase 0's sequence has three more gates after it.

**Infrastructure note:** `scripts/bench_mlx.py` gained an optional `--keys` override
(default behaviour for every prior invocation is unchanged) specifically because of this
evaluation — any future hybrid-architecture candidate will need the same explicit,
non-default key list this one did, and the flag is now there rather than needing to be
re-added under time pressure.

### 4c. LoRA rank 256 is free — and rank 16 would have been an underpowered experiment

[LoRA Learns Less and Forgets Less (arXiv 2405.09673)](https://arxiv.org/abs/2405.09673):
standard low ranks substantially underperform full finetuning **for continued pretraining
specifically**; r≈256 is needed to approach it (at 20B tokens on code CPT, r=256 scored
0.617 vs full-FT 0.545).

Measured on Ministral-14B at seq 4096:

| rank | trainable | tok/s | peak mem |
|---|---|---|---|
| 16 | 19.7M | 135.9 | 23.7 GB |
| 256 | 314.6M | **136.0** | 27.2 GB |

**Throughput is flat across a 16× rank increase.** `CPTConfig.lora_rank` is already 256.
The `preflight.py` ceiling was raised 5% → 15% to permit it (the guard fired correctly; the
threshold was calibrated for the old rank).

### 4d. Phase 1 quant gate — FAILED, and the failure is informative

Three attempts, all failed, all consistent:

| Attempt | Pooled IC | Excluding the artifact fold |
|---|---|---|
| v1 GBDT (11 features) | +0.0212 | +0.0074 |
| v2 GBDT (18 features) | +0.0067 | ~+0.002 |
| core Ridge (5 features) | +0.0181 | **−0.0007** |

**IC correlates with universe coverage at −0.74.** The one fold carrying nearly all apparent
signal (2013-2015, IC 0.0764) is the fold where **56% of the real cross-section is missing**.
That is the signature of a survivorship artifact, not alpha. `report.py` now emits this
diagnostic automatically.

**Diagnosis: massive overfitting.** In-sample IC 0.202 vs out-of-sample 0.022 (v1); v2 made
in-sample *better* (0.247) and out-of-sample *worse* (0.006). The panel looks like 31,277
rows but rows sharing a date are one correlated cross-section — **effective sample is ~826
weekly observations**.

**The `contracts.QuantSignal` is therefore a weak prior, not a validated signal.** Anything
consuming it should treat it that way; `funnel/pipeline.py` emits a coverage warning.

⚠️ **Pre-registration discipline was agreed with the user.** Three attempts already risk
p-hacking. **Do not tune further to make the gate pass.** If revisiting, fix the data first
(§7) and pre-commit to the metric before running.

### 4e. Literature review — see `docs/literature_review.md`

The most important findings, because they bound what to expect:

- **[KTD-Fin (arXiv 2605.28359)](https://arxiv.org/html/2605.28359v1)** — the only study
  found that properly controls memorisation (four-level ticker/date masking, verified with
  de-anonymisation probes). **Nine of ten agents post negative selection alpha.** Their
  anchor model returned −0.16% with visible tickers and exactly 0.00% (pure cash) when
  anonymised — its trading was driven by pretraining memory.
- **[Lookahead Propensity (arXiv 2512.23847)](https://arxiv.org/abs/2512.23847)** — LAP is
  materially positive in-sample and **collapses to zero right after the training cutoff**.
  Lopez-Lira (author of the original optimistic ChatGPT-forecasting paper) now argues
  post-cutoff evaluation is mandatory. **LAP is cheap to implement and should be added to
  the eval harness.**
- **Field evidence quality**: of 19 audited studies, **2/19** report time-consistent splits,
  **1/19** transaction costs, **1/19** survivorship handling. ARGUS already does all three.
- **[Llama-Fin (arXiv 2501.04961)](https://arxiv.org/html/2501.04961v3)** — **CPT alone
  collapses instruction-following from 7.8 to ~1.0 on MT-Bench**; *joint* CPT+IT prevents
  it. **Our `cpt.py` → `sft.py` is sequential, which is the worse configuration.** Mitigation
  is to interleave SFT-format examples into the CPT stream. Their corpus was 6B tokens;
  ours is 0.1B (~60× smaller).
- **TradingAgents (the paper the user shared)**: Sharpe 8.21 on 3 stocks over 60 days, which
  the authors themselves flag as implausible; GPT-4o's training covers the eval window. Good
  *architecture* reference (their structured-document argument is why we have `contracts`);
  not evidence of returns.

---

## 5. Decisions already made — do not relitigate

| Decision | Choice | Why |
|---|---|---|
| Quant universe | ~100 names, 2010+ | 15 names makes rank IC unmeasurable (per-day σ ≈ 0.25-0.30) |
| Price data | yfinance, caveats accepted | Free; **no delisted history** — see §7 |
| Base model | Ministral-3-14B-Base | §4b; re-examined against Qwen3.8-27B 2026-08-16, stands — §4b-addendum |
| LoRA rank | 256 | §4c — free |
| Corpus cutoff | 2025-12-31 | Matches base model cutoff; earlier buys nothing |
| Corpus scope | EDGAR + planned FedReg/GovInfo/CourtListener/Wikipedia | Patents excluded (volume trap — H01L is 4.8B tokens of device physics, ~98% of gradient would train a physicist not an analyst) |
| Transcripts | Excluded | Third-party copyrighted; 8-K prepared remarks measured at only ~2-3M tokens (1 of 10 issuers files them) |
| Eval design | Corpus time-cut + 3-arm | Arm B (base+prompt, no CPT) is essential or you measure the prompt |
| Labels | Equal-weight **leave-one-out** benchmark | NVDA can exceed 10% of cap-weighted SOXX and would compress its own relative return |

---

## 6. Gotchas that already cost time

- **`PYTHONPATH=src` on every invocation.** Not pip-installed.
- **YAML parses bare `ON` as boolean `True`** — the ticker ON Semiconductor. All symbols in
  `configs/universe/semis_v1.yaml` are quoted; `load_universe()` raises on non-str.
- **lightgbm needs `brew install libomp`** on macOS or import fails.
- **pandas 3.0 returns read-only arrays from `to_numpy()`** — copy before mutating
  (bit `shuffle_test`).
- **`uv venv` ships no pip** — use `uv pip install`.
- **EDGAR 8-K primary documents are cover pages** (~0.5-1.1k tokens). The substance is in
  EX-99 exhibits, reachable only via each filing's `index.json`. Exhibit naming is
  inconsistent — NVDA's is `q1fy27cfocommentary.htm`, which an `ex.?99` filename filter
  misses. Also exclude full-submission `.txt` dumps (they duplicate every exhibit + XBRL).
- **Stooq is behind a JS proof-of-work challenge** — dead end for scripted free data.
- **`AVGO` pre-2016 is Avago, not Broadcom Corp.** Using it as BRCM's history splices two
  different companies. Only `SGH → PENG` is a genuine continuation (a rename).
- **`mx.compile` on a 40-layer 256-expert MoE appears to hang** — `bench_mlx.py` has
  `--no-compile`, and all recorded benchmarks used it.
- **mlx-lm 0.31.3 supports MoE expert LoRA** (`LoRASwitchLinear`), so
  [#571](https://github.com/ml-explore/mlx-lm/issues/571) is fixed. The live hazard is the
  inverse: `keys=None` auto-discovers every Linear/SwitchLinear. **Always pass explicit
  `keys`.** `preflight.py` guards both directions.
- **Variable padding lengths blow up MLX's buffer cache.** With `padding=True` each batch
  pads to its own longest member, so every batch requests a new tensor shape, and MLX
  caches Metal buffers per shape and never reuses them. Measured: **the cache grew to
  48.9GB in 20 seconds** against the 55.7GB working set, after which throughput collapsed
  from 130 chunks/s to under 1 — a 35-minute job becoming a multi-day one. Symptom is
  ~60s of full speed then an apparent hang; three wrong diagnoses (bad input data, slow
  tokenizer, OS scheduling) were chased first. **Fix: bucket sequence lengths to a multiple
  of 64 before padding, and set `mx.set_cache_limit()`.** See `engine/retrieval/embed.py`.
  Same family as the gradient-checkpointing finding in §4a: exceeding the Metal working set
  does not fail loudly, it just gets slow enough to look like the work is hard.
- **Long MLX jobs run dramatically slower when launched detached.** With the cache bug
  fixed, an A/B on an identical 2,048-chunk slice measured **143.6 chunks/s foreground vs
  34.7 backgrounded** (4.1x), reproduced three times on the embedder. On the 14B extraction
  model the gap was far worse: **138 s/case foreground vs 5,677 s detached (~18x)** for
  identical work. Mechanism not isolated; `taskpolicy -c` offers no clamp that raises
  priority. **Run long MLX work in the foreground, in slices.** Both
  `build_dense(budget_seconds=...)` and `extract baseline --budget-seconds` stop cleanly at
  a checkpoint and resume, which exists precisely because of this.
- **FTS5 query terms must be individually quoted.** Bare `-` and `.` are FTS5 syntax, and
  this corpus is full of `book-to-bill`, `10-K`, `R&D` — an unquoted query raises
  `fts5: syntax error` on entirely ordinary analyst input. `search.fts_query()` handles it.
- **`PROFILES[MEMORY].key_metrics` listed `days of inventory`, which MU/WDC/STX never
  write.** It is analog/MCU vocabulary (MCHP 181 chunks, SLAB 77, TXN 76). Memory issuers
  discuss inventory as write-downs and carrying value. Since taxonomy metrics generate the
  retrieval queries, this was sending the extraction loop looking for language that is not
  there. **Fixed in session 2** — see `taxonomy.py`'s MEMORY profile for the corrected
  metrics and the measurement behind them.
- **The corpus has zero foundry coverage.** TSM/UMC/GFS have **0 chunks** — they file 20-F
  as foreign private issuers and the EDGAR builder collects 10-K/10-Q/8-K only.
  `SubSegment.FOUNDRY` will frame a briefing that retrieval cannot fill. **Fixed in session
  3, §12a**: 20-F support built and 12 tickers backfilled (+15.7M tokens); TSM/UMC/GFS now
  have real coverage. SHOP turned out to be a milder version of the same class of gap
  (converted from 40-F/6-K to domestic 10-K/10-Q filer status only in 2025-02, so it
  genuinely has zero pre-2025 10-K/10-Q history — not a bug, but a real limit of what
  20-F support fixes) and IFNNY has a permanent gap (its filings all predate the corpus's
  2010 start). **Before picking a baseline case or building a new universe file, check**
  `SELECT MIN(doc_date), MAX(doc_date) FROM chunks WHERE ticker=?` — a systematic sweep
  now exists and found all of the above in one query; don't rediscover them one case at a
  time.
- **A word-overlap trace, not a prefix match, is required to diagnose a fabrication
  correctly.** Session 3's first attempt to explain a CLOUD_SAAS fabrication (search the
  cited source for the first ~30 characters of the claimed quote, read what's nearby)
  concluded the model was substituting a "generic textbook definition" for what the filing
  said. That was **wrong** — the real text was 96% identical to the "fabrication," missing
  one word ("Our"). The prefix-match method simply landed near a different, unrelated
  sentence. Correct method: for each word-length from longest to shortest, search for that
  contiguous run anywhere in the cited document; report the best match's word-overlap
  ratio. Confirmed the corrected mechanism (near-verbatim drift, not substitution) across
  three independent cases this way. See `docs/retrieval_and_extraction.md` §8g/§9b.
- **A form-specific header pattern needs a form-specific disambiguation rule, not just a
  form-specific regex.** 20-F filings cite their own item numbers constantly in body text
  ("see Item 4 for further discussion"), and `find_sections`' "last occurrence wins"
  heuristic — correct for 10-K, which rarely self-cites this way — picks a cross-reference
  over the genuine header more often than not. The real header is reliably ALL-CAPS in the
  source HTML; cross-references are not. Fixed with an opt-in `prefer_caps` flag, scoped
  to 20-F so the 10-K path (18,000+ already-processed documents) is untouched.
- **A passage that ends mid-sentence invites the model to complete it from memory, not
  from what it was shown.** Measured: 67% of sampled passages (before a fix) ended without
  terminal punctuation, because `chunk.expand()`'s end-boundary snap only fires when a
  newline falls within the last 400 characters, which dense running prose often lacks. One
  traced case showed the model completing a truncated "...valuation allowance of $" with
  two specific dollar figures present nowhere in the shown text. Fixed with a bounded
  backward search (300 chars, never below the hit's own end) for the nearest sentence
  terminator. Down to 13% after the fix; the fabrication rate moved modestly (0.057→0.045)
  because this was only ever one of several fabrication mechanisms, not the dominant one.
- **A module can be relied upon in production language ("tested end-to-end") while having
  zero actual test coverage.** True this session for the journal (`journal/repository.py`),
  the Phase 3 gate (`engine/eval/gate.py`), and narrative extraction
  (`engine/corpus/clean.py`) — all three had been running (or were assumed reliable)
  without a single test file. None had a real bug once tested, but the absence was
  unverified, not verified-clean. Worth checking before trusting a "done" claim in this
  document against anything that actually processes real money or real filings.
- **`np.memmap` cannot be resized in place**, which matters once the corpus is expected to
  grow more than once. `ChunkStore.grow_vectors()` (session 3) writes a new, larger file,
  copies the existing rows across, and atomically replaces the original — the pattern to
  reuse rather than reinvent if the vector store needs to grow again.
- **A prompt can be more domain-specific than the taxonomy that frames it, and grepping
  for the domain enum will not catch it.** Session 3's taxonomy correction pass (measure
  real vocabulary, fix what is wrong) would not by itself have caught
  `engine/extract/prompts.py`'s system rules literally saying "extracting claims from
  semiconductor filings" — that surfaced only from reading the file end to end before
  running it on an off-domain ticker. When generalising a component, read every prompt
  string it emits, not just the structured config around it.

---

## 7. The $30 question (unresolved, user-facing)

yfinance serves no delisted history. ~20 economically significant semis exited 2010-2026
(XLNX, MXIM, IDTI, LLTC, ALTR, BRCM, CY, ISSI, ATML, FCS, PMCS, CAVM, MLNX, INPHI, MSCC,
NANO, RTEC, SMI…) and they are **disproportionately acquisition targets — i.e. winners**.
Registry with exit dates/reasons is in `data/delisted.py`.

**Cheapest fix that works: ~$30 ONE-TIME, not a subscription.** The need is ~20 dead tickers
× ~12 years ≈ 60,000 rows ≈ 3.6MB, and it is **static forever** (dead tickers get no new
data). Subscribe to EODHD's **$29.99 "EOD+Intraday All World Extended"** tier (the $19.99 EOD
tier **excludes** delisted), download, cancel. `PriceStore` caches to parquet permanently, so
the data outlives the subscription. Forward/live data stays free on yfinance — survivorship
only affects *historical* testing.

Adding an EODHD adapter is one file behind the existing `PriceProvider` Protocol; nothing
downstream changes.

**What it buys:** the single most diagnostic test available — does fold 0's IC survive when
the missing names are added back? Given §4d, expect it to collapse. That is still worth $30.

---

## 8. What to do next

**Steps 1-4 of the original plan are done** (session 2), **extended to tech, and every
cheap fix identified along the way has been closed out** (session 3, §12). What's left is
almost entirely the pilot CPT run itself, which is deliberately on hold.

1. ~~Corpus retrieval/indexing~~ — ✅ 488,400 chunks (semis + tech + 20-F), hybrid measured, **0 probes skipped**.
2. ~~Local extraction loop~~ — ✅ quote-verified, domain-neutral prompt, sentence-boundary passage fix, 150 tests green.
3. ~~Briefing export~~ — ✅ `artifacts/baseline/base/*.md`, `artifacts/baseline_tech/base/*.md`.
4. ~~Baseline extraction quality on the untrained Ministral~~ — ✅ **semis: §10d, tech:
   §11e (v2, post passage-truncation fix). Both are denominators; do not re-run either
   against a changed loop without re-baselining first.**
5. ~~Section weighting for SaaS~~ — ✅ measured (§12d): no change needed, already better
   than semis' own number.
6. ~~Foundry / 20-F coverage gap~~ — ✅ closed (§12a): 12 tickers backfilled, +15.7M
   tokens. One real, permanent gap remains (IFNNY, pre-2010 only) and one dead-ticker
   config bug fixed (SGH → PENG).
7. ~~Confirm the CLOUD_SAAS fabrication hypothesis against a second case~~ — ✅ done, and
   the original hypothesis was **wrong** — corrected to near-verbatim word-drop, confirmed
   against a third, independent case (§11e, §12b).
8. **The pilot CPT run is the only major item left, and it is intentionally on hold** — the
   user is evaluating a newer candidate base model (Qwen3.8-27B) before committing to
   which checkpoint the pilot runs against. Whenever it proceeds, two design questions:
   - **Corpus scope: CPT on semis-only tokens, or the full semis+tech+20-F corpus?**
     Training on the combined corpus makes the model less specialised per-token at fixed
     budget; semis-only leaves the tech/foundry extraction loop permanently on the
     untrained checkpoint.
   - **What to pre-register as the target metric.** The clearest, sharpest candidate
     found across both baselines: the near-verbatim word-drop rate on long sentences
     (§11e/§12b) — a real, reproduced-three-times mechanism with a plausible CPT story
     (more exposure to verbatim copies of these exact filings might improve copy
     fidelity). The 1.5% inference-claim rate (§10d) is the other standing candidate.
     Whether CPT helps either is genuinely unknown; both are measurable against the
     existing baselines once a run exists.

⚠️ **Build the loop before the pilot.** Satisfied since session 2, and every identified gap
in it closed out in session 3 (§12) rather than left as a caveat next to the pilot decision.

⚠️ **Each baseline measures the loop as configured when it ran.** The tech baseline was
re-run once already (§12c) after a real fix changed the loop; if anything else changes
before the pilot, re-baseline again rather than reasoning around the mismatch.

**In parallel, free, and valuable regardless:** start paper trading via the journal
(`argus journal record`/`close`/`status`, §12e — now tested, previously was not despite
the state table's earlier claim). Forward-recorded decisions are the only evidence immune
to the contamination that invalidates most of the published literature (§4e) — every day
not started is a day of uncontaminated evidence not accumulating. No trade has been
recorded yet; inventing one to "start" the record would defeat the point of prospective
evidence.

**Also outstanding, lower priority:** the $30 delisted-price question (§7, unrelated to
any of this). An 18-example SFT dataset now exists (§12i) — small (a real SFT set is
typically hundreds to thousands of examples; this is the 18 the existing baseline cases
yielded, not a swept target), but the interleaving machinery (§12f) is no longer blocked
on having literally nothing to interleave. 40-F/6-K support for SHOP-class gaps closed
(§12h).

---

## 9. Honest expectations

State these plainly if the user asks; they were established with evidence during the session
and the user has engaged with them directly.

- **Alpha is unlikely.** KTD-Fin found negative selection alpha under proper memorisation
  control; our own Phase 1 found ~zero once the survivorship artifact is removed.
- **What is achievable** is a disciplined research assistant with memory, verified citations,
  forced pre-commitment to entry/target/invalidation, and a calibrated track record. None of
  that requires the market to be beatable, and all of it is mostly built.
- **CPT's realistic ceiling** at 103M tokens with attention-only LoRA: native sub-segment
  vocabulary and framing, better judgement about *which* filing details matter. Not reliable
  new factual recall — that is what retrieval is for.
- **The measurement apparatus is the most valuable thing built.** It correctly detected that
  there is no signal, where most of the field's published work does not measure well enough
  to notice. Do not weaken it to get a nicer number.

---

## 10. Session 2 — retrieval and extraction (summary)

Full detail in `docs/retrieval_and_extraction.md`. The three things worth knowing here:

### 10a. The index

**306,470 chunks from all 18,009 documents.** Lexical (FTS5 BM25) builds in **11 seconds**;
dense (`bge-small-en-v1.5`, 384 dims) takes ~35 min at ~145 chunks/s. 257MB + 235MB.

The load-bearing invariant: **a chunk is `clean_doc[start:end]` and nothing else.** No
stripping, no whitespace collapsing. If chunking rewrote the text, a faithful quote would
fail verification and an unfaithful one would be indistinguishable from it. `ChunkStore`
holds offsets and reads through to `data/corpus/clean/` rather than storing text.

`as_of` is enforced as a **hard SQL filter upstream of scoring**, not a ranking preference.
A passage published after the decision date is not a candidate.

### 10b. Retrieval — hybrid wins, dense alone does not

33 rule-defined probes (gold = an anchor pattern plus a digit, so neither retriever has
access to the answer key), split into `direct` (query uses the domain term) and
`paraphrase` (query avoids it entirely).

| mode | R@5 | R@20 | MRR | direct R@20 | **para R@20** |
|---|---|---|---|---|---|
| bm25 | 0.576 | 0.727 | 0.392 | 0.944 | 0.467 |
| dense | 0.667 | 0.758 | 0.424 | 0.944 | 0.533 |
| **hybrid** | **0.727** | **0.848** | **0.435** | 0.944 | **0.733** |

All three **tie exactly on `direct`** — the expectation that BM25 would win term-matching
queries was wrong, so that family only disqualifies, it does not rank. The whole result is
paraphrase: **+27 points for hybrid**, of which dense alone contributes only +6.6. RRF acts
as a **union**: on the 6 probes where the two disagree they win 3 each, and hybrid recovers
5 of 6. **Keep the dense index, but it is justified by fusion, not on its own.**

### 10c. Extraction — honest, not yet useful

Quote handling works: a quote is **located**, never trusted. Accepted claims store the
*document's* bytes, so `Briefing.unverified_quotes()` is empty by construction; a quote that
resolves nowhere is **dropped, never downgraded to INFERENCE** (which would launder a
fabrication into acceptable output). Smoke tests hit 100% and 93% exact-quote rates with
0 and 1 fabrication, audit PASS.

**But the metrics were flattering a broken loop, and only reading the output caught it:**

1. **At 5 passages per generation the model spent its whole token budget on the first one**
   and never reached the rest — **the audit reported 100% exact quotes on a briefing that
   ignored 8 of 10 retrieved passages.** A provenance metric can look perfect precisely
   *because* evidence was skipped. Fixed (`batch_passages=1`); coverage is now structural.
2. **Boilerplate crowds out news.** Risk factors match any query lexically and are the most
   recycled text in a filing. `max_per_section=3` caps them, but the mix is still
   boilerplate-heavy against only 1 MD&A passage. **Section weighting needs revisiting.**
3. **Zero inference claims, in both runs.** The base model only copies — many claims are
   near-verbatim restatements of their own quote, no compression, no judgment, despite the
   taxonomy framing being in the prompt. **This is the most direct evidence yet of what CPT
   would have to change.**

All three were fixed before the baseline ran, except (3), which is a property of the model.

### 10d. THE BASELINE — the CPT denominator

**10 cases, all five sub-segments, `Ministral-3-14B-Base`, untrained. 23 min.**
`artifacts/baseline/baseline.{txt,json}`, briefings in `artifacts/baseline/base/`.

| metric | value |
|---|---|
| claims accepted | 330 (3.44 per passage) |
| **exact quote rate** | **0.910** (range across cases **0.77 – 1.00**) |
| whitespace-repaired | 13 |
| **fabrication rate** | **0.023** — 8 claims dropped |
| quote too short | 10 |
| sourced fraction | 0.985 |
| **inference claims** | **5 of 330 (1.5%)** |
| malformed JSON / missing-field lines | **0 / 0** |
| quote audit vs retained bytes | **10/10 PASS** |

Three things to carry forward:

1. **Format validity is perfect from a base checkpoint** — 0 malformed lines in 343 quote
   attempts. This was the biggest open risk and it is not a problem; the few-shot JSON
   Lines format handles it. It also means the Llama-Fin instruction-following worry (§4e)
   does not bite for *this* task, and the instruct arm was consequently not run.
2. **The fabrication guard earns its place.** 8 fabricated + 10 too-short = **18 of 343
   attempts (5.2%) would have reached the frontier model as unverifiable assertions
   dressed as quotations.** Nothing downstream could have caught them.
3. **The model does not reason — 1.5% inference claims, and the true rate is lower.**
   Of the 5, at least one is verbatim revenue-recognition boilerplate copied out of a
   filing and mis-tagged `inference`. The genuine ones are decent ("Revenue guidance is
   down 20% from Q2 2022, but operating expenses are down 10%"), but there are ~4 of them
   across 10 briefings. The sub-segment framing is in every prompt and produces almost no
   cycle judgment. **This is the sharpest available target for CPT**, far sharper than
   "extraction quality" in the abstract.

⚠️ **Measure CPT per-case, not on the aggregate.** The exact-rate spread is 0.77 (CRUS) to
1.00 (AMAT/AMD); an aggregate move of two points could be one case moving.

Cost: retrieval 45-55 ms/query; a 10-passage briefing on the base 14B is **138 s** typical.

---

## 11. Session 3 — extending to 26 software/internet tickers (summary)

Full detail in `docs/retrieval_and_extraction.md` §8. User asked whether the same
architecture could cover the tech stocks traded alongside semis. Answer, measured rather
than assumed: **yes — 85% of the retrieval/extraction code has zero semiconductor-specific
logic**, and the domain-specific 15% (taxonomy, retrieval probes, one prompt string) took
one session to extend and correct.

### 11a. Scope

26 tickers, 6 new `SubSegment` categories (AD_PLATFORM, DEVICE_ECOSYSTEM, CLOUD_SAAS,
CYBERSECURITY, COMMERCE_PLATFORM, STREAMING_MEDIA), in `configs/universe/tech_v1.yaml` —
**corpus-only, deliberately not wired into `quant build-dataset`/`quant validate`**, which
stay scoped to `semis_v1.yaml` per the decoupling principle in §5.

### 11b. Corpus and index, additive

~50 min to build: 6,109 filings discovered, 5,857 fetched, 5,982 packed, **45,338,624 new
tokens** (dedup rate 21.6%, higher than semis' 18.4%). Corpus total: **148,801,271
tokens, 23,991 documents.**

Indexing had to be made genuinely incremental first — the naive approach would have
re-embedded the 306,470 already-done semis chunks (~35 min wasted). `ChunkStore.
grow_vectors()` and `build_lexical(reset=False)` fixed that; verified on the real corpus,
the incremental dense pass covered only the 136,300 new chunks (~12 min) instead of a full
442,770-chunk rebuild. Index total: **442,770 chunks.**

### 11c. Two bugs caught by measurement, same discipline as MEMORY

`PROFILES_TECH` was written from general knowledge before any tech filing existed —
explicitly flagged as such in the source. Checked against the real corpus once it existed:

- **AD_PLATFORM's "CPM" metric: 5 hits** across 9,083 GOOGL/META chunks — earnings-call
  language, not filing language. Swapped for MAU (236 gold chunks).
- **CYBERSECURITY's "new logo growth": 1 hit** across 13,711 PANW/CRWD/ZS chunks —
  effectively dead vocabulary. These filings talk in retention terms, not acquisition
  terms. Re-anchored on billings and renewal rate.
- **DEVICE_ECOSYSTEM's "services revenue": 16 hits vs "Services net sales" at 136** —
  Apple's filings use "net sales" throughout, the exact same class of bug as MU never
  saying "days of inventory" in session 2.

Also found, and not something the taxonomy check would have caught on its own: the
extraction prompt's system rules literally said *"You are extracting claims from
semiconductor filings"* — fixed to be domain-neutral before any tech extraction ran.

### 11d. Retrieval evaluation

72 probes (36 semis + 36 tech after the correction above), 3 skipped (unchanged FOUNDRY
gap). Hybrid: R@20 0.739, paraphrase R@20 0.606 — consistent with the semis-only numbers
in §10b. **All 36 tech probes now resolve to real gold evidence**, versus near-zero for
two of them before the correction pass.

### 11e. THE TECH BASELINE

10 cases, all 6 new segments, `--cases tech`, **~26 min**.

| metric | tech (10 cases) | semis (10 cases, §10d) |
|---|---|---|
| exact quote rate | **0.924** | 0.910 |
| fabrication rate | **0.057** | 0.023 |
| inference claims | 1.4% | 1.5% |

**The aggregate says tech extraction is fine — even slightly better than semis. The
aggregate is hiding the finding.** 11 of 21 total fabrications (52%) come from exactly two
of ten cases — CRM and NOW, both CLOUD_SAAS. Every other segment matches or beats the
semis baseline (CRWD: 100% exact, 0 fabricated, the only tech case with a genuine
inference claim).

**Traced to a mechanism — first diagnosis was wrong, correction was the real finding.**
The original trace (one case, a 30-character prefix match) concluded the model was
substituting a "generic textbook definition" of RPO for what the filing said. Checked
properly with full word-overlap search — same session, before this was published anywhere
external — the real CRM text is *"**Our** remaining performance obligation represents all
future revenue under contract that has not yet been recognized as revenue..."* and the
"fabricated" version is that exact sentence missing the word **"Our."** 23/24 words,
contiguous, verbatim. Checked against NOW and a third, independent case (TSM/FOUNDRY,
§12b): **10 of 11 traced fabrications show 92-99% word overlap** — near-verbatim drift on
long (17-81 word) sentences, not knowledge substitution. **Settled, not a hypothesis.**
Full derivation and the corrected quote comparison: `docs/retrieval_and_extraction.md`
§8g/§9b.

One fabrication *was* a different, more concerning mechanism: a passage truncated
mid-sentence got completed with figures shown nowhere in the text. That one pointed to a
real, fixable cause — see §12c.

**A case-selection mistake, caught before it reached the record.** SHOP@2023-05-01
returned 0 claims in 0 seconds — investigated rather than shrugged off, and traced to
SHOP's indexed corpus covering only 2025-02-11..2025-12-02 (§6). Swapped for PYPL; the
zero-record was removed from `cases.jsonl` rather than left to sit there meaning nothing.

### 11f. What session 3 (first pass) did not settle — now addressed in §12

The CLOUD_SAAS mechanism was a one-case hypothesis (corrected in §11e above). Section
weighting was not re-tuned for tech's boilerplate mix (measured in §12d — no change
needed). No systematic per-ticker date-coverage audit existed (run in §12a — found 12
more gaps of the same class). All three closed out in the same session; see §12.

---

## 12. Session 3 continued: fixes, backfill, readiness work (summary)

Full detail in `docs/retrieval_and_extraction.md` §9. Everything below followed directly
from acting on §11f's open items, in the order dependencies required.

**§12a — Twelve more tickers had zero 10-K/10-Q history, for the FOUNDRY reason.**
A systematic date-coverage sweep (one SQL query, all tickers) found 12, not 3: ASML, ASX,
CAMT, GFS, HIMX, IFNNY, IMOS, NVMI, STM, TSEM, TSM, UMC — every one confirmed against
SEC's own submissions API as a pure 20-F filer with **zero** 10-K/10-Q ever filed. Built
real 20-F support (`SECTION_PATTERNS_20F` + a `prefer_caps` disambiguation fix, gotchas
§6) rather than just diagnosing the gap. **+15,736,832 tokens, corpus at 164,541,463 /
54,392 documents, index at 488,400 chunks, all 72 retrieval probes now resolve (0
skipped).** IFNNY has a genuine, permanent gap (all its filings predate 2010); SGH was a
dead ticker symbol fixed to PENG, which was already correctly indexed under `memory`
(a duplicate entry under `analog` — also fixed — had been silently overriding it).

**§12b — The CLOUD_SAAS fabrication mechanism from §11e was wrong, and the correction
matters more than the original finding.** Checked with a proper word-overlap trace
instead of a prefix match: the "textbook definition substitution" story does not survive
contact with the actual text. **Real mechanism: near-verbatim drift** — 92-99% word
overlap, one or two words dropped from an otherwise character-perfect copy of a long
(17-81 word) sentence. Confirmed across three independent cases (CRM, NOW, and a third —
TSM/FOUNDRY, the most textually different case tried all session). Not a hypothesis
anymore.

**§12c — One fabrication genuinely was different, and it pointed to a real, fixable bug.**
A passage cut off mid-sentence got completed with figures shown nowhere in the text.
Measured directly: **67% of sampled passages ended mid-sentence** before a fix (the
paragraph-snap in `chunk.expand()` only fires when a newline falls within the last 400
characters). Fixed with a bounded backward search for the nearest sentence terminator,
verified to never cut into the located span. **Down to 13% after the fix.** Re-baselined
the full 10-case tech set to measure the real effect rather than assume one: exact-quote
rate 0.924→0.939, fabrication rate 0.057→0.045 (21→17 fabricated). Modest, and correctly
so — this fix addressed one sub-mechanism, not the dominant near-verbatim-drift one (§12b),
which nothing this session touched.

**§12d — Section weighting for SaaS: measured, no change needed.** The semis-tuned
`section_caps` already achieve 73% high-value retrieval on 6 CLOUD_SAAS tickers, better
than semis' own 65%. A real null result, reported rather than manufactured into an
unneeded change.

**§12e — The trade journal had zero test coverage despite the state table saying
"tested end-to-end."** 12 tests written; the code itself had no bugs, the claim was just
unverified. Three CLI commands added (`journal record`/`close`/`status`), tested
end-to-end against a throwaway database. No trade recorded — that requires an actual
decision made together, not an invented one to fill in a checkbox.

**§12f — Two readiness pieces, explicitly not validated results.** LAP scoring
(`engine/eval/lap.py`, 14 tests) — a date-only recall probe estimating whether the model
already knows a setup's outcome from pretraining, wired into `gate.py` as a non-blocking
diagnostic warning rather than a new pass/fail threshold. SFT/CPT interleaving
(`engine/training/interleave.py`, 10 tests) — the Llama-Fin mitigation for CPT-then-SFT
collapsing instruction-following, deterministic and resumable, `CPTConfig.
sft_interleave_ratio` defaulting to 0.0 (provably a no-op at that setting). **Neither has
been run against anything real** — no CPT model, no SFT dataset, no journal of real
recommendations exist yet to probe or train on. Built and tested; not measured to help.

**§12g — A live, index-free retriever now exists.** `LiveEdgarRetriever`
(`tools/retrievers/live_edgar.py`, 13 tests) resolves a ticker to a CIK live and pulls its
most recent filings directly from `data.sec.gov` — no pre-built index required, works for
any SEC filer. Free (SEC's own APIs, no paid news service, no account). Includes 8-K
exhibit-following (reusing `sources/edgar.py`'s existing regex, not a second definition of
"exhibit"), verified live against a ticker touched nowhere else this session. Not a
replacement for the corpus — no retrieval quality applies to it, and it only reaches the
most recent few filings per call.

**§12h — SHOP's real gap (40-F/6-K) closed, not just diagnosed.** The primary 40-F
document turns out to carry **zero** narrative text — pure inline-XBRL cover-page
metadata (verified: 241KB, almost entirely tag soup). The real content is in separately
filed exhibits, using **English headings with no SEC Item numbering at all**
("Management's Discussion and Analysis", "Description of the Business", "Risk Factors"
— National Instrument 51-102 convention, not SEC's). Checked against 5 independent large
40-F filers (CNI, ENB, TD, BCE, BMO), not just SHOP, before trusting the approach:

- **Exhibit naming**: most large filers use the same `EX-99.N` convention 8-K already
  handles. SHOP is the outlier (`exhibit13mdaq42023.htm`) — one additional pattern
  (`FORTYF_EXHIBIT_PATTERN`, an OR-extension, not a change to the proven 8-K pattern)
  covers both without touching what already works.
- **Section headings**: unlike 20-F, headings here are **not** reliably ALL-CAPS, so the
  `prefer_caps` fix does not apply — plain last-occurrence logic (already proven on
  10-K/20-F table-of-contents problems) was checked directly against TD's real filing
  and correctly skips the TOC and cross-references to land on the genuine header.
- **6-K needed zero new code** — it already falls through to the existing whole-document
  "full" fallback, since it has no structure to parse in the first place (confirmed:
  SHOP's 6-Ks are just press releases).

Both `sources/edgar.py`'s `discover()` and `LiveEdgarRetriever` were updated together so
the corpus builder and the live retriever agree on what counts as an exhibit for 40-F, the
same principle already applied to 8-K. 23 new tests (synthetic fixtures reproducing the
real TD/SHOP structure measured above), all passing before the live backfill ran.

**Backfill result: +1,032,192 tokens, corpus at 165,429,393 tokens / 54,876 documents,
index at 491,103 chunks.** SHOP now spans **2015-05-21 → 2025-12-02** — the real decade of
history, not the 2025-only fragment. Dedup rate for this slice was **35.5%**, the highest
of any source measured this project (semis 18.4%, tech 21.6%, 20-F 10.6%) — plausibly
6-K press releases recycling boilerplate across many routine filings, though this is a
plausible read of one ticker's data, not a swept finding.

**Verified end to end, not just at the parser level**: `extract brief SHOP --as-of
2023-05-01` — a case that returned 0 claims in 0 seconds before this fix (§11e) — now
produces 43 claims, 98% exact quotes, 1 fabricated (correctly dropped), quote audit PASS.

**§12i — First real SFT dataset built and generated; `SFTTrainer.train()` no longer
missing.** §12f shipped SFT/CPT interleaving machinery with nothing to interleave and a
trainer with `build()`/`masked_loss()` but no `train()` method at all. Both closed:

- **`engine/training/sft_data.py`** (new, 15 tests) — evidence-grounded prompt builder.
  A real `Briefing`'s SOURCED claims (never INFERENCE ones — showing the local extractor's
  own reasoning would let generation launder it into a "recommendation" with nothing new
  happening in between) plus a two-shot demonstration using fictional tickers (XSAS/XMEM,
  same reasoning as `extract/prompts.py`'s XSAS — a real ticker in a worked example risks
  being read as fact, not format). Reuses `generate_structured` (retry-on-validation-error)
  and `SCHEMA_PROMPT` rather than rebuilding either.
- **`SFTTrainer.train()`+`evaluate()`** (`engine/training/sft.py`, 8 new tests) — same
  resumable-checkpoint discipline as `CPTTrainer.train()`, reusing `CheckpointManager`/
  `TrainState` directly (`shard_offset` repurposed as "example index within the current
  epoch" — SFT has no shard cursor of its own). `masked_loss` was previously dead code
  with a signature (`inputs, targets, mask` pre-split) that didn't match what
  `build_dataset`/`interleave.py` actually produce (one aligned `ids`+`mask` array,
  shifted internally) — fixed to match `cpt.py`'s already-proven `_sft_loss` shape, so a
  standalone SFT run and an interleaved CPT+SFT step now train on identically-shaped
  batches. Tested against a real tiny `mlx.nn.Module` (not the 14B checkpoint) so the
  orchestration logic — checkpoint cadence, resume, epoch/example bookkeeping — runs on
  real MLX tensor ops without the cost of loading Ministral for a unit test.
- **20 real `Briefing` objects reconstructed**, not freshly re-extracted. The existing
  `artifacts/baseline{,_tech}/base/*.md` files from the DEFAULT_CASES/TECH_CASES baseline
  runs were the only artifact ever persisted (markdown only, never structured JSON) —
  re-running extraction would cost another ~1-1.5h for data already paid for once.
  `scripts/build_sft_briefings.py` parses each SOURCED claim's `text` line back out of the
  markdown (verbatim — `export.py` emits it unmodified, so this is lossless for what
  `render_evidence()` actually uses) and attaches synthetic `Source` placeholders
  (`content_sha256="reconstructed"`, not a real hash) purely to satisfy `Briefing`'s
  "every claim's source_id is known" validator. **This is not a substitute for a fresh
  Phase 4 provenance audit** — `Claim.quote` in the reconstruction is markdown's
  400-char-truncated copy, not the full verified span — but nothing in the SFT pipeline
  reads `.quote`, only `.text`. All 20 round-trip through `Briefing.model_validate_json`;
  673 total sourced claims, 20-40 per case.
- **Measured, not assumed, generation yield — and it changed the plan twice.** A 3-case
  throughput probe (`max_tokens=700`, the original guess) got 1/3: two failed with a
  `sub_segment` schema-validation error, then failed every retry after. Root cause:
  `taxonomy.prompt_context()` prints the segment's human-readable **name**
  ("Semiconductor capital equipment") but never the machine key the JSON must echo back
  ("equipment") — the model had to guess it from two few-shot examples that only
  demonstrate 2 of 11 possible values. Fix: `build_prompt()` now states
  `(sub_segment field must be exactly "equipment")` explicitly, the same treatment
  `TICKER:` already gets. Re-measured: 3/3, first attempt, ~24.5s/example. Running the
  real 20-case batch surfaced a **second, different** failure: 3 cases hit the 700-token
  cap mid-`summary` — the model does not reliably keep `summary` to "one paragraph" as
  instructed and runs on verbatim-echoing evidence until truncated, so the JSON never
  closes. Raised `generate_sft_example`'s `max_tokens` to 1400 (measured: ~2x the longest
  observed truncation point) and reran the 3 failures — 1 resolved (PANW), but **2
  (GOOGL, CRM) still hit 1400 tokens exactly, still mid-`summary`, still not converging.**
  Did not raise the budget further to chase them: a completion that runs on regardless of
  budget is a different, deeper failure than one that was merely cut short, and continuing
  to throw tokens at it would be the same mistake as tuning the Phase 1 quant gate to
  pass — a metric improved by construction, not by the thing it measures getting better.
  **Final yield: 18/20 (90%), reported as measured**, not padded to 20 with a manufactured
  example.
- **Dataset validated against the real model, not just schema-checked.** All 18 examples
  load via `load_examples()`; tokenized against the real Ministral tokenizer, all 18 fit
  under `seq_len=4096` with zero truncation (2370-3080 tokens, mean 2654); completion mask
  covers 13-31% of each sequence (mean 18.5%). A real forward+backward pass on the longest
  example (3080 tokens) through the actual 14B checkpoint with LoRA attached produced a
  finite loss and, **with `grad_checkpoint()` applied** (as `SFTTrainer.build()` always
  does), peaked at **19.68GB** — safely under the 55.7GB working-set ceiling. (An earlier
  check that skipped `grad_checkpoint` by mistake read 60.88GB, over the ceiling — that
  number was an artifact of the simplified validation script, not of the real trainer
  path, and is called out here so it is not mistaken for a real finding later.)
- **Not run**: no actual SFT training pass has happened. `sft.py`'s own architecture says
  SFT should come after CPT ("running SFT afterwards restores instruction-following that
  continued pretraining degrades"; reversing the order has CPT undo the SFT) and no CPT
  pilot has run yet either. Building and validating the machinery is what this entry
  covers, not a training result.

Dataset: `data/sft/examples.jsonl` (18 lines, `SFTConfig.data_path`'s default). Source
briefings: `data/sft/briefings/*.json`.

**§12j — Briefings now carry price, insider activity, and macro context, not just
filings.** User pushback on "does this cover all necessary information" led to a scoping
conversation (price/news/analyst/sentiment/insider/macro), and a split by what actually
needs the extraction/quote-verification machinery. Structured data (numbers pulled
straight from a provider, nothing to paraphrase) doesn't: three sources were added, all
free, all no-signup —

- **Price/volume snapshot** (`tools/price_context.py`) — wraps the existing
  `PriceStore`/`YFinanceProvider` from the quant pipeline; 20d/90d/252d returns, 252d
  range, volume trend. Never uses a bar after `as_of`; a genuinely empty history (delisted,
  pre-IPO) returns `None` rather than a hollow snapshot.
- **Insider activity, Form 3/4/5** (`tools/insider.py`) — parses the raw ownership XML
  directly (reuses `EdgarClient`/`resolve_ciks`, the same primitives `sources/edgar.py`
  and `live_edgar.py` use, but does not go through the CPT corpus pipeline since this is
  structured XML with nothing narrative to section-match on). **Measured before trusting
  the schema**: NVIDIA's filing agent encodes boolean flags as `"1"`/`"0"`; Apple's as
  `"true"`/`"false"` — same SEC schema, different filing-agent encoding, both handled.
  Open-market purchases/sales (codes P/S) are kept distinguishable from routine
  compensation mechanics (grants, tax withholding, option exercises) — collapsing "9
  Form-4 filings" into one count would misrepresent routine RSU vesting as insider
  conviction.
- **Macro context** (`tools/macro.py`) — five FRED series (fed funds rate, 10Y yield,
  CPI, industrial production, unemployment) via the unauthenticated `fredgraph.csv` export
  endpoint rather than FRED's REST API, which needs a registered key — same "don't sign up
  for things without asking" reasoning `live_edgar.py` gives for not using a paid news API.
  **Revision leakage is flagged, not hidden**: CPI/industrial-production/employment series
  get revised after initial release, so a value pulled "as of" a historical date is the
  CURRENT vintage, not necessarily what was known then. Policy-rate series are final at
  publication. `MacroSeries.revision_safe` carries this into the rendered output.

All three are optional fields on `Briefing` (`contracts/context.py`), attached via
`model_copy` after the extraction loop runs, and rendered as their own markdown sections in
`export.py`. None of the three touches `audit_briefing` (the Phase 4 gate) — no claim, no
quote, nothing to verify. `extract brief` fetches all three by default; `--no-price`/
`--no-insider`/`--no-macro` opt out. 34 new tests (`tests/tools/`, plus additions to
`tests/extract/test_loop_and_export.py` for the new render sections and Briefing backward
compatibility), all against faked clients/responses — no real network call in the test
suite. Verified live end to end against NVDA @ 2024-05-01: correct split-adjusted price,
9 correctly-classified insider sells (0 open-market buys), 5 macro series, all
leakage-bounded to the as_of date.

**What this does not change**: the alpha question from §12i and the literature review
(§4e) is untouched. More inputs is not more skill — this makes the "coverage" claim in the
markdown output more true and the tool more useful for judgment, it does not make the
judgment better on its own. Full news/research-note coverage remains deferred (real
ongoing licensing cost); sentiment was deprioritized on purpose (the vaguest category, and
the one closest to the noisy/contaminated signal types §4e's literature review already
flags as where false-alpha claims concentrate). Analyst price targets turned out to be
free after all — see §12k.

**§12k — Two more gaps closed same-session: briefings can now go live, and carry a
leakage-safe analyst consensus.**

- **`extract brief --live`** (`engine/extract/live.py`, 4 new tests). The corpus has a
  hard build cutoff (2025-12-31) — `extract brief` against the index cannot see anything
  filed after that no matter what `--as-of` is passed, which surfaced directly when asked
  for GOOGL news less than 6 months old and got a `2025-12-01` briefing back. `live_edgar.
  py` (§12g) already solved the fetch side of this and was never connected to the claim-
  extraction/quote-verification half of the pipeline; `loop.py`'s batching/verification
  logic was pulled out into a shared `extract_claims_from_sources()` so the live path
  reuses it unchanged rather than re-implementing verification a second time. **Measured
  trade-off, not swept under the rug**: live passages are whole extracted filing sections
  instead of the indexed path's short ranked chunks, and a live GOOGL run scored 50%
  exact-quote / 44% fabrication-and-dropped vs the indexed path's 98%/2% on the same
  ticker — a real quality cost from longer, unchunked passages, not a bug in the
  verification step (the audit still passed; nothing fabricated reached the briefing).
  Confirmed the freshness itself works: a live GOOGL run surfaced an August 2026 $25bn
  notes offering and June 2026 $84.75bn AI-infrastructure equity raise the indexed path
  had no way to see.
- **Analyst price-target consensus** (`tools/analyst.py`, `contracts.AnalystConsensus`,
  10 new tests). yfinance's live `targetMeanPrice` field is free and real but carries no
  date — attaching it to a historical `--as-of` briefing would leak today's numbers into
  a past decision, worse than the CPI-revision leakage macro.py already flags since it
  is not a minor restatement, it is the whole future. Reconstructed instead from
  `Ticker.upgrades_downgrades` (dated per-firm actions back to 2012 for GOOGL): most
  recent rating/target per firm on or before `as_of`. **Caught and fixed a real distortion
  before shipping it**: an undated "most recent per firm" pull mixed GOOGL's pre-2022-
  split targets ($1700-$3000) in with post-split ones ($100-$500), because a firm that
  has not re-rated in years still counts as that firm's "most recent" row. Added a
  `max_staleness_days=400` filter (the same lookback convention `ExtractConfig`/
  `price_context.py` already use) — mean target went from a meaningless $526-587 to a
  sane $295-397 across historical/live test dates, with the live number landing close to
  yfinance's own (unlabelled) live consensus as a sanity check.

Both wired into `extract brief` (`--live`, `--no-analyst`), attached to `Briefing` the
same optional-field pattern as §12j's three, no interaction with the Phase 4 gate. 232
tests passing total.

**§12l — Diagnosed and fixed the §12k live-briefing quality gap's most visible symptom:
inference claims that copy the prompt's own few-shot demo verbatim.** User caught a live
GOOGL briefing's only inference line asserting a revenue DECLINE while the same briefing's
sourced claims showed revenue up 24% — self-contradictory in a way that looked like a
stale/mismatched artifact. Traced exactly, not guessed at: the line was a character-for-
character copy of `prompts.py`'s own MU/memory demo sentence ("...pricing correction
rather than a demand collapse"). Not hallucination in the usual sense — the base model
over-anchored on a demonstrated exemplar's CONTENT instead of just its pattern, and
echoed the prompt back rather than reasoning about GOOGL's actual passages.

Fix: `prompts.FEWSHOT_INFERENCE_TEXTS` extracts the three demo inference sentences
programmatically from `_FEWSHOT` (regex + json.loads, not duplicated string literals, so
it cannot drift out of sync with the prompt itself) and `claims.parse_claims` drops any
inference claim that matches one verbatim — same treatment a fabricated sourced quote
already gets (dropped and counted, never silently kept). New `ExtractionStats.
inference_copied_fewshot` counter. 4 new tests, one of which explicitly documents the
real scope limit: this is exact-match only, so a *paraphrase* of a demo sentence is not
caught, only a verbatim copy. **Verified against the exact failure that prompted it**:
re-ran the live GOOGL briefing — `82% sourced` / 2 inference lines (both copies) before,
`100% sourced` / 0 inference lines after, same run otherwise. 236 tests passing total.

This is a symptom fix, not the root-cause fix for §12k's measured quality gap (50%
exact-quote rate on live passages vs 98% indexed) — that still needs the live path
chunked the way the indexed path already is, not yet done.

**§12m — Delisted-universe research (§7's $30 question) scoped for real, EODHD adapter
built, purchase itself still pending the user.** User asked to expand `data/delisted.py`
beyond its 20 names to cover the semiconductor sector broadly. Measured via SEC's own free
registries rather than guessed at: 505 companies ever registered under SIC 3674, 114 with
any live ticker today (95 on NASDAQ/NYSE — free via yfinance, no purchase involved), 391
with none. Filtering those 391 to `>=3 years of 10-K history` (a shell-vs-real-company
proxy) leaves 163 genuine former operating companies — 150 net new after removing overlap
with the existing list. Attempted automated ticker-symbol extraction from each one's most
recent 10-K cover page: **only 32/150 (~21%) resolved**, a real structural ceiling, not a
regex bug — SEC only started mandating an explicit "Trading Symbol" cover-page field in
recent years, so most pre-~2019 filings never state the ticker in extractable cover-page
text at all (confirmed directly: Applied Micro Circuits' 2016 10-K, a real NASDAQ company,
states only "The Nasdaq Stock Market LLC" with no ticker text anywhere on the cover page).
Full research persisted to `artifacts/delisted_research/` (was ephemeral `/tmp` output
before this). **"Tech" sector has no equivalent single SIC code and was not attempted —
a separate, larger sweep, not started.**

`data/providers/eodhd_provider.py` — the actual `PriceProvider` swap `base.py`'s own
docstring anticipated — is built and tested (11 tests, no real API calls, verified against
EODHD's documented EOD endpoint shape before writing it) and ready to pull data the moment
an API key exists. **The $29.99/mo purchase itself is the user's decision and has not been
made** — nothing technical is blocking it, it is purely their call on timing. 52 tickers
(20 existing + 32 resolved) need no purchase at all and could be pulled via yfinance right
now if useful on their own.

---

## Key files

| Path | What |
|---|---|
| `docs/HANDOFF.md` | This file |
| `docs/retrieval_and_extraction.md` | Retrieval + extraction design and measurements (§1-7 semis, §8-9 tech + fixes) |
| `configs/universe/tech_v1.yaml` | 26 software/internet tickers, corpus-only |
| `configs/universe/foreign_filers_20f_v1.yaml` | 12 twenty-F tickers (TSM, UMC, ASML, STM, ...), corpus-only |
| `configs/universe/shop_40f_v1.yaml` | SHOP alone, scoped so its 40-F/6-K backfill doesn't re-touch the other 25 tech tickers |
| `artifacts/baseline_tech/` | Tech extraction baseline v2, post passage-truncation fix |
| `artifacts/baseline_tech_v1_before_truncation_fix/` | Tech baseline v1, kept for the before/after comparison |
| `artifacts/retrieval_eval.txt` | Retrieval evaluation output (72/72 probes resolving) |
| `artifacts/briefings/*.md` | Example briefings from the smoke tests |
| `src/argus/engine/eval/lap.py` | LAP contamination scoring (built, unmeasured — no CPT model yet) |
| `src/argus/engine/training/interleave.py` | SFT/CPT interleaving (built, unmeasured end-to-end) |
| `src/argus/engine/training/sft_data.py` | SFT prompt builder + generation pipeline (§12i) |
| `data/sft/briefings/*.json` | 20 reconstructed real `Briefing` objects (§12i — not a fresh provenance audit, see caveat) |
| `data/sft/examples.jsonl` | 18 generated SFT examples, schema-valid, `SFTConfig.data_path`'s default (§12i) |
| `scripts/build_sft_briefings.py`, `scripts/generate_sft_dataset.py` | One-off dataset-build scripts, resumable (§12i) |
| `src/argus/contracts/context.py` | PriceSnapshot/InsiderActivity/MacroContext schemas (§12j) |
| `src/argus/tools/{price_context,insider,macro,analyst}.py` | Structured briefing appendices: price, Form 3/4/5, FRED macro, analyst consensus (§12j, §12k) |
| `src/argus/engine/extract/live.py` | Live-sourced briefing (`extract brief --live`), for dates past the corpus cutoff (§12k) |
| `src/argus/data/providers/eodhd_provider.py` | EODHD delisted-price provider, built and tested, purchase pending (§12m) |
| `artifacts/delisted_research/` | Semiconductor delisted-universe research: 163 candidates, 32 resolved tickers, full methodology (§12m) |
| `src/argus/tools/retrievers/live_edgar.py` | Live, index-free retriever — any SEC filer, no pre-build needed |
| `artifacts/bpc_qwen38_27b.json`, `artifacts/bench_qwen38_27b_seq512*.json` | Qwen3.8-27B evaluation (§4b-addendum) — best-measured BPC, disqualified on throughput |
| `docs/phase0_model_selection.md` | Full benchmark data + scaling law |
| `docs/literature_review.md` | 10 papers, with the negative results |
| `~/.claude/plans/i-m-starting-a-new-robust-backus.md` | Original plan (superseded on architecture — see §1) |
| `artifacts/phase1_validation.txt` | The failed gate output |
| `artifacts/bench_*.json`, `artifacts/sweep/*.json` | Every raw benchmark |
| `data/corpus/manifest.json` | Corpus provenance + hashes |
| `tests/quant/test_leakage.py` | 13 leakage tests — keep them green |
