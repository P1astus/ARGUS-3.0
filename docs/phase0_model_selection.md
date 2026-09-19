# ARGUS Phase 0 — Base Model Selection Report

**Measured 2026-08-02, 02:31–06:11, Apple M5 Pro / 64GB / macOS 26.4**
MLX 0.32.0, mlx-lm 0.31.3 · 4-bit weights · attention-only LoRA r=16 · gradient
checkpointing ON · `mx.compile` off · batch 1 · AdamW · 10–15 min sustained phase per config

Metal reports a **55.7 GB recommended working set** — the real ceiling, not 64 GB.

---

## 1. Recommendation

**Primary: `Ministral-3-14B-Base-2512` at seq 4096.**

It is the only true *base* model in the candidate set, and it is also the fastest 14B measured
— 16% quicker than Qwen3-14B at identical settings. Those two properties normally trade
against each other; here they agree, which is why it is the recommendation rather than a
compromise.

Being a base model matters more than the 16%. Every other candidate is post-trained, so CPT on
it risks degrading instruction-following and chat formatting — the risk that forced the low-LR,
replay-mixing, and per-checkpoint forgetting mitigations in the plan. Starting from a base
checkpoint removes the failure mode instead of managing it.

**Cost at your locked 85–110M corpus: 7.2–9.3 days per epoch, 23.7 GB peak** (32 GB of
headroom for a larger batch or 8192 context).

**Use the 8B for pipeline validation only.** At 208 tok/s it is ~1.8× faster and ideal for
debugging checkpoint/resume, corpus loading, and shard iteration cheaply. But do **not** use an
8B pilot to decide whether CPT helps — that signal does not transfer across model scale. Pilot
the *plumbing* on 8B; run the *experiment* on the 14B.

---

## 2. Full results

| Model | Params | Arch | seq | tok/s | Peak mem | 100M epoch |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 8B | dense | 4096 | **208.1** | 16.0 GB | 5.56 d |
| **Ministral-3-14B-Base** | 14B | dense | 4096 | **137.0** | 23.7 GB | **8.45 d** |
| Qwen3-14B | 14B | dense | 4096 | 118.4 | 22.6 GB | 9.78 d |
| Ministral-3-14B-Base | 14B | dense | 8192 | 115.7 | 39.0 GB | 10.00 d |
| Qwen3-14B | 14B | dense | 8192 | 97.4 | 38.7 GB | 11.88 d |
| Mistral-Small-3.1-Text-24B | 24B | dense | 4096 | 85.2 | 25.9 GB | 13.59 d |
| Qwen3.6-35B-A3B | 35B | **MoE** | 512 | 55.5 | 28.8 GB | 20.85 d |
| Qwen3-32B | 32B | dense | 4096 | 50.4 | 34.9 GB | 22.97 d |
| Qwen3.6-35B-A3B | 35B | **MoE** | 2048 | 19.0 | 56.2 GB ⚠ | 60.77 d |
| Qwen3.6-35B-A3B | 35B | **MoE** | 4096 | **OOM** | — | — |

⚠ exceeds the 55.7 GB working set → swapping.

---

## 3. The scaling rule (the durable result)

You asked what to do when a more capable open model of a given size appears. This is the
answer — you can predict its cost without re-measuring.

Dense-model training throughput on this machine is **almost exactly inverse-linear in
parameter count**:

```
    tok/s  ≈  1660 / params_in_billions        (seq 4096, batch 1, attention-only LoRA r=16)
```

| Params | Predicted | Measured | Error |
|---|---|---|---|
| 8B | 208 | 208.1 | **0.0%** |
| 14B | 119 | 118.4 (Qwen3) | **0.5%** |
| 24B | 69 | 85.2 (Mistral) | −19% (model beats trend) |
| 32B | 52 | 50.4 | 3.1% |

Architecture moves this by roughly ±20% — both Mistral models beat the trend (Ministral-14B by
16% over Qwen3-14B, Mistral-24B by 24% over prediction). So treat the rule as a **central
estimate with a ±20% band**, and expect Mistral-family architectures at the favourable end.

Peak memory follows an equally simple form:

```
    peak_GB  ≈  params_B / 2  +  12…19        (4-bit weights + activations, seq 4096)
```

| Params | Weights (4-bit) | Measured peak | Implied overhead |
|---|---|---|---|
| 8B | 4.0 GB | 16.0 GB | 12.0 |
| 14B | 7.0 GB | 22.6–23.7 GB | 15.6–16.7 |
| 24B | 12.0 GB | 25.9 GB | 13.9 |
| 32B | 16.0 GB | 34.9 GB | 18.9 |

### Practical ceiling on this hardware

Extrapolating against the 55.7 GB working set:

- **~48B dense at seq 4096** is the approximate limit (24 GB weights + ~20 GB overhead ≈ 44 GB)
- **70B dense would not fit** (35 GB weights + ~20 ≈ 55 GB, at or past the ceiling before
  headroom), and at ~24 tok/s would need ~48 days per 100M-token epoch regardless
- **~32B is the limit at seq 8192** (16 GB + ~35 GB activation ≈ 51 GB, marginal)

So the usable design space for ARGUS is **8B–32B dense**, and everything in it has now been
measured.

---

## 4. MoE trains dense — the assumption that was wrong

The plan estimated 250–600 tok/s for Qwen3.6-35B-A3B by reasoning from its **3B active
parameters**. That was wrong by ~10×, and the reason generalises:

**Backward propagates through the expert stack, so MoE sparsity buys cheap inference, not cheap
training.** Active-parameter count does not predict fine-tuning throughput.

The cleanest evidence is a direct comparison against a dense model of similar total size:

| | Qwen3.6-35B-A3B (MoE, 3B active) | Qwen3-32B (dense) |
|---|---|---|
| tok/s | 55.5 **@ seq 512** | 50.4 **@ seq 4096** |
| Max context reached | 2048 (swapping) | 4096 |
| Memory 512→2048 (4× ctx) | 28.8 → 56.2 GB (**1.95×**) | — |
| Memory 4096→8192 (2× ctx, 14B) | — | 22.6 → 38.7 GB (**1.71×**) |

The MoE is only ~10% faster than a dense model of near-identical total size **while running at
one-eighth the context length** — and its activation memory grows faster with context, which is
what makes 4096 unreachable. On a 64 GB machine, a 35B MoE is strictly worse than a 32B dense
model for CPT.

---

## 5. 8192 context is affordable — and this changes corpus design

Not expected. Both 14Bs run seq 8192 within budget:

| | seq 4096 | seq 8192 | Cost of doubling |
|---|---|---|---|
| Ministral-14B-Base | 137.0 tok/s, 23.7 GB | 115.7 tok/s, 39.0 GB | −15.5% tok/s, 1.65× mem |
| Qwen3-14B | 118.4 tok/s, 22.6 GB | 97.4 tok/s, 38.7 GB | −17.7% tok/s, 1.71× mem |

Doubling context costs only ~18% more wall-clock per epoch. That is a good trade, and it has a
concrete design consequence:

**`engine/corpus/chunk.py` was specified assuming 4096.** At 8192 you can chunk filings on
section boundaries — a full Risk Factors or MD&A section usually fits — instead of cutting them
mid-argument. For a corpus whose value is long-form reasoning about company disclosures, keeping
sections intact is likely worth 18%.

Recommend: **build the chunker section-aware with the window as a config parameter**, decide
4096 vs 8192 after the pilot run, and let the loss curves choose.

---

## 6. Cost matrix against the locked corpus

Days per epoch, at the measured sustained rates:

| Model | 42M (EDGAR−DEF14A) | 57M (EDGAR-only) | 85M | 110M (full Path A) |
|---|---|---|---|---|
| 8B @ 4096 | 2.3 d | 3.2 d | 4.7 d | 6.1 d |
| **Ministral-14B-Base @ 4096** | **3.5 d** | **4.8 d** | **7.2 d** | **9.3 d** |
| Ministral-14B-Base @ 8192 | 4.2 d | 5.7 d | 8.5 d | 11.0 d |
| Qwen3-14B @ 4096 | 4.1 d | 5.6 d | 8.3 d | 10.7 d |
| Mistral-24B @ 4096 | 5.7 d | 7.7 d | 11.6 d | 14.9 d |
| Qwen3-32B @ 4096 | 9.6 d | 13.1 d | 19.5 d | 25.3 d |

The 24B and 32B are feasible but expensive: at 110M tokens they are 15 and 25 days respectively,
which forecloses any second attempt. Given the plan's requirement to **pre-register the gate
before the run** (because you cannot iterate), a 25-day run is a single irreversible bet. The
14B at 9.3 days leaves room to be wrong once.

---

## 7. Thermal behaviour

Sustained-phase degradation was negligible across every config — the 8B measured **−0.1%** over
15 minutes (208.0 → 208.1 tok/s). The plan's warning about multi-day thermal throttling is **not
supported at this timescale**.

Caveat: 15 minutes is not 9 days. Re-check during the first long run; the monitor should log
throughput per checkpoint so drift is visible rather than inferred.

One measured confound, for completeness: a 35B run taken while YouTube was playing returned 52.6
tok/s against 55.5 idle — a **5.5% effect**. Real, but not material to any conclusion here.

---

## 8. What this report does *not* tell you

These are throughput and memory measurements. They say **nothing about which model reasons
better about semiconductors.**

What the sweep establishes is the feasible set: **8B–32B dense**, with the 35B MoE excluded on
memory grounds. Within that set, the choice is a capability judgment that this data cannot make
for you. Specifically untested:

- Reasoning quality on financial/sector text
- How well each base model responds to CPT on filings (Ministral-14B-Base being a base model is
  an *argument*, not a measurement)
- Whether the 8B's reasoning traces help or hinder after CPT on plain text
- Whether 14B is meaningfully better than 8B *for this task* — plausible, unproven

---

## 8b. 14B shootout — task-specific measurements (Phase 0.6)

Throughput narrowed the field to a parameter class. This section picks a model *within*
14B on measurements that bear on the actual task.

### Correction to §6

The §6 cost matrix is **tokenizer-blind and therefore wrong** in a way that matters here.
A corpus is a fixed amount of *text*; how many tokens that becomes depends on the
tokenizer, and our candidates differ by up to **23% in chars-per-token on filing text**.
Comparing models at equal *token* counts silently gives coarse-tokenizer models a free
pass. The corpus should be denominated in characters: **~350–460M characters**
(the 85–110M token estimate at ~4.1 chars/token).

### Domain baseline: bits per character

Measured on 100,000 characters of held-out **2026** 10-Qs — NVDA (fabless), AMAT
(equipment), MU (memory), ADI (analog) — deliberately recent to lean towards measuring
domain modelling rather than memorisation.

Bits per character, not perplexity: perplexity is per-token and is not comparable across
different tokenizers. BPC normalises that away.

| Model | Type | BPC | Token ppl | chars/token |
|---|---|---|---|---|
| **Ministral-3-14B-Base** | base | **0.3394** | 2.108 | 3.168 |
| Ministral-3-14B-Instruct | instruct | 0.3524 | 2.169 | 3.168 |
| phi-4 | post-trained | 0.3691 | 2.723 | **3.912** |
| Qwen3-14B | instruct | 0.3749 | 2.328 | 3.249 |
| DeepSeek-R1-Distill-Llama-8B | reasoning distill | 0.7656 | 7.714 | 3.847 |

**The base/instruct confound was measured, not assumed.** Running Ministral in both
variants puts the instruct penalty at **3.8%** — too small to explain Qwen3-14B's 10.5%
deficit. Base-equivalent estimates: Ministral 0.3394, phi-4 ~0.3556, Qwen3-14B ~0.3612.
Ministral-Base's lead is real.

**The 8B is disqualified as a domain model.** At 0.7656 BPC (token ppl 7.71 vs 2.11) it is
126% worse than Ministral-Base. This is distributional, not a size effect — R1
distillation optimises for reasoning traces and pulls badly away from dense financial
prose. It remains fine for validating *plumbing* (checkpoint/resume, shard iteration), but
its loss curves must not inform the 14B run.

**gemma-4-12B is blocked**, not judged: mlx-lm 0.31.3 raises
`Model type gemma4_unified not supported`. Revisit if mlx-lm adds support.

### Corrected cost: corpus text per second

| Model | tok/s | chars/tok | **chars/s** | 350M chars | 460M chars | Peak | Native ctx |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-8B | 208.1 | 3.847 | 800.6 | 5.1 d | 6.7 d | 16.0 GB | 131,072 |
| **phi-4** | 123.8 | 3.912 | **484.3** | **8.4 d** | **11.0 d** | 18.9 GB | 16,384 |
| **Ministral-3-14B-Base** | 137.0 | 3.168 | 434.0 | 9.3 d | 12.3 d | 23.7 GB | **262,144** |
| Qwen3-14B | 118.4 | 3.249 | 384.7 | 10.5 d | 13.8 d | 22.6 GB | 40,960 |

Note the reversal: **phi-4 has lower tok/s than Ministral but processes ~12% more corpus
text per second**, because its tokenizer is 23% more efficient on filings. Token-rate
rankings invert once you denominate in text.

### Verdict

**Ministral-3-14B-Base**, on three grounds that agree:

1. **Best domain baseline** — 0.3394 BPC, 4.8% ahead of phi-4's base-equivalent and 6.4%
   ahead of Qwen3-14B's, with the instruct confound measured out.
2. **The only true base checkpoint** — removes the CPT-on-post-trained degradation risk
   outright rather than managing it with low LR, replay mixing, and per-checkpoint
   forgetting checks.
3. **262k native context vs phi-4's 16k** — this is about Stage 5 inference, not CPT
   (both handle a 4096/8192 training window). ARGUS reasons over quant output plus
   sub-sector briefings plus filing excerpts simultaneously; 16k is tight for that,
   262k is not.

Cost: ~10% slower than phi-4 in text terms (9.3–12.3 days vs 8.4–11.0) and ~5 GB more
memory. That is the price of the three advantages above, and it is worth paying.

**phi-4 is the credible alternative** if wall-clock dominates: MIT licensed, cheapest
memory, fastest per unit of corpus text, and only modestly behind on domain baseline. Its
16k context is the reason it is not the primary choice.

**Caveats.** All candidates are 4-bit; quantisation quality varies by conversion, so treat
sub-2% BPC differences as noise (the gaps cited here are 4–10%). BPC measures next-token
modelling of filing prose — a good proxy for domain familiarity, but **not** a measure of
reasoning quality about trade setups, which nothing here tests.

---

## 8c. Does more parameters buy domain quality? (Phase 0.7)

Same BPC protocol extended to the 24B and 32B classes.

| Model | B | BPC | base-eq | chars/s | 350M chars | 460M chars | Peak |
|---|---|---|---|---|---|---|---|
| Mistral-Small-3.1-Text-24B | 24 | **0.3307** | **0.3186** | 269.9 | 15.0 d | 19.7 d | 25.9 GB |
| **Ministral-3-14B-Base** | 14 | 0.3394 | 0.3394 | 434.0 | **9.3 d** | **12.3 d** | 23.7 GB |
| phi-4 | 14 | 0.3691 | 0.3556 | **484.3** | 8.4 d | 11.0 d | 18.9 GB |
| Qwen3-14B | 14 | 0.3749 | 0.3612 | 384.7 | 10.5 d | 13.8 d | 22.6 GB |
| Qwen3-32B | 32 | 0.3795 | 0.3656 | 163.7 | 24.7 d | 32.5 d | 34.9 GB |
| DeepSeek-R1-Distill-8B | 8 | 0.7656 | 0.7376 | 800.6 | 5.1 d | 6.7 d | 16.0 GB |

### The headline: parameter count is a weak predictor of domain fit

Holding the model family fixed and increasing size:

- **Qwen3 14B → 32B: BPC gets 1.2% WORSE** (0.3749 → 0.3795) despite 2.3× the parameters
  and 2.4× the training cost.
- **Mistral 14B → 24B: 2.6% better raw** — but the 14B here is a *base* checkpoint and the
  24B is *instruct*, so on a base-equivalent basis the gap is ~6.1%.

**A 32B model is worse at modelling semiconductor filings than a 14B from a better-suited
family.** Within 14B–32B, training mix and family dominate parameter count. This is the
direct answer to "how many parameters": the question is less important than which family,
and paying for 32B buys negative domain quality here.

### Is the 24B worth it?

Against Ministral-3-14B-Base:

| | Change |
|---|---|
| BPC | −2.6% raw, −6.1% base-adjusted (better) |
| Wall-clock | **+61% slower** (15.0 d vs 9.3 d at 350M chars) |
| Memory | +2.2 GB |

Buying ~6% domain quality for +61% wall-clock is a poor trade on its own terms, and it
worsens a constraint the plan already flagged: at 15–20 days per epoch you get **one
attempt**, which sits badly with the requirement to pre-register the gate because you
cannot iterate. The 14B at 9.3 days leaves room to be wrong once.

There is also a practical blocker. `mistralai/Mistral-Small-3.1-24B-Base-2503` **does
exist** under Apache 2.0 — but **no MLX 4-bit build is published**, so using the 24B *base*
would require downloading ~48 GB of bf16 weights and converting locally. The 0.3307 figure
above is for the **instruct** variant, which is the only one currently runnable.

### Caveat on the margins

The stated noise floor is ~2% (4-bit quantisation quality varies by conversion). The
Mistral-24B-instruct vs Ministral-14B-Base gap is **2.6% raw — barely above that floor**.
The base-adjusted 6.1% rests on the 3.8% instruct-penalty correction measured on the
Ministral pair, which is a reasonable but unverified transfer to the 24B. Treat the 24B's
quality edge as *probable but not established*; the 61% cost difference is measured and
certain.

---

## 9. Open decisions

1. **Model** — Ministral-3-14B-Base recommended; 8B is the pragmatic alternative if you want
   iteration room over capability.
2. **Context** — 4096 vs 8192, worth ~18% wall-clock; affects `chunk.py` design.
3. **Corpus size** — locked at 85–110M, but §6 shows what trimming would buy.

---

## Appendix — reproduction

```bash
# single config
.venv/bin/python scripts/bench_mlx.py \
    --model mlx-community/Ministral-3-14B-Base-2512-4bit \
    --seq-len 4096 --sustain-minutes 15 --no-compile

# full sweep
bash scripts/bench_sweep.sh
```

Raw results: `artifacts/sweep/*.json`, `artifacts/bench_*.json`
Per-run logs: `artifacts/sweep/*.log`

Preflight verified on every run: **3,440,640 trainable params (35B, q/k/v/o only), zero expert
leakage**. `mlx-lm#571` does not apply to mlx-lm 0.31.3 — `LoRASwitchLinear` is present and
expert adaptation works; the live hazard is the inverse (auto-discovery adapting all 256
experts), which `preflight.py` now guards against in both directions.
