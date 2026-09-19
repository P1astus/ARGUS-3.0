# Literature Review — LLM Trading Frameworks and Financial Domain Adaptation

**Compiled 2026-08-02** in response to the TradingAgents paper (arXiv 2412.20138).
Question driving the search: **is the nine-day CPT run justified, and does the wider
literature support LLM-based trading decision support at all?**

Short answer: the evidence has moved substantially since TradingAgents, and it does not
favour the CPT run as originally specified. Two findings change our design directly, and
one changes how much weight to put on any published trading result.

---

## 1. The single most important paper: memorisation-controlled evaluation

**"From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents"**
([arXiv 2605.28359](https://arxiv.org/html/2605.28359v1)) — KTD-Fin, CSI300, 2024-2026.

This is the most rigorous evaluation of LLM trading agents found, because it is the only
one that properly controls for the model already knowing the answer. Four-level masking:

| Level | Tickers | Dates |
|---|---|---|
| Bright | real | real |
| Stock-blind | anonymised | real |
| Date-blind | real | anonymised |
| Blinded | anonymised | anonymised |

Masking was *verified* with de-anonymisation probes across ten attacker models: ticker
recovery 3.0%, joint ticker+date recovery 1.5%.

**Findings:**

- The anchor model returned **−0.16% with visible tickers but exactly 0.00% (pure cash)
  when anonymised** — i.e. its trading was driven by pretraining memory, and with the
  memory removed it declined to trade at all.
- Barra-style attribution found **nine of ten agents post NEGATIVE selection alpha.**
  Cumulative returns came from passive factor exposure, not stock picking.

This is the closest thing to a definitive answer in the literature: **once memorisation is
properly controlled, LLM stock selection is not merely weak, it is negative.**

## 2. Lookahead bias is measurable, and it explains most published results

**"Detecting Lookahead Bias in LLM Forecasts"** ([arXiv 2512.23847](https://arxiv.org/abs/2512.23847))
introduces **Lookahead Propensity (LAP)** — a date-only recall query for a firm-date pair
that estimates the probability the model has internalised the realised outcome.

- LAP is **materially positive throughout the in-sample period** and **collapses
  essentially to zero right after the training-data cutoff.**
- LLM forecast predictive power is **amplified on high-LAP firm-date pairs**, and the
  interaction **loses significance on post-cutoff samples.**

Translation: the apparent skill largely *is* the contamination. Notably,
**Lopez-Lira, Tang & Zhu (2025)** — same author as the original optimistic "Can ChatGPT
Forecast Stock Price Movements?" result — now argue one **must** use a sample period after
the knowledge cutoff.

**Actionable for us:** LAP is cheap and implementable. It should be added to our eval
harness as a per-setup contamination score, so we can report results conditioned on it
rather than merely asserting our post-cutoff window is clean.

## 3. Field-wide evidence quality is very poor — and ours is already better

A review of **77 studies, 19 given detailed audits**
([evidence ledger](https://insights.wisdomchain.com/agentic-trading-evidence-ledger/),
summarising [arXiv 2606.08285](https://arxiv.org/pdf/2606.08285)):

| Reported | Count |
|---|---|
| Time-consistent split protocols | **2 / 19** |
| Explicit transaction-cost model | **1 / 19** |
| Universe / survivorship handling | **1 / 19** |
| Execution timing semantics | 11 / 19 |
| Lowest reproducibility tier | 15 / 19 (none reached highest) |

**ARGUS already does all three of the rare ones**: purged walk-forward with embargo,
10bps round-trip costs in the gate, and a per-fold survivorship-coverage disclosure that
fires an automatic warning. On the axes this review measures, our Phase 1 methodology sits
in the top few percent of the published literature.

That is worth stating plainly because it reframes the Phase 1 "failure": we did not fail
where others succeeded. We measured something most papers do not measure, and reported what
it said.

## 4. On TradingAgents specifically

The paper that prompted this review reports 26.62% CR and **Sharpe 8.21** on AAPL.

- The authors themselves flag the Sharpe as implausible (p.12 footnote): it *"exceeds our
  expected empirical range"*, attributed to *"few pullbacks"* in the window.
- The sample is **3 stocks over ~60 trading days** (Jan 1 – Mar 29 2024), with no
  confidence intervals, significance tests, or bootstrap anywhere in the paper.
- GPT-4o / o1-preview training data covers Q1 2024, so by the LAP result above, the
  evaluation window is exactly where lookahead bias is largest.
- Independent commentary notes results are **not reproducible**: returns depend on model,
  temperature, date range and sampling, and the framework is best read as *"a research
  scaffold for studying multi-agent analysis, not a strategy with a fixed, replicable
  return."* One documented run: ~7% over 30 days vs S&P 4.5%, with **22% drawdown**.

**What it does support:** the architecture. Structured inter-agent documents over free
prose (their §4.1 "telephone effect" argument) is the same reasoning behind our `contracts`
layer, and their bull/bear debate maps onto our bull/base/bear scenarios. Worth borrowing;
not worth citing as evidence of returns.

---

## 5. Findings that change our CPT design

### 5a. Our LoRA rank was far too low — and fixing it is free

**"LoRA Learns Less and Forgets Less"** ([arXiv 2405.09673](https://arxiv.org/abs/2405.09673),
TMLR/ICLR) tested LoRA vs full finetuning on **20B-token continued pretraining**:

- *"In standard low-rank settings, LoRA substantially underperforms full finetuning."*
- **Domain adaptation is far more rank-sensitive than instruction tuning.** At 20B tokens
  on code CPT, **r=256 scored 0.617 vs full-FT 0.545**; on math CPT r=256 scored 0.616 vs
  0.613. Rank ~256 approaches or beats full finetuning; rank 16 does not.
- Full finetuning learns perturbations **10-100× higher rank** than typical LoRA settings.
- LoRA *does* forget less, and **rank controls the amount of forgetting** — relevant to our
  catastrophic-forgetting concern.

**Our config used rank 16.** Measured on the selected model (Ministral-3-14B-Base, seq 4096):

| rank | trainable | tok/s | peak mem |
|---|---|---|---|
| 16 | 19.7M (0.73%) | 135.9 | 23.7 GB |
| 64 | 78.6M (2.85%) | 137.4 | 24.4 GB |
| 128 | 157.3M (5.55%) | 135.9 | 25.3 GB |
| **256** | **314.6M (10.52%)** | **136.0** | **27.2 GB** |

**Throughput is flat across a 16× rank increase** — the adapters are negligible beside the
base model's forward/backward pass. Cost of the fix: **+3.5GB memory, zero time.**

Running CPT at rank 16 would have been an underpowered version of the experiment: a null
result would not have distinguished "CPT doesn't help" from "our rank was too low."
**Default changed to 256.** (The preflight ceiling needed raising from 5% to 15% to permit
it — the guard fired correctly, the threshold was simply calibrated for the old rank.)

### 5b. Sequential CPT→SFT is the wrong order; joint is materially better

**"Demystifying Domain-adaptive Post-training for Financial LLMs"**
([arXiv 2501.04961](https://arxiv.org/html/2501.04961v3)) — the FinDaP/Llama-Fin recipe,
**~6B CPT tokens**, ~3M IT prompts, ~32K preference pairs:

- **CPT alone causes catastrophic forgetting**: MT-Bench instruction-following collapsed to
  **~1.0 versus 7.8 for the base model.**
- **Joint CPT+IT (simultaneous) prevented the collapse** far better than sequential
  training. Mixed general-domain data acted as replay; downsampling CPT to match IT size
  improved results.
- Forgetting decreased progressively CPT → IT → PA.
- Encouragingly for the domain-adaptation thesis: **Llama-Fin (8B) beat Palmyra-Fin (70B)
  and GPT-4o** on tasks similar-but-unseen relative to its training data.

**Our plan is sequential CPT then SFT — the configuration this paper found worst.** Two
mitigations, in order of importance: interleave a fraction of SFT-format examples into the
CPT stream rather than running them as separate phases, and keep the planned 25% general
replay. Note also **their corpus was 6B tokens; ours is ~0.1B — roughly 60× smaller.**

### 5c. RAG generally beats unsupervised fine-tuning for knowledge injection

Multiple comparisons find **RAG consistently outperforms unsupervised fine-tuning for both
in- and out-of-distribution knowledge**, with the gap largest on less-common facts; hybrid
approaches do best. On FinanceBench (SEC filings), fine-tuned *embeddings* plus reasoning
iterations drove most of the gain.

This supports the ordering suggested before this review: **tool-grounding/RAG attacks
factual accuracy — CPT's known weakness — more directly and far more cheaply.** Path B was
deferred; this is an argument for un-deferring it ahead of the nine-day run.

---

## 6. What I would now recommend

1. **Do not skip the pilot.** Everything above raises the prior that CPT buys style rather
   than skill. The 5-10M-token pilot (~7-13h) is the cheap discriminator.
2. **Run CPT at rank 256, not 16** — free, and required for the experiment to be a fair
   test of the hypothesis.
3. **Interleave SFT-format data into CPT** rather than running them sequentially.
4. **Add LAP scoring to the eval harness** — per-setup contamination measurement, so
   results can be reported conditional on it.
5. **Consider promoting RAG ahead of CPT.** Better supported, far cheaper, targets the
   right weakness.
6. **Expect negative selection alpha** and design around it. KTD-Fin found 9/10 agents
   negative once memorisation was controlled. Our Phase 3 gate already refuses to treat
   "beats a no-skill quant arm" as a pass; this is the evidence justifying that choice.

**The honest summary:** the literature does not support LLM stock *selection*. It offers
qualified support for domain adaptation improving domain *task* performance (Llama-Fin), and
for agent architecture improving structure and explainability. Those are decision-support
properties, not alpha — which is exactly what the project was specified to be.

---

## Sources

- [From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents (2605.28359)](https://arxiv.org/html/2605.28359v1)
- [Detecting Lookahead Bias in LLM Forecasts (2512.23847)](https://arxiv.org/abs/2512.23847)
- [Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Based Trading Systems (2606.08285)](https://arxiv.org/pdf/2606.08285)
- [LoRA Learns Less and Forgets Less (2405.09673)](https://arxiv.org/abs/2405.09673)
- [Demystifying Domain-adaptive Post-training for Financial LLMs (2501.04961)](https://arxiv.org/html/2501.04961v3)
- [TradingAgents: Multi-Agents LLM Financial Trading Framework (2412.20138)](https://arxiv.org/abs/2412.20138)
- [The Data Efficiency Frontier of Financial Foundation Models (2512.12384)](https://arxiv.org/abs/2512.12384)
- [Can ChatGPT Forecast Stock Price Movements? (2304.07619)](https://arxiv.org/pdf/2304.07619) · Lopez-Lira, Tang & Zhu (2025) on post-cutoff requirement
- [Agentic Trading: When LLM Agents Meet Financial Markets (2605.19337)](https://arxiv.org/html/2605.19337v1)
- [TrustTrade (2603.22567)](https://arxiv.org/pdf/2603.22567) · [ContestTrade (2508.00554)](https://arxiv.org/pdf/2508.00554)
