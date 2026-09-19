# Retrieval + extraction — design and measurements

**Written 2026-08-03, extended 2026-08-15.** Companion to `docs/HANDOFF.md`. Numbers here
are measured on this machine unless flagged. §1-7 cover the second build session
(semis-only: corpus indexing, the local extraction loop, briefing export, the untrained
baseline). §8 covers the third: extending the same machinery to 26 software/internet
tickers the user also trades. **The corpus and index numbers in §2-3 describe the state as
of session 2 (103M tokens, 306,470 chunks) — §8 has the current totals (148.8M tokens,
442,770 chunks).** Left as-is rather than edited in place, because the session-2 numbers
are what the semis-only baseline in §7 was measured against.

---

## 1. Why retrieval exists at all

Handoff §9 is blunt about the ceiling: CPT at 103M tokens will not install reliable
factual recall. So the local model is not allowed to *remember* what a filing said — it is
shown the passage and asked to copy from it. Everything in this document follows from
that one constraint.

```
catalog.sqlite ──► chunk ──► FTS5 (BM25)  ─┐
 18,009 docs      306,470     contentless  ├─► RRF ─► top-k passages
 103M tokens      chunks    ┌──────────────┘         (verbatim substrings)
                            └─ bge-small vectors                │
                                                                ▼
                                          provenance store ─► few-shot prompt
                                          (retains the exact          │
                                           bytes shown)               ▼
                                                    Ministral-14B → JSONL claims
                                                                      │
                                        quote resolver: located? ─────┤
                                          exact / whitespace / DROP   ▼
                                                              contracts.Briefing
                                                                      │
                                                                      ▼
                                                            markdown hand-off
```

The chain has one invariant, and every design choice below defends it:

> **A chunk is `clean_doc[start:end]` and nothing else.**

No stripping, no whitespace collapsing, no re-joining. If chunking rewrote the text, a
faithful quote from the model would fail verification and an unfaithful one would be
indistinguishable from it — the audit would report noise. This is why `ChunkStore` stores
offsets and reads through to `data/corpus/clean/`, rather than storing chunk text.

---

## 2. The index

| | |
|---|---|
| Chunks | **306,470** from 18,009 documents |
| Chunk geometry | 1,800 chars target, 300 overlap, snapped to paragraph breaks |
| Lexical build | **11 seconds** |
| Dense build | ~35 min at ~140 chunks/s |
| Index size | 257 MB (FTS5) + 235 MB (vectors) |
| Embedding model | `bge-small-en-v1.5` (33M params, 384 dims, 512-token window) |

**Chunk size is in characters, not tokens**, because offsets are the contract and tokens
are not addressable in the source text. Measured on this corpus, bge's wordpiece runs
~5.0 chars/token on filing prose, so 1,800 chars lands near 360 tokens — inside the 512
limit at p95, which matters because silent truncation would embed only the first half of
a passage while retrieval still returned all of it.

**FTS5 is contentless (`content=''`).** It keeps the inverted index and no copy of the
text. Duplicating 466 MB of filings into SQLite would create a second copy that could
drift from the one quotes are verified against. The cost is that the index is
build-once/rebuild — contentless FTS5 cannot be updated in place — which is acceptable
given a corpus rebuild is ~2.5h anyway.

**Vectors are a float16 memmap keyed by `chunk_id - 1`, pre-normalised.** 235 MB is small
enough to hold resident and brute-force, which removes an ANN index and its recall error
from the system entirely. Search is a dot product over the filtered rows.

### 2a. The 49GB stall — read this before touching the embedder

The dense pass was predicted at 34 min and instead collapsed to **under 1 chunk/s after
about 60 seconds**, which would have made it a multi-day job. Three wrong diagnoses were
made and discarded before the real one:

- *Pathological input?* No — the exact 512-chunk batch that took 17 minutes inside the
  build runs in 3.35s standalone.
- *Slow tokenizer?* No — 0.2s for those same 512 chunks.
- *macOS background QoS?* Partly, and this was asserted too early. It is a real but
  secondary effect (see 2b), not the cause.

The cause: with `padding=True` every batch pads to *its own* longest member, so
consecutive batches request shapes `(64, 431)`, `(64, 468)`, `(64, 448)`… Each distinct
shape allocates fresh Metal buffers that MLX caches and never gets to reuse.

```
it  0  cache= 11.35GB
it  1  cache= 19.51GB
it  2  cache= 27.77GB
it  3  cache= 38.18GB
it  4  cache= 48.94GB   <- against a 55.7GB working set; thrashing from here
```

**Fix:** round sequence length up to a multiple of 64 before padding
(`embed.LENGTH_BUCKET`), collapsing hundreds of shapes into eight, plus
`mx.set_cache_limit(4GB)` as a backstop. Cache then pins at ~4 GB and throughput holds at
**134 chunks/s flat over 45 consecutive batches**. The bucket histogram over a real slice
is `{384: 1, 448: 115, 512: 244}` — three shapes, not three hundred.

This is the same class of failure as the gradient-checkpointing finding in handoff §4a:
on this machine, exceeding the Metal working set does not fail loudly, it just gets
slow enough to look like the work is hard.

### 2b. Background execution penalty (unresolved)

With the cache bug fixed, an instrumented **foreground** run held 134 chunks/s dead flat.
A **backgrounded** run of the same code degraded to ~27 chunks/s. An isolated A/B on an
identical 2,048-chunk slice measured 143.6 chunks/s foreground vs 34.7 backgrounded.

The effect is real and reproducible; the mechanism was not isolated, and `taskpolicy -c`
does not expose a clamp that raises priority (only `utility`, which is lower). **Run long
MLX jobs in the foreground.** `build_dense(budget_seconds=...)` stops cleanly at a
checkpoint so a long build can be sliced under a wall-clock cap and resumed.

---

## 3. Search

Order is load-bearing: **hard filter, then score.**

```python
candidates = store.candidate_ids(tickers=[...], as_of=..., since=..., sections=[...])
bm25  = fts5 bm25 over candidates
dense = normalised dot product over candidates
fused = reciprocal rank fusion (k=60)
```

`as_of` is a **leakage boundary, not a preference**. A briefing dated 2024-06-01 must be
unable to see a filing published 2024-08-01, and "unable" means *not a candidate* — it is
excluded in SQL, upstream of any scoring. Every retrieval number in this project is only
meaningful because of that ordering. `tests/retrieval/test_store_and_search.py::
test_as_of_excludes_future_documents` is the regression guard.

**Two scorers, because they fail differently.** BM25 is exact-term and unbeatable for
`book-to-bill`, `days of inventory`, a specific dollar figure. Dense is paraphrase-tolerant.
Fusion is reciprocal-rank rather than a weighted score sum because BM25 returns negative
log-odds-ish values and cosine returns [-1,1]; any weighting between them is a
hyperparameter waiting to be overfit — and handoff §4d is the record of what overfitting
costs this project.

**FTS5 gotcha:** query terms must be individually quoted. FTS5 treats bare `-` and `.` as
syntax, and this corpus is full of `book-to-bill`, `10-K`, `R&D`. An unquoted query raises
`fts5: syntax error` on entirely ordinary analyst input.

---

## 4. Extraction

### 4a. The quote resolver

A quote is never trusted; it is **located**. `engine/extract/quotes.py` returns one of:

| Status | Meaning | Action |
|---|---|---|
| `exact` | literal substring of the retained passage | accept |
| `whitespace` | matches once whitespace runs are equivalent | accept, store the **document's** span |
| `fabricated` | appears in no shown passage | **drop the claim** |
| `too_short` | under 24 chars — appears by chance, proves nothing | drop |

Two properties make this more than bookkeeping:

1. **A repaired claim stores the document's bytes, not the model's rendering.** The
   whitespace path matches in normalised space but reads the span back out of the original
   string via an index map. So `Briefing.unverified_quotes()` is empty by construction for
   anything that survives — the check measures provenance integrity, not the model's typing.
2. **A fabricated claim is dropped, never downgraded to `INFERENCE`.** Downgrading would
   launder a fabrication into acceptable-looking output. The claim is discarded and the
   fabrication is *counted*.

`exact_quote_rate` is reported ahead of any repaired rate on purpose. If only the
post-repair number were tracked, a model that paraphrases everything and a model that
copies everything would score identically.

### 4b. Output format: JSON Lines, not one object

The extractor runs on a **base** checkpoint that has never been instruction-tuned, so the
prompt is few-shot and completion-shaped rather than an instruction. Output is one JSON
object per line because a single nested object is all-or-nothing — one malformed brace 900
tokens in loses the whole extraction. With JSON Lines a truncated line costs one claim.
Since format validity is a *measured* quantity, the format should not manufacture failures
that are really just length.

### 4c. Queries come from the taxonomy, not the model

Retrieval queries are generated from the sub-segment profile in `tools/taxonomy.py`.
Letting the model choose what to search for would make briefing coverage depend on the
model's priors — which is exactly what CPT is supposed to change, and would make a
before/after comparison uninterpretable because the two arms would have read different
documents. Fixed queries per segment mean **the CPT delta is measured on extraction,
holding retrieval constant.**

---

## 5. The briefing export

`engine/extract/export.py`. Designed around what the reader cannot do: the frontier model
has not seen the filings and cannot fetch them.

- Sourced and inference claims are in **separate sections**, not interleaved with footnote
  markers. Interleaved, the two look identical when skimmed, and that is how an inference
  becomes a reported fact.
- Every sourced claim shows **the actual span**, indented — not a citation pointer. A
  pointer the reader cannot follow is decoration.
- The audit block (exact rate, repairs, fabrications dropped, malformed lines) ships
  *inside* the export, so the reader sees the reliability of the extraction they are
  reading.
- An empty briefing says "no claim survived quote verification" explicitly, because
  silence must not read as "no news".

---

## 6. Measurements

See `artifacts/retrieval_eval.txt` and `artifacts/baseline/baseline.txt` for the
generated output; §7 of this document records the interpretation.

### 6a. Retrieval evaluation design

Relevance is defined by a **rule the retrievers have no access to** — an anchor pattern
plus a digit — rather than by hand or by a model. A model-labelled gold set scored against
model-based retrieval is circular: the embedder and the judge share a notion of similarity,
so dense wins by construction.

Two probe families, reported separately and never blended:

- **direct** — the query contains the anchor term (`"book-to-bill ratio and bookings"`).
  BM25 should win these. A mode that loses badly here is disqualified regardless of
  paraphrase performance, because real analyst queries contain the domain terms.
- **paraphrase** — the query avoids every anchor word and describes the concept
  (`"how many new customer orders were received compared with how much was shipped"`).
  Lexical match is unavailable by construction. **This is the only family where dense can
  justify its 35 minutes of index build.**

Probes with no gold chunk under their filter are reported as `skipped`, not averaged in
as failures.

### 6b. Baseline extraction design

Measured without labels: exact quote rate, fabrication rate, claim yield per passage,
sourced fraction, format validity. **Not** measured: whether the claims are the *important*
ones. That is salience, it needs judgment, and the honest place for it is a human or
frontier model reading the briefings — which the harness writes out for exactly that.

A good number here means "extraction is not lying". It does not mean "extraction is good".

Both **base** and **instruct** arms are run. Handoff §4b measured a 3.8% instruct BPC
penalty on domain text, but instruction-following is a different axis from domain fit, and
this loop asks for a format a base model has never been trained to produce. Running both
separates *the task is hard* from *the base checkpoint cannot follow the format* — and a
large gap is direct evidence for the Llama-Fin interaction in handoff §4e, which would
change how the CPT run should be configured.

---

## 7. Results

### 7a. Retrieval — hybrid earns its keep, dense alone does not

33 probes scored (3 skipped, all foundry — see 7b). `artifacts/retrieval_eval.txt`.

| mode | R@5 | R@20 | MRR | direct R@20 | **para R@20** | ms |
|---|---|---|---|---|---|---|
| bm25 | 0.576 | 0.727 | 0.392 | 0.944 | 0.467 | 45 |
| dense | 0.667 | 0.758 | 0.424 | 0.944 | 0.533 | 47 |
| **hybrid** | **0.727** | **0.848** | **0.435** | 0.944 | **0.733** | 55 |

**All three tie exactly on `direct` (0.944).** The expectation going in was that BM25
would win term-matching queries outright; it does not — dense finds them too. So the
direct family turns out to be uninformative for ranking the modes, and useful only as a
disqualifier.

**The entire result is in `paraphrase`: 0.467 → 0.733, +27 points for hybrid.** Dense
*alone* adds only +6.6 over BM25. The gain comes from fusion behaving as a **union, not an
average**: on the 6 probes where BM25 and dense disagree, each wins 3, and hybrid recovers
5 of the 6.

Their aggregate scores were identical in a first run (both 0.462 on paraphrase), which
looked like a bug — two retrievers cannot normally agree to three decimals. It was a
coincidence of them succeeding on disjoint probes. Worth remembering before trusting a
suspicious tie: the per-probe breakdown is in `artifacts/retrieval_eval.json`.

**Conclusion: keep the dense index, but only because of RRF.** A dense-only system would
have been a regression on R@20 relative to the effort.

### 7b. Two corpus defects the probes exposed

**1. `PROFILES[MEMORY].key_metrics` lists a metric its issuers never use.** The probe
anchored on `days of inventory` returned **zero** gold chunks across MU, WDC and STX. The
phrase is analog/microcontroller vocabulary — MCHP 181 chunks, SLAB 77, TXN 76, LSCC 35 —
while memory issuers write about inventory as *write-downs and carrying value*. This
matters beyond the eval: `taxonomy.py` metrics generate the retrieval queries, so the
memory profile currently sends the extraction loop hunting for language that is not there.
The probes were re-anchored (and a `days of inventory` probe added under ANALOG, where it
belongs); **the taxonomy itself is still wrong and should be fixed.**

**2. The corpus has no foundry coverage at all.** TSM, UMC and GFS have **0 chunks**.
Foundries are foreign private issuers filing 20-F, and the EDGAR builder collects
10-K/10-Q/8-K only. `SubSegment.FOUNDRY` and `PROFILES[FOUNDRY]` exist and will happily
frame a briefing for which retrieval can return nothing. The foundry probes are retained
so this keeps surfacing as `skipped` rather than being forgotten.

### 7c. Extraction baseline — the untrained numbers

**10 cases, all five sub-segments, `Ministral-3-14B-Base`, 138.5 s/case.**
`artifacts/baseline/baseline.txt`. **This is the denominator for the CPT comparison.**

| metric | value |
|---|---|
| claims accepted | **330** |
| claim yield / passage | 3.44 |
| **exact quote rate** | **0.910** |
| whitespace-repaired | 13 |
| **fabrication rate** | **0.023** (8 claims dropped) |
| quote too short | 10 |
| sourced fraction | 0.985 |
| **inference claims** | **5 of 330 (1.5%)** |
| malformed JSON lines | **0** |
| missing-field lines | **0** |
| quote audit vs retained bytes | **10/10 PASS** |

| case | segment | psg | claims | sourced | exact | fab |
|---|---|---|---|---|---|---|
| AMAT 2019-11-01 | equipment | 10 | 39 | 1.00 | **1.00** | 0 |
| LRCX 2023-05-01 | equipment | 9 | 20 | 1.00 | 0.83 | 2 |
| MU 2023-03-01 | memory | 10 | 35 | 0.94 | 0.91 | 0 |
| WDC 2022-08-01 | memory | 10 | 39 | 1.00 | 0.97 | 0 |
| NVDA 2024-05-01 | fabless | 10 | 34 | 1.00 | 0.89 | 1 |
| AMD 2022-11-01 | fabless | 10 | 34 | 1.00 | **1.00** | 0 |
| ADI 2024-02-01 | analog | 10 | 39 | 0.95 | 0.92 | 0 |
| MPWR 2023-08-01 | analog | 7 | 20 | 1.00 | 0.83 | 3 |
| ON 2025-05-01 | analog | 10 | 27 | 0.96 | 0.93 | 1 |
| CRUS 2021-08-01 | fabless | 10 | 43 | 1.00 | **0.77** | 1 |

**Format validity is perfect: zero malformed lines, zero missing fields, across 343 quote
attempts.** From a checkpoint with no post-training at all. The few-shot JSON Lines format
carried it entirely — this was the biggest open risk going in and it is simply not a
problem.

**The fabrication guard is not decorative.** 8 claims invented evidence and were dropped.
Without the resolver those would have reached the frontier model as citations, and nothing
downstream could have caught them. 10 more quotes were too short to be evidence. So **18
of 343 attempts (5.2%) would have been unverifiable assertions dressed as quotations.**

**Exact-quote rate varies far more by case than the aggregate suggests: 0.77 to 1.00.**
The 0.910 headline hides that CRUS is a coin-flip worse than AMAT. Whatever CPT is measured
on, it should be measured per-case — an aggregate move of a couple of points could be one
case changing.

**The model does not reason: 5 inference claims out of 330 (1.5%) — and the real rate is
lower still.** At least one of the five is verbatim revenue-recognition boilerplate lifted
from a filing and mis-tagged `inference`, which the schema cannot catch (an INFERENCE claim
carries no quote to verify, by design). The genuine ones are respectable — *"Revenue
guidance is down 20% from Q2 2022, but operating expenses are down 10%"*, *"The company is
cutting capex to address the near-term environment"* — but there are roughly four of them
across ten briefings.

The sub-segment framing from `taxonomy.py` is in every prompt and is producing almost no
judgment about the cycle. **This is the clearest statement available of what CPT would have
to change**, and a much sharper target than "extraction quality" in the abstract. It also
notes a blind spot in the audit: the mechanism guards sourced claims rigorously and inference
claims not at all, which is correct (there is nothing to verify) but means a model that
mislabels copies as inference degrades the briefing without moving any metric.

### 7d. Three defects found by reading the output, none visible in the metrics

**1. Batching silently discarded 80% of the evidence.** At 5 passages per generation the
model spent its whole 1100-token budget on the *first* passage — ten "claims" split out of
one bulleted risk factor — then degenerated into copying source text. Passages 2-5 were
never reached. **The audit reported a flawless 100% exact-quote rate on a briefing that
ignored eight of ten retrieved passages.** This is the failure mode to fear: a provenance
metric that looks perfect *because* most of the evidence was skipped. Fixed by one passage
per generation (`batch_passages=1`, `max_tokens=400`), which makes coverage structural
rather than something the model chooses. After the fix all 10 passages contributed (2-6
claims each, 40 total).

**2. Boilerplate crowds out news.** Risk-factors passages match almost any query lexically
and are the most-recycled text in a filing (the corpus dedup pass found 18.4% within-issuer
near-duplicates, concentrated there). `max_per_section=3` now caps them. Even so the
retrieved mix is boilerplate-heavy — 3 risk_factors + 3 10-K `business` against only 1
MD&A — and MD&A and 8-K/EX are where the quarter's actual news is. **Section weighting is
worth revisiting.**

**3. Zero inference claims, in both runs.** The model produced not one, despite the
few-shot demonstrating them. It only copies. Many accepted claims are near-verbatim
restatements of their own quote — no compression, no judgment. This is the most direct
evidence yet about what CPT would need to change: the taxonomy framing is in the prompt and
is having no effect on whether the model reasons about the cycle.

Defects 1 and 2 were fixed before the baseline was run, so the numbers in 7c describe the
loop as it now stands. Defect 3 is a property of the model, which is the point.

**So: extraction is not lying. Whether it is useful is only partly answered.** Claims drawn
from MD&A and earnings releases are genuinely good — a single CSP at 22% of NVDA revenue;
gross-margin drivers; inventory provisions $681m → $442m net at 2.4pp of margin. Claims
from risk factors and the 10-K business section are noise. The section quotas moved the
high-value share from 40% to 65%; the remaining 35% is still mostly wasted context.

### 7e. Cost

| | |
|---|---|
| Retrieval per query | 45-55 ms |
| Briefing, 10 passages, base 14B | **138 s** typical (312 s worst case observed) |
| Generation throughput | ~19 tok/s at a 5k-token prompt |
| Baseline, 10 cases, base arm | **23 min** across 4 resumable foreground runs |

The instruct arm was **not** run. The base checkpoint's format validity came in perfect, so
the question that arm was designed to answer — *can a base model follow this format at all,
per Llama-Fin §4e* — is already answered in the affirmative and the comparison lost most of
its value.

---

## 8. Extending beyond semis: 26 software/internet tickers (session 3)

The user trades semis and tech together and asked whether the same machinery could cover
both. It could, mechanically: measuring it (not assuming it) was the point of this
session. **85% of the retrieval/extraction code has zero semiconductor-specific logic in
it** — chunking, embedding, quote-checking, claim parsing and provenance never reference
a domain concept. What *is* domain-specific is concentrated in two small files
(`taxonomy.py`, the sub-segment cycle framing, and `probes.py`, the retrieval eval), plus
— found only once this session actually touched it — one prompt string that had "semis"
baked in where it shouldn't have been (§8c).

### 8a. Scope

26 US-listed tickers across 6 new `SubSegment` business-model categories, added to the
same enum as the five semiconductor ones. `configs/universe/tech_v1.yaml` is a **separate,
corpus-only universe file** — it is not wired into `quant build-dataset` / `quant
validate`. Handoff §5 decouples the quant universe from the text corpus on purpose, and
the Phase 1 gate (failed, §4d) describes a cross-section specific to `semis_v1.yaml`;
folding mega-cap tech into that cross-section would be a real, separate decision, not a
side effect of this file existing.

| Category | Tickers |
|---|---|
| `TECH_AD_PLATFORM` | GOOGL, META |
| `TECH_DEVICE_ECOSYSTEM` | AAPL |
| `TECH_CLOUD_SAAS` | MSFT, ORCL, CRM, ADBE, INTU, NOW, WDAY, SNOW, DDOG, NET, MDB, TEAM, IBM, CSCO, PLTR |
| `TECH_CYBERSECURITY` | PANW, CRWD, ZS |
| `TECH_COMMERCE_PLATFORM` | AMZN, SHOP, PYPL, UBER |
| `TECH_STREAMING_MEDIA` | NFLX |

All 26 resolved to SEC CIKs cleanly — unlike the semis FOUNDRY gap (TSM/UMC/GFS, foreign
private issuers filing 20-F), none of these tickers had that problem. One did surface a
narrower version of it: see §8f on SHOP.

### 8b. Corpus build

```
STAGE 1 discover:  26 issuers -> 6,109 filings found,       ~17 min
STAGE 2 fetch:     5,857 documents fetched (rate-limited),  ~25 min
STAGE 3 clean:     -> 7,626 section documents,               <1 min
STAGE 4 dedupe:    1,644/7,626 removed (21.6%),               <1 min
STAGE 5 pack:      5,982 documents -> 45,338,624 tokens,      ~1 min
```

**~50 minutes total**, additive to the existing corpus (same `catalog.sqlite`, same
`data/corpus/clean/`) rather than a separate build. New totals: **23,991 packed documents,
148,801,271 tokens** (up from 18,009 / 103.5M).

**Dedup rate 21.6%, vs 18.4% for semis.** Higher, and worth a specific note: tech filings
— particularly SaaS companies' risk-factor and RPO-explanation paragraphs — recycle
boilerplate at least as aggressively as semis filings do, and this session's extraction
findings (§8g) trace a real failure mode back to exactly that recycled language surviving
the near-duplicate filter as two "not quite identical" passages.

### 8c. The prompt was more semis-specific than the taxonomy was

Before running anything on a tech ticker, a check of every file that referenced
`SubSegment` for hardcoded assumptions (`grep`-checked: no exhaustive 5-way switches
exist, so extending the enum was safe) also turned up something the taxonomy correction
alone would not have caught: **`engine/extract/prompts.py`'s system rules literally said
"You are extracting claims from semiconductor filings,"** and both few-shot examples were
semis-only (an AMAT equipment quarter, an MU memory quarter). On a *base* checkpoint —
which continues patterns rather than following instructions (handoff §4b) — that sentence
and those examples are exactly the kind of thing that biases extraction toward chip-cycle
vocabulary even when the document in front of it is a Salesforce 10-K.

Fixed: the rules sentence is now domain-neutral, and a third few-shot example
(fictional ticker `XSAS`, deliberately not a real company, so it cannot be mistaken for a
verified fact rather than a formatting demonstration) demonstrates the same JSON Lines /
quoting discipline on SaaS-flavoured claims — net revenue retention, RPO.

### 8d. Incremental indexing (a real capability gap, not scope creep)

The first attempt at re-indexing would have re-chunked and re-embedded the entire corpus
from scratch — throwing away the ~35 minutes already spent embedding the 306,470 semis
chunks. Fixed properly rather than eaten as a one-time cost, because the corpus will grow
again:

- **`build_lexical(reset=False)`** now diffs against `store.indexed_docs()` and chunks
  only documents not already in the index. Verified on the real corpus: correctly
  identified 18,009/23,991 documents as already-indexed and chunked only the 5,982 new
  ones, in **6.2 seconds**.
- **`ChunkStore.grow_vectors()`** extends the vector memmap to a new row count while
  preserving every embedding already computed — `np.memmap` cannot be resized in place, so
  this writes a new file, copies the old rows across, and atomically replaces the
  original. Verified with a synthetic two-wave test before touching the real 306k-row
  file: old rows byte-identical after growth, new rows zeroed and ready to fill.
- **`build_dense`** now grows in place instead of wiping and rebuilding on a size mismatch.
  Measured on the real run: it correctly recognised 306,470/442,770 chunks were already
  embedded and resumed from there — the incremental embedding pass covered only the
  136,300 new chunks, in **~12 minutes** at the same ~145 chunks/s established in session
  2, instead of the ~50 minutes a full rebuild would have cost.

### 8e. Taxonomy correction pass — measured, not assumed, exactly like MEMORY

`PROFILES_TECH` was written *before* any tech filing existed in the corpus, explicitly
flagged in the source as hypotheses (see the comment above `PROFILES_TECH` in
`taxonomy.py`). Once the corpus existed, the same discipline that caught MEMORY's "days of
inventory" bug in session 2 was applied again — scan real chunks for the hypothesised
vocabulary, keep what is measured, fix what is not:

| Segment | Hypothesis | Measured | Verdict |
|---|---|---|---|
| AD_PLATFORM | CPM (cost-per-mille) is a key metric | **5 hits** across 9,083 GOOGL/META chunks | Wrong — CPM is earnings-call language, rarely stated as a filing figure. Swapped for MAU (**236** gold chunks, ahead of DAU's 197) |
| DEVICE_ECOSYSTEM | "services revenue" | **16 hits** across 5,048 AAPL chunks | Wrong phrasing — Apple's filings say **"Services net sales"** (**136** hits). Same class of bug as MU/"days of inventory": the company's own vocabulary, not the generic term |
| CLOUD_SAAS | 4 metrics listed (NRR, RPO, billings, seats) | deferred revenue (**3,841**) and services revenue (**2,330**) are the two most common figures, ahead of all four originally listed | Both added; NRR itself measured weakest of the group at 318 — still usable, but many issuers narrate retention rather than stating a clean number |
| CYBERSECURITY | "new logo growth" is a key metric | **1 hit** across 13,711 PANW/CRWD/ZS chunks | Dead vocabulary. These filings talk in terms of retention, not acquisition: billings 548, ARR 359, renewal rate 159. Re-anchored on billings and renewal rate |
| COMMERCE_PLATFORM | GMV, take rate | 86 / 176 hits across 17,470 chunks | Confirmed, no correction needed |
| STREAMING_MEDIA | subscriber net adds, churn, ARPU | 129 / 84 / 84 hits across 5,484 NFLX chunks | Confirmed, no correction needed |

Two of six segments needed a real correction before their probes or extraction queries
were trustworthy — the same failure rate, roughly, as the five original semis segments
(one of which, MEMORY, needed the same kind of fix). **Writing plausible-sounding metrics
from general knowledge and skipping the measurement step would have been wrong 33% of the
time in this sample.** That is the entire argument for not skipping it.

### 8f. Retrieval evaluation

72 probes total (36 semis + 36 new tech, 3 direct/paraphrase pairs per new segment) after
the correction pass above; 3 skipped (all FOUNDRY, the pre-existing 20-F gap — unchanged
by this session).

| mode | R@5 | R@20 | MRR | direct R@20 | para R@20 |
|---|---|---|---|---|---|
| bm25 | 0.522 | 0.681 | 0.388 | 0.833 | 0.515 |
| dense | 0.580 | 0.725 | 0.434 | 0.917 | 0.515 |
| hybrid | 0.536 | 0.739 | 0.387 | 0.861 | 0.606 |

Broadly consistent with the semis-only numbers in §7a — hybrid still wins on paraphrase
recall, dense still does the heavier lifting there than BM25 alone. **All 36 tech probes
now resolve to a real, non-empty gold set** (before the §8e correction pass, `newlogo_*`
had 0-1 gold chunks and `cpm_*` had essentially none). Some individual tech probes still
miss at k=20 even on direct queries (e.g. `cyberrenewal_direct`, `takerate_direct`) — that
is a ranking-quality question for future tuning, categorically different from a probe
built on vocabulary that does not exist, which is what this section's correction pass was
for.

### 8g. Extraction baseline: 10 cases across the 6 new segments

`TECH_CASES` in `baseline.py`, same discipline as `DEFAULT_CASES` — span every segment,
vary the period rather than clustering on one moment. `--cases tech` on the existing,
unmodified `extract baseline` CLI. **~26 minutes**, base arm only, foreground, in 3
resumable slices.

| case | segment | claims | sourced | exact | fabricated |
|---|---|---|---|---|---|
| GOOGL 2023-02-01 | ad_platform | 28 | 1.00 | 0.93 | 2 |
| META 2023-05-01 | ad_platform | 38 | 1.00 | 0.97 | 1 |
| AAPL 2023-11-01 | device_ecosystem | 31 | 1.00 | 0.97 | 1 |
| **CRM 2024-05-01** | **cloud_saas** | 33 | 1.00 | **0.82** | **6** |
| **NOW 2023-08-01** | **cloud_saas** | 35 | 1.00 | **0.85** | **5** |
| PANW 2023-08-01 | cybersecurity | 33 | 1.00 | 0.92 | 2 |
| CRWD 2024-03-01 | cybersecurity | 41 | 0.98 | 1.00 | 0 |
| AMZN 2023-02-01 | commerce_platform | 37 | 0.97 | 0.90 | 0 |
| PYPL 2023-05-01 | commerce_platform | 26 | 0.96 | 0.96 | 1 |
| NFLX 2023-01-01 | streaming_media | 43 | 0.95 | 0.93 | 3 |

| metric | tech (10 cases) | semis (10 cases, §7c) |
|---|---|---|
| exact quote rate | **0.924** | 0.910 |
| fabrication rate | **0.057** | 0.023 |
| inference claims | 1.4% (5/345) | 1.5% (5/330) |
| quote audit vs retained bytes | 10/10 PASS | 10/10 PASS |

**The aggregate exact-quote rate is not worse — it is a hair better than semis.** The
aggregate hides the real story: **11 of 21 total fabrications (52%) come from just two of
the ten cases, both CLOUD_SAAS** (CRM and NOW). Every other segment matches or beats the
semis baseline outright — CRWD hit 100% exact / 0 fabricated and even produced an
inference claim, the only tech case that did. This is not "tech extraction is worse than
semis extraction"; it is "one specific business model's filing style is harder," and the
aggregate number would have hidden that if the per-case table had not been checked
(handoff §10d already warned against trusting the aggregate for exactly this reason).

**A case-selection mistake, caught and fixed rather than reported around.** The original
tenth case was SHOP @ 2023-05-01, which returned 0 claims in 0 seconds. Investigating
found the cause: SHOP's indexed corpus covers only 2025-02-11 .. 2025-12-02 — a narrower
version of the FOUNDRY 20-F gap (Shopify is a Canadian filer; most of its EDGAR history is
under a different convention than the 10-K/10-Q the builder collects). The `as_of` filter
did exactly what it is supposed to do — correctly refused to show the model a 2025 filing
when asked about 2023 — the mistake was picking a case whose ticker had no coverage in
that window at all. Replaced with PYPL (clean 10-K/10-Q history from its 2015 spinoff),
and the bad zero-record was removed from `cases.jsonl` rather than left to quietly
understate the fabrication rate (a 0-attempt case would not have moved the percentage, but
would have been a data point that meant nothing sitting next to ones that did).

**Root cause of the CLOUD_SAAS fabrications — first diagnosis was wrong, and the
correction is the more important finding. Recorded here rather than quietly fixed,**
because getting this right changed what was worth building next (§9).

The first pass (checking only CRM, and only a 30-character prefix match) concluded the
model was substituting a "generic textbook definition" of RPO for what the filing
actually said. That conclusion does not survive a full-word-overlap search. Checked
properly — searching for the longest contiguous word run of each fabricated quote
anywhere in its *own* source document, not just near where a crude anchor first
matched — the real CRM 10-K text is:

> *"**Our** remaining performance obligation represents all future revenue under contract
> that has not yet been recognized as revenue and includes unearned revenue and unbilled
> amounts."*

The model's "fabricated" version is that sentence with exactly one word dropped: **"Our."**
23 of 24 words, contiguous, verbatim. Tracing every fabrication in both CRM and NOW
(§8e) the same way — word-overlap, not prefix-match — found the same pattern in 10 of 11
cases: **92-99% word overlap**, one or two words dropped from an otherwise character-perfect
copy of a long (17-81 word) sentence. This is **near-verbatim drift on long spans, not
knowledge substitution.** A third, independent confirmation came from FOUNDRY (§9b below):
TSM's fabrications traced the same way showed 29/30 to 64/65 word overlap on similarly
long, numeric-dense sentences — the mechanism generalises across a completely different
company and filing type.

One fabrication was genuinely different: a passage that was cut off mid-sentence
("...the Company had a valuation allowance of $") got *completed* with two dollar figures
present nowhere in the shown text. That is a distinct, more concerning mechanism —
completing a truncated passage rather than mis-copying a complete one — and it pointed to
a real, fixable cause: **§9a**.

Revised explanation for why CLOUD_SAAS concentrates the problem: not that its boilerplate
invites substitution, but that its filings' load-bearing sentences (RPO explanations,
attrition disclaimers) run unusually long and dense — 30-80 words is a lot of surface
area for a one-word drop, and CRM/NOW's writing style produces more such sentences than a
typical semis MD&A does.

### 8h. What this does and does not settle

**Settled:** the architecture generalises. 26 tickers, 6 new business-model categories,
45.3M new tokens, in one session, on infrastructure built for a different domain, with two
real (not hypothetical) bugs caught by measurement before they could quietly bias
retrieval or extraction.

**Not settled, as of this section:**
- ~~CLOUD_SAAS's fabrication mechanism is a hypothesis from one traced case~~ — checked
  against NOW, corrected, and confirmed against a third, independent case (FOUNDRY/TSM).
  See the correction above and §9b. **Settled: near-verbatim word-drop on long sentences,
  not knowledge substitution.**
- ~~The boilerplate/section-weighting fix was not re-tuned for tech~~ — measured (§9d):
  the semis-tuned caps already achieve 73% high-value retrieval on SaaS tickers,
  *better* than semis' own 65%. No re-tuning needed; re-tuning without this measurement
  would have been change for its own sake.
- ~~Foundry-equivalent gaps may exist elsewhere~~ — swept systematically (§9a): 12
  tickers were foreign 20-F filers with zero 10-K/10-Q history, now backfilled. One
  (IFNNY) has a real, permanent gap (all its filings predate the corpus's 2010 start).
  One dead ticker symbol (SGH → PENG) fixed in the universe config.
- **No tech-specific baseline exists for the instruct arm**, same reasoning as §7e. Still
  open.
- **A live, index-free retriever now exists** (§9e) that was not built as of this
  writing, and was not anticipated to be needed until a real gap in ticker coverage
  motivated it.

---

## 9. Session 3 continued: fixes, backfill, and readiness work

Everything below followed from the user asking, in plain terms, to act on §8h's open
items. Ordered as they were done, because later items depended on earlier ones' findings.

### 9a. Twelve tickers had zero 10-K/10-Q history — because they don't file one

A systematic per-ticker date-coverage audit (`MIN(doc_date)`/`MAX(doc_date)` grouped by
ticker across the whole index) found 13 tickers whose earliest indexed chunk was
suspiciously recent. Checked one by one against SEC's own submissions API rather than
guessed at:

- **12 are pure 20-F filers** (ASML, ASX, CAMT, GFS, HIMX, IFNNY, IMOS, NVMI, STM, TSEM,
  TSM, UMC) — foreign private issuers that have **never filed a 10-K or 10-Q**, confirmed
  against SEC's own submissions data for every one of them. This is the FOUNDRY gap from
  §6/§8, and it turned out to be four times bigger than scoped: not just TSM/UMC/GFS, but
  9 more tickers across EQUIPMENT, FABLESS and ANALOG that were silently missing from the
  corpus with no warning anywhere.
- **1 (SHOP) is a dead-end**, already covered by §8h's investigation — Shopify only began
  domestic 10-K/10-Q filing 2025-02-11; before that it filed 40-F/6-K, a Canadian-format
  wrapper this project does not parse (§9a below explains why 20-F was tractable and 40-F
  was not).
- **SGH was a stale ticker symbol**, not a data gap at all — `data/delisted.py` already
  documented SGH → Penguin Solutions (PENG) as a rename; the universe config still listed
  the dead "SGH" symbol, which silently resolved to no CIK. Fixed to "PENG" — and a
  pre-existing duplicate PENG entry under the wrong segment (`analog` instead of
  `memory`) was found and removed in the same pass.

**20-F support was built, not just diagnosed.** `find_sections`/`extract_narrative`
already dispatched patterns by structure, not by hardcoded 10-K assumptions, so adding a
second pattern set (`SECTION_PATTERNS_20F`: Item 3 = risk info, Item 4 = business, Item 5
= MD&A, Item 11 = market risk — verified against a real TSM 20-F, not guessed from memory)
was the right shape of fix. It surfaced a second, independent bug: 20-F filings cite their
own item numbers constantly in body text ("see Item 4 for further discussion"), and the
existing "last occurrence wins" heuristic — correct for 10-K, which doesn't do this — was
picking a mixed-case cross-reference over the genuine, ALL-CAPS header. Fixed with an
opt-in `prefer_caps` flag on `find_sections`, scoped to 20-F only so the already-working
10-K path (18,000+ documents) is untouched. Reproduced as a synthetic test fixture before
trusting the fix, then verified against the real TSM document.

**Result: ~50 minutes end to end, +15,736,832 tokens, corpus at 164,541,463 tokens /
54,392 documents.** Dedup rate for this slice was 10.6% — lower than either semis (18.4%)
or tech (21.6%), because 20-F is annual-only, so there's a full year rather than a quarter
between filings for boilerplate to recycle across. Index extended incrementally (same
`grow_vectors`/`build_lexical(reset=False)` machinery from §8d) to 488,400 chunks.
**All 72 retrieval probes now resolve to real gold evidence — zero skipped**, versus 3
FOUNDRY probes with no evidence base at all before this.

### 9b. The real fabrication mechanism (corrected) generalises to FOUNDRY too

A TSM smoke test (20-F, foundry, the most textually different case tried all session)
came in at 69% exact / 4 fabricated — the weakest single case measured. Traced with the
same word-overlap method that corrected the CLOUD_SAAS finding (§8g): every one of the 4
fabrications showed 29/30 to 64/65 word overlap — 94-97% verbatim, one or two words
dropped from long (30-65 word), numeric-dense sentences about YoY revenue and margin
percentages. **Same mechanism, third independent confirmation** (CRM, NOW, now TSM),
across three different companies, filing types, and sub-segments. Not a new bug; expected
variance given FOUNDRY's particularly long compound sentences.

### 9c. A real, structural fix for the one genuinely different fabrication

The valuation-allowance case in §8g's original CRM trace — a passage truncated mid-sentence
("...had a valuation allowance of $") that got completed with figures shown nowhere —
turned out not to be a one-off. **Measured directly: 20 of 30 sampled passages (67%)
across CRM, NOW and AMAT ended mid-sentence.** `chunk.expand()`'s existing paragraph-snap
only fires when a newline falls within the last 400 characters of the window, which dense
running prose (accounting notes, especially) often does not have.

Fixed with a bounded sentence-boundary trim: if a passage's end doesn't land on
terminal punctuation, search backward up to 300 characters (never below the hit's own
end, so the located span is never cut) for the last sentence end and trim there. Still a
pure substring operation — the exact-quote invariant is untouched. **Measured after the
fix: 4 of 30 (13%)**, down from 67%. The remaining 4 are cases with no sentence-ending
punctuation anywhere in the lookback window (tables, dense enumerations) — correctly left
alone rather than shrunk to nothing.

**Re-baselined the full 10-case tech set to measure the real effect, not assume one:**

| metric | before (§8g) | after |
|---|---|---|
| exact quote rate | 0.924 | **0.939** |
| fabrication rate | 0.057 | **0.045** (21 → 17 fabricated, -19% relative) |
| claims accepted | 345 | 368 |

Modest, and exactly as predicted going in: the truncation fix addresses one specific
sub-mechanism (1 of 6 traced CRM fabrications), not the dominant near-verbatim-drift
mechanism (§8g/§9b), which this fix does not touch and nothing in this session addressed.
**Whether CPT would help the dominant mechanism — more exposure to verbatim copying of
these exact filings — is an open, testable question for the eventual pilot, not answered
here.**

### 9d. Section weighting for SaaS: measured, no change needed

Requested explicitly; the honest answer is a null result, reported rather than papered
over with an unneeded change. Measured the retrieved section mix for 6 CLOUD_SAAS tickers
(CRM, NOW, SNOW, DDOG, WDAY, ORCL) both uncapped and with the existing semis-tuned
`section_caps`:

| | uncapped | with current (semis-tuned) caps |
|---|---|---|
| high-value share (mdna + earnings) | 57% | **73%** |

73% is *better* than semis' own post-cap number (65%, §7d). The caps built for chip-filing
boilerplate generalise to SaaS boilerplate without modification. The RPO/attrition
fabrication problem (§8g, §9b) turned out to be unrelated to section selection — it comes
from *within* mdna itself, which is exactly the section the caps already prioritise —
so no section-weighting change would have touched it regardless.

### 9e. A live, index-free retriever — `LiveEdgarRetriever`

Session 1's handoff flagged "no retrievers yet" as a gap; every retriever built since
(`CorpusRetriever`, session 2) has required a document already be in the pre-built index.
This is the first retriever that needs no index at all: given a ticker and an `as_of`
date, it resolves the ticker to a CIK live, pulls the most recent qualifying filings
directly from `data.sec.gov`, and runs them through the same `extract_narrative` used at
corpus-build time (form-type-aware, so a 20-F issuer gets real sections rather than a
silent empty result).

**Why not a paid news API**, which `SourceType.NEWS`/`WEB` in the contracts clearly
anticipate: every real one costs money or needs an account, and spending money without
asking is off the table. SEC's submissions and filing-index APIs are free, unauthenticated,
and public domain — the same trust boundary the whole corpus already relies on.

**8-K exhibit-following was added, not left as a known gap.** A first live test against
COST (a ticker touched nowhere else this session, a deliberate out-of-band check) returned
only 8-K cover pages — the exact ~0.5-1.1k-token-cover-page problem the handoff already
documents for corpus building. Rather than document it as a limitation, the existing,
already-tested `sources/edgar.py` exhibit-detection regex (`EXHIBIT_PATTERN`,
`SKIP_FILES`, `FULL_SUBMISSION`) was reused directly, so the live retriever and the
corpus-build pipeline agree on what counts as an exhibit rather than maintaining two
independent definitions. Re-verified live: the same COST call now returns real earnings
releases ("$57.4B Net Sales +9.1% Growth...") alongside the cover pages.

Tested with a fake `EdgarClient` (no live network calls in the test suite) covering the
`as_of` leakage boundary, unresolvable tickers, partial fetch failures, 20-F dispatch,
exhibit-following, and CIK-resolution caching — 13 tests.

**What this is not:** a replacement for the corpus. It fetches only the most recent few
filings per call, live, on every use — no retrieval quality (BM25/dense/hybrid, §7a/§8f)
applies to it, and there is no historical depth. It exists for the case the corpus can't
serve: a ticker outside the ~95 pre-indexed names.

### 9f. Two readiness pieces, explicitly not validated results

Both from the original session-1 handoff's outstanding-work list (§4e), both genuinely
buildable and testable without a trained model or real market data existing yet — and
both explicitly **not** run against anything real, because nothing real exists yet to run
them against. Treat "built and tested" as exactly that, not as "measured to help."

**LAP (Lookahead Propensity) scoring** — `engine/eval/lap.py`. From the literature
review's single most actionable finding (arXiv 2512.23847): a date-only recall probe that
estimates whether a model already knows a setup's outcome from pretraining, independent
of any reasoning. Implemented as a forced-choice UP/DOWN probe against a (ticker, as_of,
horizon) triple with no retrieved context — exactly what a real recommendation would never
be shown, on purpose. Wired into `gate.py` as an optional, non-blocking diagnostic: an
elevated LAP score becomes a **warning attached to an otherwise-passing gate result**, not
a new pass/fail threshold (a hard LAP threshold would just move the p-hacking surface
§4d's pre-registration discipline exists to close). 14 tests on the scoring logic itself
(chance-level scoring, unparsed-response handling, contamination-set partitioning); 9 more
on the gate integration specifically, confirming a contaminated-looking result still shows
PASS on its pre-registered criteria while carrying a warning a human would actually read.
**No real setup has been probed** — there is no LAP number to report, because no CPT model
and no journal of real recommendations exist yet.

**SFT/CPT interleaving** — `engine/training/interleave.py`. The Llama-Fin mitigation
(handoff §4e) for the finding that sequential CPT-then-SFT collapsed instruction-following
from 7.8 to ~1.0 on MT-Bench: mix individual SFT-format examples into the same step
sequence CPT already iterates, at a configurable ratio, rather than running two separate
phases. Deterministic scheduling (not stochastic — a coin-flip schedule would make two
runs with the same seed diverge in which tokens get how much gradient signal, which
`data_iter.py`'s own resume-determinism requirement already rules out elsewhere), fully
resumable (`InterleaveState` persists into `TrainState`, backward-compatible with
checkpoints saved before this existed). **`CPTConfig.sft_interleave_ratio` defaults to
0.0**, at which the interleave path is provably a no-op — verified directly:
`interleave()` at ratio 0.0 produces output byte-identical to plain `ShardIterator`
iteration, batch for batch. 10 tests cover the mixing schedule, cycling behaviour, and
resume correctness, all pure Python/numpy — no MLX involved, and **no actual training run
was executed**, since no SFT dataset exists yet (still true as of this writing) and a real
run is the multi-hour pilot explicitly deferred pending evaluation of a newer candidate
base model.

### 9g. The trade journal: found untested, fixed, given a CLI

Unrelated to the tech-corpus work, prompted by the user asking to start paper trading.
The handoff's own state table claimed the journal was "tested end-to-end" since session
1; **no test file for it existed anywhere in the repository.** 12 tests written (short
vs. long return-sign inversion, open/closed trade partitioning, non-trades correctly
excluded from the trades table but retained for calibration, performance-by-arm
separation) — all passed against the *existing, unmodified* code, so the claim was
optimistic rather than the implementation being wrong, but the discrepancy is worth
recording plainly.

Three CLI commands added (`journal record`, `journal close`, `journal status`), tested
end to end against a throwaway database including both the FLAT (non-trade) and directional
paths. Deliberately, **no trade was recorded** as part of this work — inventing one to
"start" the track record would defeat the entire point of a prospectively-recorded
journal, which is that decisions are logged before the outcome is known.
