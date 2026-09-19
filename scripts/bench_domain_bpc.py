"""Phase 0.6 -- measure how well each candidate already models semiconductor filings.

Throughput tells you what a model costs. This tells you what you are buying: how much
domain knowledge the base checkpoint already has, which is precisely what CPT has to
improve on. It is also the same quantity the reference DAPT study optimised
(arXiv 2512.12384 measured domain validation loss), so it is the metric our Gate 3a
will track -- measured here on the untrained baselines.

WHY BITS PER CHARACTER, NOT PERPLEXITY
Perplexity is per *token*, so it is not comparable across models with different
tokenizers -- and our candidates differ by up to 15% in tokens-per-character on filing
text. A model with a coarser tokenizer gets an artificially flattering perplexity
because each token carries more text. Bits per character normalises this away:

    BPC = total_negative_log_likelihood_in_bits / total_characters

Lower is better, and the number is directly comparable across tokenizers.

CAVEATS, stated up front because they bound the conclusion:
  * All candidates are 4-bit quantised. Quantisation quality varies by conversion and
    adds noise; treat differences under ~2% as inconclusive.
  * Some candidates are base checkpoints and some are instruct. Instruct tuning shifts
    the distribution and usually costs a little BPC on raw prose. Compare like with like
    where possible; the report labels which is which.
  * Recent filings are used deliberately (see --corpus), so this leans towards measuring
    genuine domain modelling rather than memorisation of documents seen in pretraining.
"""

from __future__ import annotations

import os
import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

UA = os.environ.get("ARGUS_SEC_UA", "ARGUS Research contact@example.com")

# Recent filings spanning sub-segments (fabless, equipment, memory, analog) so the score
# is not dominated by one issuer's house style. URLs are resolved from the EDGAR
# submissions API rather than hardcoded -- guessed accession paths 404.
ISSUERS = [
    ("NVDA", "1045810"),   # fabless
    ("AMAT", "6951"),      # equipment
    ("MU", "723125"),      # memory
    ("ADI", "6281"),       # analog
]


def latest_filing_urls(forms=("10-Q", "10-K"), per_issuer=1):
    """Resolve real primary-document URLs for the most recent filings per issuer."""
    out = []
    for ticker, cik in ISSUERS:
        try:
            meta = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json",
                headers={"User-Agent": UA}), timeout=30).read())
        except Exception as e:
            print(f"  WARN submissions lookup failed for {ticker}: {type(e).__name__}")
            continue
        r = meta["filings"]["recent"]
        taken = 0
        for form, acc, doc, dt in zip(r["form"], r["accessionNumber"],
                                      r["primaryDocument"], r["filingDate"]):
            if form in forms and doc.endswith((".htm", ".txt")) and taken < per_issuer:
                out.append((f"{ticker} {form} {dt}",
                            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                            f"{acc.replace('-', '')}/{doc}"))
                taken += 1
    return out


def fetch_text(url: str) -> str:
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30
    ).read().decode("utf8", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_corpus(cache: Path) -> str:
    if cache.exists():
        return cache.read_text()
    parts = []
    for label, url in latest_filing_urls():
        try:
            t = fetch_text(url)
            if len(t) < 5000:
                print(f"  WARN {label} too short ({len(t)} chars), skipping", flush=True)
                continue
            parts.append(t)
            print(f"  fetched {label}: {len(t):,} chars", flush=True)
        except Exception as e:
            print(f"  WARN could not fetch {label}: {type(e).__name__}", flush=True)
    text = "\n\n".join(parts)
    if len(text) < 10_000:
        raise RuntimeError(
            f"corpus is only {len(text)} chars -- refusing to report BPC on a corpus "
            "this small; check EDGAR fetching above")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def bits_per_char(model, tokenizer, text: str, window: int, stride: int) -> dict:
    """Sliding-window teacher-forced NLL over `text`, normalised per character.

    Non-overlapping scoring would penalise tokens at window boundaries, which have no
    left context. We therefore stride by less than the window and only score the tail
    of each window -- every scored token sees at least (window - stride) tokens of
    context.
    """
    ids = tokenizer.encode(text)
    n_tokens = len(ids)
    n_chars = len(text)

    total_nll = 0.0
    n_scored = 0
    start = 0
    t0 = time.perf_counter()

    while start < n_tokens - 1:
        end = min(start + window, n_tokens)
        chunk = ids[start:end]
        if len(chunk) < 2:
            break

        inp = mx.array([chunk[:-1]])
        tgt = mx.array([chunk[1:]])
        logits = model(inp).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok_nll = -mx.take_along_axis(logprobs, tgt[..., None], axis=-1).squeeze(-1)[0]

        # Score only tokens with adequate left context (all of them in window 0).
        skip = 0 if start == 0 else (window - stride)
        scored = tok_nll[skip:]
        total_nll += float(mx.sum(scored))
        n_scored += scored.size
        mx.eval(total_nll)

        if end >= n_tokens:
            break
        start += stride

    elapsed = time.perf_counter() - t0
    bpc = (total_nll / math.log(2)) / n_chars
    return {
        "chars": n_chars,
        "tokens": n_tokens,
        "chars_per_token": n_chars / n_tokens,
        "scored_tokens": n_scored,
        "nll_nats_total": total_nll,
        "nll_per_token": total_nll / max(n_scored, 1),
        "token_perplexity": math.exp(total_nll / max(n_scored, 1)),
        "bits_per_char": bpc,
        "eval_seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1536)
    ap.add_argument("--max-chars", type=int, default=120_000,
                    help="cap corpus size to keep each model's eval to a few minutes")
    ap.add_argument("--corpus", default="data/eval/semi_bpc_corpus.txt")
    ap.add_argument("--out", default="artifacts/domain_bpc.json")
    args = ap.parse_args()

    print("building held-out semiconductor corpus ...", flush=True)
    text = build_corpus(Path(args.corpus))[: args.max_chars]
    print(f"  corpus: {len(text):,} chars\n", flush=True)

    results = {}
    for mid in args.models:
        print(f"--- {mid} ---", flush=True)
        try:
            model, tokenizer = load(mid)
            model.eval()
            r = bits_per_char(model, tokenizer, text, args.window, args.stride)
            results[mid] = r
            print(f"  chars/token   {r['chars_per_token']:.3f}", flush=True)
            print(f"  token ppl     {r['token_perplexity']:.3f}", flush=True)
            print(f"  BITS/CHAR     {r['bits_per_char']:.4f}   <-- comparable", flush=True)
            print(f"  ({r['eval_seconds']}s)\n", flush=True)
            del model
            mx.clear_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:120]}\n", flush=True)
            results[mid] = {"failed": True, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    ok = {k: v for k, v in results.items() if not v.get("failed")}
    if ok:
        print("=" * 78)
        print(f"{'model':46s} {'BPC':>8} {'ch/tok':>8} {'vs best':>8}")
        print("-" * 78)
        best = min(v["bits_per_char"] for v in ok.values())
        for k, v in sorted(ok.items(), key=lambda kv: kv[1]["bits_per_char"]):
            print(f"{k:46s} {v['bits_per_char']:>8.4f} {v['chars_per_token']:>8.3f} "
                  f"{100 * (v['bits_per_char'] / best - 1):>7.1f}%")
        print("=" * 78)
        print("lower BPC = better baseline modelling of semiconductor filings")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
