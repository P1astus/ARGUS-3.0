"""Phase 0.4 -- measure real CPT training throughput on this machine.

The Path A plan estimates one epoch over ~100M tokens at 250-600 tok/s, i.e. somewhere
between 46 and 111 hours. That 2.4x spread is too wide to commit a multi-day run against,
and MoE routing overhead in MLX is the dominant unknown. This script replaces the estimate
with a measurement.

It reports two numbers, because they differ and only one of them predicts a multi-day run:

  * steady-state throughput, after warmup
  * sustained throughput, after running long enough to provoke thermal throttling

Usage:
    python scripts/bench_mlx.py --model mlx-community/Qwen3.6-35B-A3B-4bit \
        --seq-len 4096 --batch-size 1 --sustain-minutes 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_lm.tuner.trainer import grad_checkpoint  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

from argus.engine.training.preflight import assert_trainable, collect  # noqa: E402


# Attention-only LoRA. Explicit keys, never auto-discovery -- on a 40-layer, 256-expert
# MoE, letting mlx-lm discover targets attaches adapters to the whole expert stack.
ATTENTION_ONLY_KEYS = ["self_attn.q_proj", "self_attn.k_proj",
                       "self_attn.v_proj", "self_attn.o_proj"]


def build(model_id: str, rank: int, num_layers: int, expert_lora: bool,
          use_grad_checkpoint: bool = True, keys: list[str] | None = None):
    print(f"loading {model_id} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    model.freeze()
    lora_cfg = {"rank": rank, "scale": 20.0, "dropout": 0.0}
    if keys is not None:
        # Explicit override for architectures where ATTENTION_ONLY_KEYS is wrong, not
        # just incomplete -- a hybrid model with linear-attention layers alongside
        # standard attention (e.g. Qwen3.5-family Gated DeltaNet blocks) has trainable
        # projections ATTENTION_ONLY_KEYS was never written to find, and silently
        # adapting only the standard-attention layers would both under-adapt the model
        # and make the throughput/memory numbers measure the wrong configuration.
        lora_cfg["keys"] = keys
    elif not expert_lora:
        lora_cfg["keys"] = ATTENTION_ONLY_KEYS

    n_layers = num_layers if num_layers > 0 else len(model.layers)
    linear_to_lora_layers(model, n_layers, lora_cfg)

    # Without this, backward retains activations for all 40 layers. Measured peak was
    # 84.6GB at seq_len=512 -- past the 55.7GB Metal working set, so the machine swaps
    # and throughput collapses to single-digit tok/s. Not optional on this hardware.
    if use_grad_checkpoint:
        grad_checkpoint(model.layers[0])
        print("  gradient checkpointing: ON", flush=True)

    model.train()

    report = assert_trainable(model, expect_experts=expert_lora, expect_router=False)
    print(report.render(), flush=True)
    return model, tokenizer, report


def make_batch(vocab_size: int, batch_size: int, seq_len: int, rng: np.random.Generator):
    """Synthetic token batch. Content is irrelevant to throughput; shape is not."""
    arr = rng.integers(0, vocab_size, size=(batch_size, seq_len + 1), dtype=np.int32)
    tokens = mx.array(arr)
    return tokens[:, :-1], tokens[:, 1:]


def loss_fn(model, inputs, targets):
    logits = model(inputs).astype(mx.float32)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def run_phase(model, opt, state, vocab, batch_size, seq_len, steps, rng, label,
              time_budget_s: float | None = None, use_compile: bool = True):
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def _step(inputs, targets):
        loss, grads = loss_and_grad(model, inputs, targets)
        opt.update(model, grads)
        return loss

    # mx.compile has to trace the full graph. On a 40-layer, 256-expert MoE that trace
    # is very expensive and can dominate a short benchmark, so it stays switchable --
    # the honest throughput number is the one that matches how the real run is
    # configured, whichever that turns out to be.
    if use_compile:
        print(f"  [{label}] mx.compile enabled; graph traces on first step", flush=True)
        step = mx.compile(_step, inputs=state, outputs=state)
    else:
        step = _step

    tok_per_step = batch_size * seq_len
    times: list[float] = []
    started = time.perf_counter()

    for i in range(steps):
        inputs, targets = make_batch(vocab, batch_size, seq_len, rng)
        step_t0 = time.perf_counter()
        loss = step(inputs, targets)
        mx.eval(state, loss)
        dt = time.perf_counter() - step_t0
        times.append(dt)

        # The first step under mx.compile pays the full graph trace. Report it
        # separately and exclude it from the throughput stats below.
        if i == 0:
            print(f"  [{label}] first step {'(incl. trace) ' if use_compile else ''}"
                  f"{dt:.1f}s", flush=True)

        if (i + 1) % 5 == 0 or i == 0:
            recent = float(np.mean(times[-5:]))
            peak = mx.get_peak_memory() / 1e9
            print(f"  [{label}] step {i + 1:>4}/{steps}  "
                  f"{tok_per_step / recent:>7.1f} tok/s  "
                  f"loss {float(loss):.3f}  peak {peak:.1f}GB", flush=True)

        if time_budget_s is not None and time.perf_counter() - started > time_budget_s:
            print(f"  [{label}] time budget reached after {i + 1} steps", flush=True)
            break

    return times, tok_per_step


def summarize(times: list[float], tok_per_step: int) -> dict:
    # Drop step 0: under mx.compile it carries the one-off graph trace, which would
    # otherwise drag the median down and understate real throughput.
    arr = np.array(times[1:] if len(times) > 1 else times)
    return {
        "steps": len(arr),
        "median_s_per_step": float(np.median(arr)),
        "tok_per_s_median": float(tok_per_step / np.median(arr)),
        "tok_per_s_p10": float(tok_per_step / np.percentile(arr, 90)),
        "tok_per_s_p90": float(tok_per_step / np.percentile(arr, 10)),
    }


def project(tok_per_s: float, corpus_tokens: int) -> dict:
    hours = corpus_tokens / tok_per_s / 3600
    return {"tok_per_s": round(tok_per_s, 1),
            "epoch_hours": round(hours, 1),
            "epoch_days": round(hours / 24, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--num-layers", type=int, default=-1,
                    help="LoRA depth; -1 adapts all layers")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--steady-steps", type=int, default=25)
    ap.add_argument("--sustain-minutes", type=float, default=30.0,
                    help="0 skips the thermal-throttling phase")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing (expect OOM/swap on 64GB)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip mx.compile; isolates MoE graph-trace cost")
    ap.add_argument("--expert-lora", action="store_true",
                    help="adapt MoE experts too (default: attention-only)")
    ap.add_argument("--keys", default=None,
                    help="comma-separated explicit LoRA target keys, overriding "
                         "ATTENTION_ONLY_KEYS -- for architectures (e.g. hybrid "
                         "linear-attention models) where the default target list is "
                         "wrong rather than just incomplete")
    ap.add_argument("--corpus-tokens", type=int, default=100_000_000)
    ap.add_argument("--out", default="artifacts/bench_mlx.json")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    rng = np.random.default_rng(0)
    model, tokenizer, report = build(args.model, args.rank, args.num_layers,
                                     args.expert_lora,
                                     use_grad_checkpoint=not args.no_grad_checkpoint,
                                     keys=keys)
    vocab = tokenizer.vocab_size

    opt = optim.AdamW(learning_rate=1e-5)
    state = [model.state, opt.state, mx.random.state]

    print(f"\nconfig: seq_len={args.seq_len} batch={args.batch_size} "
          f"rank={args.rank} expert_lora={args.expert_lora}", flush=True)

    print("\n--- warmup ---", flush=True)
    run_phase(model, opt, state, vocab, args.batch_size, args.seq_len,
              args.warmup_steps, rng, "warmup", use_compile=not args.no_compile)

    print("\n--- steady state ---", flush=True)
    steady_times, tok_per_step = run_phase(
        model, opt, state, vocab, args.batch_size, args.seq_len,
        args.steady_steps, rng, "steady", use_compile=not args.no_compile)
    steady = summarize(steady_times, tok_per_step)

    sustained = None
    if args.sustain_minutes > 0:
        print(f"\n--- sustained ({args.sustain_minutes:.0f} min, thermal) ---", flush=True)
        sus_times, _ = run_phase(
            model, opt, state, vocab, args.batch_size, args.seq_len,
            steps=10_000, rng=rng, label="sustain",
            time_budget_s=args.sustain_minutes * 60,
            use_compile=not args.no_compile)
        # Use only the last third, once the machine has heated up.
        tail = sus_times[max(len(sus_times) // 3 * 2, 1):]
        sustained = summarize(tail, tok_per_step)

    predictive = (sustained or steady)["tok_per_s_median"]
    result = {
        "model": args.model,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "expert_lora": args.expert_lora,
        "lora_keys": keys if keys is not None else (
            None if args.expert_lora else ATTENTION_ONLY_KEYS),
        "compiled": not args.no_compile,
        "grad_checkpoint": not args.no_grad_checkpoint,
        "trainable_params": report.trainable_params,
        "trainable_pct": round(report.trainable_pct, 5),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "steady": steady,
        "sustained": sustained,
        "projection": project(predictive, args.corpus_tokens),
        "corpus_tokens": args.corpus_tokens,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 62)
    print(f"steady-state : {steady['tok_per_s_median']:.1f} tok/s")
    if sustained:
        drop = 100 * (1 - sustained["tok_per_s_median"] / steady["tok_per_s_median"])
        print(f"sustained    : {sustained['tok_per_s_median']:.1f} tok/s  "
              f"({drop:+.1f}% vs steady)")
    print(f"peak memory  : {result['peak_memory_gb']:.1f} GB of 55.7 GB usable")
    p = result["projection"]
    print(f"\nprojected {args.corpus_tokens/1e6:.0f}M-token epoch: "
          f"{p['epoch_hours']} h ({p['epoch_days']} days) at {p['tok_per_s']} tok/s")

    print("\nPhase 0 exit criterion:")
    if predictive > 500:
        print("  > 500 tok/s  -> PROCEED with full Path A scope")
    elif predictive >= 300:
        print("  300-500 tok/s -> PROCEED, plan checkpoint-wise early stopping seriously")
    else:
        print("  < 300 tok/s  -> STOP AND REVISIT: cut corpus to ~50M (EDGAR-only),")
        print("                  or reconsider the 8B distill")
    print("=" * 62)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
