"""Continued pretraining loop (MLX, LoRA).

Configured from Phase 0 measurements rather than defaults:

  * gradient checkpointing is ON unconditionally -- without it peak memory was 84.6GB
    against a 55.7GB working set, and throughput collapsed ~14x from swapping
  * LoRA targets are passed EXPLICITLY; mlx-lm's auto-discovery attaches adapters to every
    Linear/SwitchLinear it finds, which on an MoE means the whole expert stack
  * preflight assertions run before any long job starts
  * checkpoint + eval every N tokens, so the run can be stopped early when the loss curve
    flattens rather than paying for tokens past saturation

The base model is a config field, never an import.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CPTConfig:
    # Selected in Phase 0: best domain BPC (0.3394), only true base checkpoint at 14B,
    # fastest 14B measured, Apache-2.0, 262k native context.
    model: str = "mlx-community/Ministral-3-14B-Base-2512-4bit"

    shard_dir: str = "data/corpus/shards"
    out_dir: str = "artifacts/cpt"

    seq_len: int = 4096
    batch_size: int = 1
    grad_accum: int = 8            # effective batch without the memory cost

    # Rank 256, not the usual 16. "LoRA Learns Less and Forgets Less" (arXiv 2405.09673)
    # shows standard low ranks substantially underperform full finetuning for CONTINUED
    # PRETRAINING specifically, and that r~256 is needed to approach it. Measured on this
    # model, the cost is negligible: 136.0 tok/s at r=256 vs 135.9 at r=16 (throughput is
    # dominated by the base model's forward/backward), for +3.5GB peak memory.
    # Running CPT at r=16 would have been an underpowered version of the experiment.
    lora_rank: int = 256
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    lora_keys: tuple[str, ...] = ("self_attn.q_proj", "self_attn.k_proj",
                                  "self_attn.v_proj", "self_attn.o_proj")
    lora_layers: int = -1          # -1 = all

    # Low LR because CPT on a base checkpoint should adapt, not overwrite. Even on a base
    # model, a high LR over ~100M narrow-domain tokens degrades general capability.
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    lr_schedule: str = "cosine"
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    max_tokens: int | None = None  # None = one pass over the corpus
    epochs: int = 1

    # Interleaving mitigates the Llama-Fin finding (handoff §4e): sequential CPT-then-SFT
    # collapsed instruction-following from 7.8 to ~1.0 on MT-Bench; joint training did not.
    # Default OFF -- this must not change the behaviour of a plain CPT run by accident.
    # See engine/training/interleave.py for the mixing logic and what is/isn't validated.
    sft_examples_path: str | None = None
    sft_interleave_ratio: float = 0.0

    # Every ~10M tokens: reproduces the reference study's checkpoint methodology on our
    # own corpus, and makes early stopping possible.
    eval_every_tokens: int = 10_000_000
    checkpoint_every_tokens: int = 5_000_000
    eval_batches: int = 50

    expect_experts: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


class CPTTrainer:
    """LoRA continued pretraining with resumable state."""

    def __init__(self, config: CPTConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self._loss_history: list[float] = []

    def build(self):
        """Load the model, attach adapters, and run preflight assertions."""
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load

        from argus.engine.training.preflight import assert_trainable

        c = self.config
        log.info("loading %s", c.model)
        self.model, self.tokenizer = load(c.model)
        mx.eval(self.model.parameters())

        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            c.lora_layers if c.lora_layers > 0 else len(self.model.layers),
            {"rank": c.lora_rank, "scale": c.lora_scale,
             "dropout": c.lora_dropout, "keys": list(c.lora_keys)},
        )

        # Measured: without this, 84.6GB peak and ~14x slower from swapping.
        grad_checkpoint(self.model.layers[0])
        self.model.train()

        report = assert_trainable(self.model, expect_experts=c.expect_experts,
                                 expect_router=False)
        log.info("%s", report.render())

        self.optimizer = optim.AdamW(learning_rate=c.learning_rate,
                                     weight_decay=c.weight_decay)
        return self

    def _loss(self, model, inputs, targets):
        import mlx.core as mx
        import mlx.nn as nn
        logits = model(inputs).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def _sft_loss(self, model, inputs, mask):
        """Masked loss for an interleaved SFT batch -- identical rationale to
        sft.py's masked_loss: training on the prompt would teach market-state
        description rather than the target task."""
        import mlx.core as mx
        import mlx.nn as nn
        x, y = inputs[:-1], inputs[1:]
        m = mask[1:]   # shift with the targets; a prompt/completion boundary at
                       # position i means position i's prediction is the first one
                       # that should carry loss
        logits = model(x[None, :]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, y[None, :], reduction="none")
        return (losses[0] * m).sum() / mx.maximum(m.sum(), 1)

    def evaluate(self, val_shards: list[dict], n_batches: int = 50) -> float:
        """Mean validation loss on held-out shards.

        This is Gate 3a. It is cheap, runs per checkpoint, and can fail a run out early --
        but passing it does NOT imply the task-level gate (3e). The reference study
        measured only validation loss and explicitly left the translation to task
        performance as future work, so the two must not be conflated.
        """
        import mlx.core as mx
        import numpy as np

        self.model.eval()
        total, count = 0.0, 0
        c = self.config

        for meta in val_shards:
            arr = np.load(meta["path"], mmap_mode="r")
            need = c.seq_len * c.batch_size + 1
            offset = 0
            while offset + need <= len(arr) and count < n_batches:
                flat = np.asarray(arr[offset:offset + need], dtype=np.int64)
                x = mx.array(flat[:-1].reshape(c.batch_size, c.seq_len))
                y = mx.array(flat[1:].reshape(c.batch_size, c.seq_len))
                loss = self._loss(self.model, x, y)
                mx.eval(loss)
                total += float(loss)
                count += 1
                offset += need
            if count >= n_batches:
                break

        self.model.train()
        return total / max(count, 1)

    def train(self, resume: bool = True) -> dict:
        import mlx.core as mx
        import mlx.nn as nn

        from argus.engine.training.checkpoint import CheckpointManager, TrainState
        from argus.engine.training.data_iter import ShardIterator

        c = self.config
        if self.model is None:
            self.build()

        data = ShardIterator(c.shard_dir, seq_len=c.seq_len,
                             batch_size=c.batch_size, seed=c.seed)
        train_shards, val_shards = data.validation_split()
        log.info("corpus: %d tokens, %d sequences, %d val shards",
                 data.total_tokens, data.total_sequences, len(val_shards))

        ckpt = CheckpointManager(Path(c.out_dir) / "checkpoints")
        state = (ckpt.load(self.model, self.optimizer) if resume else None) or TrainState()

        total_tokens = c.max_tokens or data.total_tokens * c.epochs
        total_steps = max(total_tokens // (c.seq_len * c.batch_size * c.grad_accum), 1)

        # sft_interleave_ratio defaults to 0.0, at which `interleave()` yields exactly
        # `data.iterate(...)`'s stream (verified: tests/training/test_interleave.py::
        # TestZeroRatioIsInert) -- a plain CPT run's behaviour is unchanged by this branch
        # existing.
        from argus.engine.training.interleave import BatchKind, InterleaveState, interleave
        sft_examples = []
        sft_config = None
        if c.sft_interleave_ratio > 0 and c.sft_examples_path:
            from argus.engine.training.sft import SFTConfig, load_examples
            sft_examples = load_examples(c.sft_examples_path)
            sft_config = SFTConfig(seq_len=c.seq_len)
            log.info("interleaving %d SFT examples at ratio %.3f (Llama-Fin mitigation, "
                     "handoff §4e)", len(sft_examples), c.sft_interleave_ratio)
        interleave_state = InterleaveState(sft_position=state.sft_position,
                                           cpt_batches_since_sft=state.cpt_batches_since_sft)

        loss_and_grad = nn.value_and_grad(self.model, self._loss)
        sft_loss_and_grad = nn.value_and_grad(self.model, self._sft_loss)
        started = time.perf_counter()
        accum, accum_n = None, 0
        last_eval = state.tokens_seen
        last_ckpt = state.tokens_seen

        for item in interleave(data, sft_examples, self.tokenizer, sft_config,
                               c.sft_interleave_ratio, state=interleave_state,
                               start_shard=state.shard_id,
                               start_offset=state.shard_offset, epochs=c.epochs):
            if item.kind == BatchKind.CPT:
                batch = item.cpt
                x, y = mx.array(batch.inputs), mx.array(batch.targets)
                loss, grads = loss_and_grad(self.model, x, y)
                state.tokens_seen += batch.tokens
                state.shard_id, state.shard_offset = batch.shard_id, batch.offset
            else:
                inputs = mx.array(item.sft_inputs)
                mask = mx.array(item.sft_mask)
                loss, grads = sft_loss_and_grad(self.model, inputs, mask)
                state.tokens_seen += int(item.sft_inputs.shape[0])
                state.sft_position = interleave_state.sft_position
                state.cpt_batches_since_sft = interleave_state.cpt_batches_since_sft

            # Gradient accumulation: a larger effective batch without the memory a larger
            # literal batch would need. An SFT step's gradient is accumulated the same
            # way a CPT step's is -- both update the same adapter, which is the entire
            # point of interleaving rather than running two separate training phases.
            accum = grads if accum is None else _tree_add(accum, grads)
            accum_n += 1

            if accum_n >= c.grad_accum:
                accum = _tree_scale(accum, 1.0 / accum_n)
                if c.grad_clip:
                    accum, _ = optim_clip(accum, c.grad_clip)
                self.optimizer.learning_rate = cosine_lr(
                    state.step, c.learning_rate, c.warmup_steps, total_steps)
                self.optimizer.update(self.model, accum)
                mx.eval(self.model.parameters(), self.optimizer.state)
                accum, accum_n = None, 0
                state.step += 1

            self._loss_history.append(float(loss))

            if state.step % 10 == 0 and accum_n == 0:
                elapsed = time.perf_counter() - started + state.wall_seconds
                log.info("step %d | %d tok | loss %.4f | %.1f tok/s | peak %.1fGB",
                         state.step, state.tokens_seen, float(loss),
                         state.tokens_seen / max(elapsed, 1e-6),
                         mx.get_peak_memory() / 1e9)

            if state.tokens_seen - last_eval >= c.eval_every_tokens:
                val = self.evaluate(val_shards, c.eval_batches)
                state.history.append({"step": state.step, "tokens": state.tokens_seen,
                                      "val_loss": val})
                state.best_val_loss = min(state.best_val_loss, val)
                log.info("EVAL @ %d tokens: val_loss %.4f (best %.4f)",
                         state.tokens_seen, val, state.best_val_loss)
                last_eval = state.tokens_seen

            if state.tokens_seen - last_ckpt >= c.checkpoint_every_tokens:
                state.wall_seconds = time.perf_counter() - started + state.wall_seconds
                started = time.perf_counter()
                ckpt.save(state.step, self.model, self.optimizer, state, c.to_dict())
                last_ckpt = state.tokens_seen

            if c.max_tokens and state.tokens_seen >= c.max_tokens:
                break

        final_val = self.evaluate(val_shards, c.eval_batches)
        state.history.append({"step": state.step, "tokens": state.tokens_seen,
                              "val_loss": final_val, "final": True})
        state.wall_seconds += time.perf_counter() - started
        ckpt.save(state.step, self.model, self.optimizer, state, c.to_dict())

        summary = {"steps": state.step, "tokens": state.tokens_seen,
                   "final_val_loss": final_val, "best_val_loss": state.best_val_loss,
                   "history": state.history, "wall_seconds": state.wall_seconds}
        Path(c.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(c.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, s):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * s, a)


def optim_clip(grads, max_norm: float):
    import mlx.optimizers as optim
    return optim.clip_grad_norm(grads, max_norm)
