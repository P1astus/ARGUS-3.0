"""Supervised fine-tuning: output-format shaping on top of the CPT'd base.

SCOPE IS DELIBERATELY NARROW
CPT teaches the domain; SFT teaches the shape. At ~100M domain tokens the realistic
outcome is native sub-segment vocabulary and framing, not new factual recall -- so SFT
should not try to teach analysis. It teaches the model to emit a bull/base/bear writeup
with entry, target and invalidation levels, validated against contracts.Recommendation.

WHY SFT COMES AFTER CPT, NOT BEFORE
Continued pretraining on plain prose degrades instruction-following and output format.
Running SFT afterwards restores it. Reversing the order would have CPT undo the SFT.

LOSS IS MASKED TO COMPLETIONS
Training on the prompt tokens as well would teach the model to generate market-state
descriptions -- fluent, and exactly not the task.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from argus.contracts.recommendation import Recommendation

log = logging.getLogger(__name__)


@dataclass
class SFTExample:
    prompt: str
    completion: str

    @classmethod
    def from_recommendation(cls, prompt: str, rec: Recommendation) -> "SFTExample":
        # Serialising through the schema guarantees every training target is itself valid.
        # Training on a malformed example teaches malformed output.
        return cls(prompt=prompt, completion=rec.model_dump_json(indent=2))


@dataclass
class SFTConfig:
    base_adapter: str | None = None      # adapter produced by CPT; None = train from base
    model: str = "mlx-community/Ministral-3-14B-Base-2512-4bit"
    data_path: str = "data/sft/examples.jsonl"
    out_dir: str = "artifacts/sft"

    seq_len: int = 4096
    batch_size: int = 1
    grad_accum: int = 4

    # Small and gentle: SFT sets are a few hundred to a few thousand examples, and a high
    # LR here would overwrite what CPT installed.
    learning_rate: float = 1e-5
    epochs: int = 3
    warmup_steps: int = 20
    grad_clip: float = 1.0

    lora_rank: int = 16
    lora_keys: tuple[str, ...] = ("self_attn.q_proj", "self_attn.k_proj",
                                  "self_attn.v_proj", "self_attn.o_proj")

    mask_prompt_loss: bool = True
    seed: int = 0

    # Checkpoint/eval cadence is in STEPS, not tokens like cpt.py -- an SFT set is a few
    # hundred to a few thousand examples, so a whole run can be under a thousand steps and
    # token-based cadence (tuned for a ~100M-token CPT run) would rarely fire at all.
    eval_every_steps: int = 20
    checkpoint_every_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def build_dataset(examples: list[SFTExample], tokenizer, spec: SFTConfig
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tokenise into (input_ids, loss_mask) pairs.

    The mask is what confines learning to the completion.
    """
    out = []
    for ex in examples:
        p_ids = tokenizer.encode(ex.prompt)
        c_ids = tokenizer.encode(ex.completion, add_special_tokens=False)
        ids = (p_ids + c_ids)[:spec.seq_len]

        mask = np.zeros(len(ids), dtype=np.float32)
        if spec.mask_prompt_loss:
            mask[len(p_ids):] = 1.0
        else:
            mask[:] = 1.0

        if mask.sum() == 0:      # completion truncated away entirely
            continue
        out.append((np.array(ids, dtype=np.int64), mask))
    return out


def load_examples(path: str | Path) -> list[SFTExample]:
    """Read JSONL, validating every completion against the schema.

    A malformed target is dropped loudly rather than trained on.
    """
    examples: list[SFTExample] = []
    dropped = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        try:
            Recommendation.model_validate(json.loads(obj["completion"]))
        except Exception:
            dropped += 1
            continue
        examples.append(SFTExample(prompt=obj["prompt"], completion=obj["completion"]))

    if dropped:
        log.warning("dropped %d SFT examples whose completion failed schema validation",
                    dropped)
    return examples


class SFTTrainer:
    """Format-shaping pass on the CPT'd model."""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self._loss_history: list[float] = []

    def build(self):
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load

        from argus.engine.training.preflight import assert_trainable

        c = self.config
        self.model, self.tokenizer = load(c.model)
        mx.eval(self.model.parameters())
        self.model.freeze()

        linear_to_lora_layers(self.model, len(self.model.layers),
                              {"rank": c.lora_rank, "scale": 20.0, "dropout": 0.0,
                               "keys": list(c.lora_keys)})

        # Continue from the CPT adapter so SFT refines the domain-adapted model rather
        # than a fresh one -- otherwise the CPT run is discarded.
        if c.base_adapter:
            from mlx.utils import tree_unflatten
            weights = mx.load(str(Path(c.base_adapter) / "adapter.safetensors"))
            self.model.update(tree_unflatten(list(weights.items())))
            log.info("loaded CPT adapter from %s", c.base_adapter)

        grad_checkpoint(self.model.layers[0])
        self.model.train()
        assert_trainable(self.model, expect_experts=False)
        self.optimizer = optim.AdamW(learning_rate=c.learning_rate)
        return self

    def masked_loss(self, model, ids, mask):
        """Next-token loss over one (prompt+completion) sequence, masked to the
        completion. Shifts internally -- `ids`/`mask` are the single aligned arrays
        `build_dataset` produces, the same shape `cpt.py`'s interleaved `_sft_loss`
        consumes, so a standalone SFT run and an interleaved CPT+SFT step train on
        identically-shaped batches.
        """
        import mlx.core as mx
        import mlx.nn as nn
        x, y = ids[:-1], ids[1:]
        m = mask[1:]   # shift with the targets -- a prompt/completion boundary at
                       # position i means position i's prediction is the first one
                       # that should carry loss
        logits = model(x[None, :]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, y[None, :], reduction="none")
        return (losses[0] * m).sum() / mx.maximum(m.sum(), 1)

    def evaluate(self, val_set: list[tuple]) -> float:
        """Mean masked loss over a held-out slice of the SFT set.

        Analogous to cpt.py's evaluate() but over in-memory examples rather than shards
        -- an SFT set is small enough (a few hundred to a few thousand examples) that
        there is no reason to stream it from disk.
        """
        import mlx.core as mx

        if not val_set:
            return float("nan")

        self.model.eval()
        total = 0.0
        for ids, mask in val_set:
            loss = self.masked_loss(self.model, mx.array(ids), mx.array(mask))
            mx.eval(loss)
            total += float(loss)
        self.model.train()
        return total / len(val_set)

    def train(self, resume: bool = True) -> dict:
        """Format-shaping pass with the same resumable-checkpoint discipline as
        CPTTrainer.train() -- reuses CheckpointManager/TrainState directly rather than a
        parallel mechanism, so a killed SFT run resumes the same way a killed CPT run
        does. `TrainState.shard_offset` is repurposed here as "example index within the
        current epoch" (SFT has no shard cursor of its own); `TrainState.epoch` tracks
        which pass over the dataset is in progress.
        """
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        from argus.engine.training.checkpoint import CheckpointManager, TrainState
        from argus.engine.training.cpt import _tree_add, _tree_scale, cosine_lr, optim_clip

        c = self.config
        if self.model is None:
            self.build()

        examples = load_examples(c.data_path)
        if not examples:
            raise ValueError(f"no valid SFT examples found in {c.data_path}")
        dataset = build_dataset(examples, self.tokenizer, c)
        if not dataset:
            raise ValueError(f"every example in {c.data_path} was truncated to nothing "
                             f"by seq_len={c.seq_len}")

        # Deterministic train/val split. This is not meant to be a rigorous held-out
        # benchmark -- on a set this small it exists to catch the run overfitting, the
        # same role eval plays in cpt.py but scaled to SFT's much smaller data volume.
        rng = np.random.RandomState(c.seed)
        order = rng.permutation(len(dataset))
        n_val = max(1, len(dataset) // 10) if len(dataset) >= 10 else 0
        val_set = [dataset[i] for i in order[:n_val]]
        train_set = [dataset[i] for i in order[n_val:]]
        log.info("SFT dataset: %d examples (%d train, %d val)",
                 len(dataset), len(train_set), len(val_set))

        ckpt = CheckpointManager(Path(c.out_dir) / "checkpoints")
        state = (ckpt.load(self.model, self.optimizer) if resume else None) or TrainState()

        steps_per_epoch = max(len(train_set) // (c.batch_size * c.grad_accum), 1)
        total_steps = steps_per_epoch * c.epochs

        loss_and_grad = nn.value_and_grad(self.model, self.masked_loss)
        started = time.perf_counter()
        accum, accum_n = None, 0
        last_eval, last_ckpt = state.step, state.step

        for epoch in range(state.epoch, c.epochs):
            for i in range(state.shard_offset, len(train_set)):
                ids, mask = train_set[i]
                loss, grads = loss_and_grad(self.model, mx.array(ids), mx.array(mask))
                state.tokens_seen += len(ids)
                state.shard_offset = i + 1

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
                    log.info("step %d/%d | epoch %d | loss %.4f | %.1f ex/s | peak %.1fGB",
                             state.step, total_steps, epoch, float(loss),
                             state.step * c.grad_accum / max(elapsed, 1e-6),
                             mx.get_peak_memory() / 1e9)

                if val_set and accum_n == 0 and state.step - last_eval >= c.eval_every_steps:
                    val = self.evaluate(val_set)
                    state.history.append({"step": state.step, "epoch": epoch,
                                          "val_loss": val})
                    state.best_val_loss = min(state.best_val_loss, val)
                    log.info("EVAL @ step %d: val_loss %.4f (best %.4f)",
                             state.step, val, state.best_val_loss)
                    last_eval = state.step

                if accum_n == 0 and state.step - last_ckpt >= c.checkpoint_every_steps:
                    state.wall_seconds = time.perf_counter() - started + state.wall_seconds
                    started = time.perf_counter()
                    ckpt.save(state.step, self.model, self.optimizer, state, c.to_dict())
                    last_ckpt = state.step

            state.epoch = epoch + 1
            state.shard_offset = 0

        final_val = self.evaluate(val_set)
        state.history.append({"step": state.step, "epoch": c.epochs,
                              "val_loss": final_val, "final": True})
        state.best_val_loss = min(state.best_val_loss, final_val) if val_set else state.best_val_loss
        state.wall_seconds += time.perf_counter() - started
        ckpt.save(state.step, self.model, self.optimizer, state, c.to_dict())

        summary = {"steps": state.step, "tokens": state.tokens_seen,
                   "examples": len(dataset), "final_val_loss": final_val,
                   "best_val_loss": state.best_val_loss, "history": state.history,
                   "wall_seconds": state.wall_seconds}
        Path(c.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(c.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
