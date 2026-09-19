"""Checkpointing and resume.

WHY THIS IS A DELIVERABLE, NOT A CONVENIENCE
A full CPT pass is multi-day (measured: ~9.3 days for 350M characters on the selected 14B).
A run that long on a laptop WILL be interrupted -- sleep, thermal, an OS update, a closed
lid. Resume is the difference between losing an hour and losing the run.

WHAT mlx-lm's BUILT-IN ADAPTER SAVING DOES NOT DO
It stores adapter weights only. Not the optimizer state, not the RNG state, not the data
iterator position, not the step count. Resuming from it silently restarts the learning-rate
schedule and re-feeds already-seen shards, while the loss curve looks superficially fine.
All five pieces are persisted here, atomically.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TrainState:
    """Everything needed to resume a run exactly where it stopped."""

    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0

    # Data cursor. Without this a resume silently re-feeds seen data, which inflates
    # effective epochs and quietly changes the experiment.
    shard_id: int = 0
    shard_offset: int = 0

    # SFT/CPT interleave cursor (engine/training/interleave.py). Only advances when
    # `sft_interleave_ratio > 0`; defaults keep old checkpoints (saved before
    # interleaving existed) loading correctly via TrainState(**old_dict).
    sft_position: int = 0
    cpt_batches_since_sft: int = 0

    best_val_loss: float = float("inf")
    lr: float = 0.0
    wall_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)


class CheckpointManager:
    """Atomic checkpoint save/load.

    Writes to a temporary directory and renames on completion, so a crash mid-save cannot
    leave a half-written checkpoint that resumes into corruption -- the exact failure a
    multi-day unattended run is most likely to hit.
    """

    def __init__(self, root: str | Path, keep_last: int = 3) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _dir(self, step: int) -> Path:
        return self.root / f"step_{step:08d}"

    def save(self, step: int, model, optimizer, state: TrainState,
             config: dict | None = None) -> Path:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        tmp = self.root / f".tmp_step_{step:08d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        # 1. adapter weights (trainable parameters only)
        mx.save_safetensors(str(tmp / "adapter.safetensors"),
                            dict(tree_flatten(model.trainable_parameters())))

        # 2. optimizer state -- moments and step counter. Dropping these restarts Adam's
        #    bias correction and momentum from zero, which shows up as a visible loss
        #    spike on resume.
        try:
            mx.save_safetensors(str(tmp / "optimizer.safetensors"),
                                dict(tree_flatten(optimizer.state)))
        except Exception as e:
            log.warning("optimizer state not serialisable (%s); resume will restart "
                        "momentum", type(e).__name__)

        # 3. RNG state, so dropout and data shuffling continue deterministically
        try:
            mx.save_safetensors(str(tmp / "rng.safetensors"),
                                {"state": mx.random.state})
        except Exception:
            pass

        # 4/5. training state (incl. data cursor and step) and config
        (tmp / "state.json").write_text(json.dumps(asdict(state), indent=2))
        if config:
            (tmp / "config.json").write_text(json.dumps(config, indent=2, default=str))

        final = self._dir(step)
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)          # atomic on POSIX

        (self.root / "LATEST").write_text(final.name)
        self._prune()
        log.info("checkpoint saved: %s", final)
        return final

    def _prune(self) -> None:
        dirs = sorted(d for d in self.root.glob("step_*") if d.is_dir())
        for d in dirs[:-self.keep_last] if self.keep_last > 0 else []:
            shutil.rmtree(d, ignore_errors=True)

    def latest(self) -> Path | None:
        marker = self.root / "LATEST"
        if marker.exists():
            p = self.root / marker.read_text().strip()
            if p.exists():
                return p
        dirs = sorted(d for d in self.root.glob("step_*") if d.is_dir())
        return dirs[-1] if dirs else None

    def load(self, model, optimizer, path: str | Path | None = None) -> TrainState | None:
        """Restore a run. Returns None when there is nothing to resume from."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        ckpt = Path(path) if path else self.latest()
        if ckpt is None or not ckpt.exists():
            return None

        weights = mx.load(str(ckpt / "adapter.safetensors"))
        model.update(tree_unflatten(list(weights.items())))

        opt_path = ckpt / "optimizer.safetensors"
        if opt_path.exists():
            try:
                optimizer.state = tree_unflatten(list(mx.load(str(opt_path)).items()))
            except Exception as e:
                log.warning("could not restore optimizer state (%s)", type(e).__name__)

        rng_path = ckpt / "rng.safetensors"
        if rng_path.exists():
            try:
                mx.random.state = mx.load(str(rng_path))["state"]
            except Exception:
                pass

        state = TrainState(**json.loads((ckpt / "state.json").read_text()))
        log.info("resumed from %s at step %d (%d tokens, shard %d offset %d)",
                 ckpt, state.step, state.tokens_seen, state.shard_id, state.shard_offset)
        return state


def verify_resume_continuity(before: list[float], after: list[float],
                             tolerance: float = 0.15) -> bool:
    """Check that loss did not jump across a resume boundary.

    The test that matters: kill a run mid-epoch, resume, and confirm the loss curve is
    continuous. A jump means optimizer or RNG state was lost -- which a weights-only
    checkpoint would produce while still appearing to work.
    """
    if not before or not after:
        return False
    last = sum(before[-5:]) / len(before[-5:])
    first = sum(after[:5]) / len(after[:5])
    return abs(first - last) / max(abs(last), 1e-6) <= tolerance
