"""Resumable, deterministic shard iterator.

The cursor is (shard_id, offset), which is why pack.py writes fixed-size .npy shards:
resuming means memory-mapping one file and seeking, not replaying the corpus.

Determinism matters for the same reason resume does. A multi-day run that is interrupted
must continue the *same* data order, or the effective epoch structure changes silently and
the run stops being the experiment that was configured.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Batch:
    inputs: np.ndarray       # (batch, seq_len)
    targets: np.ndarray      # (batch, seq_len) -- inputs shifted by one
    shard_id: int
    offset: int
    tokens: int


class ShardIterator:
    """Iterate packed token shards with a resumable cursor.

    Args:
        shard_dir: directory containing index.json and shard_*.npy
        seq_len: sequence length; must divide the shard token counts
        batch_size: sequences per batch
        shuffle_shards: permute shard ORDER (deterministically) but never within a shard.
            Within-shard order is preserved so that a cursor offset is meaningful --
            shuffling inside a shard would make (shard_id, offset) ambiguous on resume.
    """

    def __init__(self, shard_dir: str | Path, seq_len: int = 4096, batch_size: int = 1,
                 shuffle_shards: bool = True, seed: int = 0) -> None:
        self.dir = Path(shard_dir)
        index_path = self.dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"no index.json in {self.dir}; run the packer first")

        self.index = json.loads(index_path.read_text())
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed

        if self.index.get("seq_len") and self.index["seq_len"] != seq_len:
            raise ValueError(
                f"shards were packed at seq_len={self.index['seq_len']} but the iterator "
                f"was configured for {seq_len}; repack or reconfigure")

        self.shards = list(self.index["shards"])
        if shuffle_shards:
            order = np.random.default_rng(seed).permutation(len(self.shards))
            self.shards = [self.shards[i] for i in order]

        self._pos = {s["shard_id"]: i for i, s in enumerate(self.shards)}

    @property
    def total_tokens(self) -> int:
        return int(self.index["total_tokens"])

    @property
    def total_sequences(self) -> int:
        return int(self.index["total_sequences"])

    def steps_per_epoch(self) -> int:
        return self.total_sequences // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        return self.iterate()

    def iterate(self, start_shard: int = 0, start_offset: int = 0,
                epochs: int = 1) -> Iterator[Batch]:
        """Yield batches, optionally resuming mid-shard.

        `start_shard` is a shard_id, not a position in the shuffled order, so a checkpoint
        remains valid even if the shuffle seed changes -- it resolves to the right file
        either way.
        """
        # +1 so a sequence and its shifted target both fit.
        need = self.seq_len * self.batch_size + 1

        for epoch in range(epochs):
            begin = self._pos.get(start_shard, 0) if epoch == 0 else 0
            offset = start_offset if epoch == 0 else 0

            for si in range(begin, len(self.shards)):
                meta = self.shards[si]
                arr = np.load(meta["path"], mmap_mode="r")
                n = len(arr)

                while offset + need <= n:
                    flat = np.asarray(arr[offset:offset + need], dtype=np.int64)
                    x = flat[:-1].reshape(self.batch_size, self.seq_len)
                    y = flat[1:].reshape(self.batch_size, self.seq_len)
                    yield Batch(inputs=x, targets=y, shard_id=meta["shard_id"],
                                offset=offset, tokens=self.seq_len * self.batch_size)
                    offset += self.seq_len * self.batch_size

                offset = 0

    def validation_split(self, fraction: float = 0.01) -> tuple[list[dict], list[dict]]:
        """Hold out whole shards for validation.

        Whole shards rather than random sequences: sequences from the same document share
        vocabulary and phrasing, so a random split would leak documents across the
        boundary and understate validation loss.
        """
        n_val = max(1, int(len(self.shards) * fraction))
        return self.shards[n_val:], self.shards[:n_val]
