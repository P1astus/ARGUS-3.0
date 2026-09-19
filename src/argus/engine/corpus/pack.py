"""Chunking and shard packing.

Two decisions, both measured rather than assumed.

CHUNK WINDOW IS A CONFIG PARAMETER, NOT A CONSTANT
The Phase 0.5 sweep found seq 8192 costs only ~18% more wall-clock than 4096 on a 14B and
fits in ~39GB. That is affordable, and it changes what chunking should do: at 8k a full
Risk Factors or MD&A section usually fits intact, whereas at 4k it gets cut mid-argument.
For a corpus whose value is long-form reasoning about disclosures, keeping sections whole
is likely worth the 18%. The chunker is therefore section-aware with the window
configurable, and the choice is deferred to the pilot run's loss curves.

STREAMING MEMMAP SHARDS
100M tokens as uint32 is ~400MB, which does fit in memory -- but the tokeniser's Python
objects during packing do not, comfortably. Writing fixed-size .npy shards and reading them
back via memmap also makes the resumable data cursor trivial: (shard_id, offset).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ChunkSpec:
    window: int = 4096            # revisit after the pilot; 8192 is affordable
    overlap: int = 128            # carry context across a forced split
    min_chunk: int = 256          # below this a chunk is mostly padding


def chunk_text(text: str, tokenizer, spec: ChunkSpec = ChunkSpec()) -> list[list[int]]:
    """Split on paragraph boundaries where possible, hard-split only when forced.

    Paragraph-aware packing keeps arguments intact; a naive fixed stride would cut
    sentences in half and teach the model to continue fragments.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[list[int]] = []
    current: list[int] = []

    for para in paragraphs:
        ids = tokenizer.encode(para, add_special_tokens=False) \
            if hasattr(tokenizer, "encode") else tokenizer(para)

        # A single paragraph longer than the window must be split, but only then.
        if len(ids) > spec.window:
            if current:
                chunks.append(current)
                current = []
            for i in range(0, len(ids), spec.window - spec.overlap):
                piece = ids[i:i + spec.window]
                if len(piece) >= spec.min_chunk:
                    chunks.append(piece)
            continue

        if len(current) + len(ids) > spec.window:
            if len(current) >= spec.min_chunk:
                chunks.append(current)
            current = current[-spec.overlap:] if spec.overlap else []

        current.extend(ids)

    if len(current) >= spec.min_chunk:
        chunks.append(current)
    return chunks


class ShardWriter:
    """Fixed-size token shards as .npy, plus a JSON index.

    Args:
        shard_tokens: tokens per shard. 8M x uint32 = 32MB per file -- small enough to
            memmap cheaply, large enough that a 100M-token corpus is ~13 files rather than
            thousands.
    """

    def __init__(self, out_dir: str | Path, shard_tokens: int = 8_000_000,
                 seq_len: int = 4096) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.seq_len = seq_len
        self._buf: list[int] = []
        self._shard_id = 0
        self._index: list[dict] = []
        self._total = 0

    def add(self, token_ids: list[int]) -> None:
        self._buf.extend(token_ids)
        while len(self._buf) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n: int) -> None:
        if n <= 0:
            return
        # Truncate to a whole number of sequences so the training iterator never has to
        # handle a ragged tail mid-shard.
        n = (n // self.seq_len) * self.seq_len
        if n == 0:
            return
        arr = np.array(self._buf[:n], dtype=np.uint32)
        path = self.out_dir / f"shard_{self._shard_id:05d}.npy"
        np.save(path, arr)
        self._index.append({
            "shard_id": self._shard_id,
            "path": str(path),
            "tokens": int(n),
            "sequences": int(n // self.seq_len),
        })
        self._total += n
        self._buf = self._buf[n:]
        self._shard_id += 1

    def close(self) -> dict:
        self._flush(len(self._buf))
        index = {
            "seq_len": self.seq_len,
            "shard_tokens": self.shard_tokens,
            "total_tokens": self._total,
            "total_sequences": sum(s["sequences"] for s in self._index),
            "shards": self._index,
        }
        (self.out_dir / "index.json").write_text(json.dumps(index, indent=2))
        log.info("packed %d tokens into %d shards", self._total, len(self._index))
        return index


def interleave_replay(
    domain_chunks: list[list[int]],
    replay_chunks: list[list[int]],
    replay_ratio: float = 0.25,
    seed: int = 0,
) -> list[list[int]]:
    """Mix general-domain text into the domain corpus.

    Non-optional at this corpus size. ~100M tokens of narrow financial prose against a
    post-trained checkpoint is exactly the regime where general capability and output
    format degrade, and a replay fraction is the standard mitigation. Interleaved rather
    than concatenated so the mixture is uniform across the run -- a block of replay at the
    end would leave the model's final state skewed toward it.
    """
    if not replay_chunks or replay_ratio <= 0:
        return domain_chunks

    rng = np.random.default_rng(seed)
    n_replay = int(len(domain_chunks) * replay_ratio / max(1 - replay_ratio, 1e-6))
    picks = rng.integers(0, len(replay_chunks), size=n_replay)

    merged = domain_chunks + [replay_chunks[i] for i in picks]
    order = rng.permutation(len(merged))
    return [merged[i] for i in order]
