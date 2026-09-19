"""Local dense embeddings (MLX).

Model: `bge-small-en-v1.5` (33M params, 384 dims, 512-token window). Small on purpose --
the dense side of retrieval runs after a hard metadata filter that has already cut the
candidate set from ~300k chunks to a few hundred, so embedding *quality at the margin*
matters far less than being able to index the whole corpus in one sitting. Measured at
151 chunks/s at batch 64 on this machine (see docs/retrieval.md).

BGE asymmetry: queries are prefixed with an instruction, passages are not. This is part
of how the model was trained; skipping it measurably degrades retrieval, so the prefix
lives here rather than at call sites where it could be forgotten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL = "mlx-community/bge-small-en-v1.5-bf16"

# Sequence lengths are rounded up to a multiple of this before padding.
#
# MEASURED, AND THE CAUSE OF A 17-MINUTE STALL: with `padding=True` every batch pads to
# its own longest member, so consecutive batches ask MLX for tensors of shape (64, 431),
# (64, 468), (64, 448)... Each distinct shape allocates fresh Metal buffers that the
# allocator caches and never gets to reuse, and the cache grew to 49GB in 20 seconds --
# against a 55.7GB working set. Past that the allocator thrashes and throughput collapses
# from 130 chunks/s to under 1. Bucketing collapses hundreds of shapes into eight, so the
# cache is reused instead of grown. `cache_limit` is the belt to this braces.
LENGTH_BUCKET = 64
DEFAULT_CACHE_LIMIT = 4 * 1024 ** 3


@dataclass
class EmbedConfig:
    model: str = DEFAULT_MODEL
    dim: int = 384
    batch_size: int = 64
    max_length: int = 512
    cache_limit_bytes: int = DEFAULT_CACHE_LIMIT


class Embedder:
    """Lazily-loaded MLX encoder. Import of mlx is deferred so the retrieval package
    remains importable (and testable) on a machine without MLX."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_embeddings.utils import load
        log.info("loading embedding model %s", self.config.model)
        self._model, self._tok = load(self.config.model)
        if self.config.cache_limit_bytes:
            mx.set_cache_limit(self.config.cache_limit_bytes)

    def _bucket(self, batch: list[str]) -> int:
        """Longest sequence in the batch, rounded up to a LENGTH_BUCKET multiple.

        Tokenising twice (once to measure, once to pad) costs ~0.4ms per batch against a
        ~3.4s forward pass, which is a rounding error next to the allocator behaviour it
        prevents.
        """
        longest = max((len(self._tok._tokenizer(t)["input_ids"]) for t in batch),
                      default=LENGTH_BUCKET)
        longest = min(longest, self.config.max_length)
        return min(self.config.max_length,
                   max(LENGTH_BUCKET,
                       -(-longest // LENGTH_BUCKET) * LENGTH_BUCKET))

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode to L2-normalised float32. Normalising here means every consumer can
        treat similarity as a plain dot product."""
        import mlx.core as mx

        if not texts:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        self._load()
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        out = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            enc = self._tok.batch_encode_plus(
                batch, return_tensors="mlx", padding="max_length", truncation=True,
                max_length=self._bucket(batch))
            res = self._model(enc["input_ids"], attention_mask=enc["attention_mask"])
            vecs = np.array(res.text_embeds, copy=False).astype(np.float32)
            out.append(vecs)

        mat = np.vstack(out)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
