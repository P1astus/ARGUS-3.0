"""Interleave SFT-format examples into the CPT token stream.

WHY THIS EXISTS
Llama-Fin (arXiv 2501.04961, handoff §4e): CPT alone collapsed instruction-following from
7.8 to ~1.0 on MT-Bench; running CPT then SFT sequentially -- which is what `cpt.py` then
`sft.py` does today -- is exactly the configuration that collapse happens under. Joint
CPT+IT training prevented it. Their corpus was 6B tokens; ours is ~0.16B (packed
seq_len=4096, ~103M tokens for semis alone), roughly 40x smaller, so if their model needed
mixing to avoid degrading, there is no reason to expect ours does not.

WHAT INTERLEAVING MEANS HERE
Not a separate training phase -- individual SFT examples inserted into the same step
sequence CPT already iterates, at a configurable ratio, each carrying its own loss
(next-token everywhere for a CPT batch; masked to the completion only for an SFT batch,
reusing sft.py's masking rationale exactly). The training loop picks the right loss
function per batch by its `kind` tag; nothing about batch *shape* needs to unify.

BACKWARD COMPATIBILITY IS LOAD-BEARING
`CPTConfig.sft_interleave_ratio` defaults to 0.0. At that setting `interleave()` is not
even called -- `cpt.py`'s existing pure-CPT path is untouched, byte-for-byte, because that
path already has hours of runway ahead of it (the eventual pilot run) and changing its
behavior by default would be exactly the kind of silent scope change this project's
pre-registration discipline (§4d) exists to prevent.

WHAT IS NOT VALIDATED
No SFT dataset exists yet (handoff's own state table: "SFT dataset -- not built"), and no
CPT run has happened. This is readiness -- the mixing logic is deterministic and fully
unit-tested without MLX -- not a result. Do not treat "interleaving is implemented" as
"interleaving was measured to help"; only a real run with and without it would show that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

import numpy as np

from argus.engine.training.data_iter import Batch, ShardIterator
from argus.engine.training.sft import SFTConfig, SFTExample, build_dataset


class BatchKind(StrEnum):
    CPT = "cpt"     # plain next-token loss over the whole sequence
    SFT = "sft"      # loss masked to the completion (sft.py's masked_loss)


@dataclass
class InterleavedBatch:
    kind: BatchKind
    cpt: Batch | None = None                          # set when kind == CPT
    sft_inputs: np.ndarray | None = None               # set when kind == SFT
    sft_mask: np.ndarray | None = None
    sft_index: int = 0        # position in the (cycled) SFT example list -- for resume


@dataclass
class InterleaveState:
    """The extra resume state interleaving needs on top of ShardIterator's own
    (shard_id, offset) cursor -- how far through the cycling SFT example list the run
    had gotten, and how many CPT batches have been emitted since the last SFT insertion,
    so a resumed run reproduces the same interleave pattern rather than restarting the
    ratio accounting from zero."""

    sft_position: int = 0
    cpt_batches_since_sft: int = 0


def _sft_due(ratio: float, batches_since_sft: int) -> bool:
    """Deterministic scheduling: insert an SFT step once `1/ratio` CPT batches have
    passed. A ratio of 0.1 means "1 SFT step per 10 CPT steps", not a coin flip -- a
    stochastic schedule would make two runs with the same seed diverge in which tokens
    get how much gradient signal, which is exactly the kind of nondeterminism
    `data_iter.py`'s own docstring calls out as unacceptable for a resumable multi-day
    job.
    """
    if ratio <= 0:
        return False
    every = max(1, round(1.0 / ratio))
    return batches_since_sft >= every


def interleave(shard_iter: ShardIterator, sft_examples: list[SFTExample], tokenizer,
              sft_config: SFTConfig, ratio: float,
              state: InterleaveState | None = None,
              start_shard: int = 0, start_offset: int = 0,
              epochs: int = 1) -> Iterator[InterleavedBatch]:
    """Mix `shard_iter`'s CPT batches with cycling SFT examples at `ratio`.

    `ratio` is SFT batches per CPT batch, e.g. 0.1 -> roughly one SFT step per ten CPT
    steps. If `sft_examples` is empty this degenerates to a pure CPT stream regardless of
    `ratio` -- there is nothing to interleave, and failing loudly would make an
    empty-by-accident SFT set (e.g. every example dropped by schema validation, see
    sft.load_examples) crash a run that could otherwise proceed as CPT-only.
    """
    st = state or InterleaveState()

    if not sft_examples or ratio <= 0:
        for batch in shard_iter.iterate(start_shard=start_shard,
                                        start_offset=start_offset, epochs=epochs):
            yield InterleavedBatch(kind=BatchKind.CPT, cpt=batch)
        return

    tokenised = build_dataset(sft_examples, tokenizer, sft_config)
    if not tokenised:
        # Every example was truncated to nothing by seq_len (build_dataset's own guard) --
        # same reasoning as the empty-list case above.
        for batch in shard_iter.iterate(start_shard=start_shard,
                                        start_offset=start_offset, epochs=epochs):
            yield InterleavedBatch(kind=BatchKind.CPT, cpt=batch)
        return

    for batch in shard_iter.iterate(start_shard=start_shard, start_offset=start_offset,
                                    epochs=epochs):
        yield InterleavedBatch(kind=BatchKind.CPT, cpt=batch)
        st.cpt_batches_since_sft += 1

        if _sft_due(ratio, st.cpt_batches_since_sft):
            ids, mask = tokenised[st.sft_position % len(tokenised)]
            yield InterleavedBatch(kind=BatchKind.SFT, sft_inputs=ids, sft_mask=mask,
                                   sft_index=st.sft_position)
            st.sft_position += 1
            st.cpt_batches_since_sft = 0
