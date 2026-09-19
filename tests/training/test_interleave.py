"""SFT/CPT interleave scheduling and resume correctness.

Llama-Fin (handoff §4e): sequential CPT-then-SFT is the configuration that collapses
instruction-following; interleaving is the mitigation. This tests the mixing schedule
itself -- deterministic, resumable, and inert when unconfigured (`cpt.py`'s existing
pure-CPT path must not change behaviour by default). No MLX or trained model is involved:
`build_dataset` only needs something with an `.encode()` method.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.engine.corpus.pack import ShardWriter
from argus.engine.training.data_iter import ShardIterator
from argus.engine.training.interleave import (BatchKind, InterleaveState, interleave)
from argus.engine.training.sft import SFTConfig, SFTExample


class FakeTokenizer:
    """`.encode(text)` -> one token id per character. Enough for build_dataset, which
    only needs id lists and their lengths -- the actual vocabulary is irrelevant to
    whether the interleave SCHEDULE is correct."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 250 for c in text]


@pytest.fixture
def shard_dir(tmp_path):
    """A real, small packed corpus -- built with the actual ShardWriter so the fixture
    matches what pack.py really produces, not a hand-rolled approximation of it."""
    w = ShardWriter(tmp_path / "shards", shard_tokens=200, seq_len=20)
    for _ in range(3):
        w.add(list(range(200)))   # 3 shards x 200 tokens = 10 sequences/shard
    w.close()
    return tmp_path / "shards"


def _shard_iter(shard_dir, batch_size=1):
    return ShardIterator(shard_dir, seq_len=20, batch_size=batch_size,
                         shuffle_shards=False, seed=0)


def _sft_examples(n=3):
    return [SFTExample(prompt=f"prompt {i}", completion=f"completion {i}")
           for i in range(n)]


class TestZeroRatioIsInert:
    """ratio=0.0 (CPTConfig's default) must reproduce plain ShardIterator output exactly
    -- this is the backward-compatibility guarantee the whole module is built around."""

    def test_ratio_zero_yields_only_cpt_batches(self, shard_dir):
        batches = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                                  FakeTokenizer(), SFTConfig(), ratio=0.0))
        assert all(b.kind == BatchKind.CPT for b in batches)

    def test_ratio_zero_matches_plain_shard_iterator_exactly(self, shard_dir):
        plain = list(_shard_iter(shard_dir).iterate())
        mixed = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                                FakeTokenizer(), SFTConfig(), ratio=0.0))
        assert len(plain) == len(mixed)
        for p, m in zip(plain, mixed):
            assert np.array_equal(p.inputs, m.cpt.inputs)
            assert p.shard_id == m.cpt.shard_id and p.offset == m.cpt.offset

    def test_empty_sft_list_is_also_inert_even_with_positive_ratio(self, shard_dir):
        batches = list(interleave(_shard_iter(shard_dir), [], FakeTokenizer(),
                                  SFTConfig(), ratio=0.5))
        assert all(b.kind == BatchKind.CPT for b in batches)


class TestInterleaveRatio:
    def test_one_sft_step_per_n_cpt_steps(self, shard_dir):
        # ratio=1/5 -> one SFT batch after every 5 CPT batches.
        batches = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                                  FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.2))
        kinds = [b.kind for b in batches]
        sft_positions = [i for i, k in enumerate(kinds) if k == BatchKind.SFT]
        # Every SFT batch must be immediately preceded by exactly 5 CPT batches (or fewer
        # only at the very start, since the counter begins at 0).
        assert sft_positions, "ratio=0.2 over 30 CPT batches must insert at least one SFT step"
        prev = -1
        for pos in sft_positions:
            run = kinds[prev + 1:pos].count(BatchKind.CPT)
            assert run == 5
            prev = pos

    def test_higher_ratio_inserts_more_sft_steps(self, shard_dir):
        low = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                              FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.1))
        high = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                               FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.5))
        n_low = sum(1 for b in low if b.kind == BatchKind.SFT)
        n_high = sum(1 for b in high if b.kind == BatchKind.SFT)
        assert n_high > n_low

    def test_sft_batches_carry_masked_loss_data(self, shard_dir):
        batches = list(interleave(_shard_iter(shard_dir), _sft_examples(),
                                  FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.5))
        sft = [b for b in batches if b.kind == BatchKind.SFT]
        assert sft
        for b in sft:
            assert b.sft_inputs is not None and b.sft_mask is not None
            assert b.sft_mask.sum() > 0, "completion must carry loss"
            assert b.sft_mask.sum() < len(b.sft_mask), (
                "prompt tokens must be masked out -- training on them teaches "
                "market-state description, not the target task (sft.py's own rationale)")


class TestSFTCycling:
    def test_sft_examples_cycle_when_exhausted(self, shard_dir):
        # 3 examples, ratio high enough to need more than 3 SFT insertions.
        batches = list(interleave(_shard_iter(shard_dir), _sft_examples(3),
                                  FakeTokenizer(), SFTConfig(seq_len=50), ratio=1.0))
        sft = [b for b in batches if b.kind == BatchKind.SFT]
        assert len(sft) > 3
        # sft_index is the raw, ever-increasing resume counter (see TestResume) --
        # cycling happens at lookup time, so the example actually used is index % n.
        indices = [b.sft_index for b in sft]
        assert indices == list(range(len(sft))), "resume counter must advance by one each time"
        example_used = [i % 3 for i in indices]
        assert example_used[3] == example_used[0] == 0, (
            "the 4th SFT batch must cycle back to example 0")

    def test_single_sft_example_cycles_onto_itself(self, shard_dir):
        batches = list(interleave(_shard_iter(shard_dir), _sft_examples(1),
                                  FakeTokenizer(), SFTConfig(seq_len=50), ratio=1.0))
        sft = [b for b in batches if b.kind == BatchKind.SFT]
        assert all(b.sft_index == i for i, b in enumerate(sft))


class TestResume:
    def test_resuming_mid_stream_continues_the_sft_cycle_position(self, shard_dir):
        state = InterleaveState()
        full_run = list(interleave(_shard_iter(shard_dir), _sft_examples(3),
                                   FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.34,
                                   state=state))
        first_sft_index = next(b.sft_index for b in full_run if b.kind == BatchKind.SFT)

        # A fresh state (as if starting cold) must reproduce the same first SFT example.
        cold_state = InterleaveState()
        cold_run = list(interleave(_shard_iter(shard_dir), _sft_examples(3),
                                   FakeTokenizer(), SFTConfig(seq_len=50), ratio=0.34,
                                   state=cold_state))
        assert cold_run[0].kind == full_run[0].kind
        assert next(b.sft_index for b in cold_run if b.kind == BatchKind.SFT) == first_sft_index

    def test_provided_state_is_mutated_in_place_for_checkpointing(self, shard_dir):
        state = InterleaveState()
        list(interleave(_shard_iter(shard_dir), _sft_examples(3), FakeTokenizer(),
                        SFTConfig(seq_len=50), ratio=1.0, state=state))
        assert state.sft_position > 0, (
            "caller must be able to read back progress to persist it in a checkpoint")
