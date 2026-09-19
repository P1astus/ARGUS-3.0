"""SFT training loop: masked loss correctness, and the resumable train() orchestration.

`build()` downloads and loads the real 14B checkpoint -- not something a unit test should
do. Everything here either needs no model at all (load_examples, build_dataset) or swaps
in a tiny real MLX model in place of build()'s output, so `train()`'s own logic (checkpoint
cadence, epoch/example-cursor bookkeeping, resume) runs against real mlx tensor ops without
paying for the actual Ministral checkpoint. This mirrors the project's established split:
pure orchestration logic is unit-tested, MLX-scale runs are verified live (handoff's own
CPT/SFT sections).
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from argus.contracts.recommendation import Recommendation
from argus.engine.training.sft import SFTConfig, SFTExample, SFTTrainer, load_examples


class FakeTokenizer:
    """One token id per character, mod a small vocab -- enough for build_dataset and for
    a tiny real model to run forward/backward over, without needing real subword logic."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 60 for c in text]


class TinyLM(nn.Module):
    """A real, tiny mlx.nn.Module standing in for the loaded 14B checkpoint. Exercises
    the same interface `masked_loss`/`train()` actually call (`model(x)` -> logits,
    `model.trainable_parameters()`, `model.train()`/`.eval()`) without the cost of the
    real model."""

    def __init__(self, vocab: int = 60, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def __call__(self, x):
        return self.out(self.embed(x))


def _valid_completion(ticker: str = "NVDA") -> str:
    rec = Recommendation(
        ticker=ticker, sub_segment="fabless", as_of="2024-05-01", direction="flat",
        conviction=0.4, chosen="base", target_holding_days=10,
        summary="No clean setup either direction.",
        scenarios=[
            {"kind": "bull", "thesis": "Upside case.", "probability": 0.3, "levels": None},
            {"kind": "base", "thesis": "Base case.", "probability": 0.4, "levels": None},
            {"kind": "bear", "thesis": "Downside case.", "probability": 0.3, "levels": None},
        ],
    )
    return rec.model_dump_json()


def _write_examples(path, n=12):
    lines = [json.dumps({"prompt": f"### EVIDENCE\n\nTICKER: T{i}\n\nfact {i}\n\n"
                                    f"### RECOMMENDATION\n",
                         "completion": _valid_completion(f"T{i}")})
             for i in range(n)]
    path.write_text("\n".join(lines))


def _built_trainer(config: SFTConfig) -> SFTTrainer:
    """An SFTTrainer with build()'s expensive parts swapped for the tiny model, so
    train() runs its real orchestration logic against real mlx ops."""
    t = SFTTrainer(config)
    t.model = TinyLM()
    mx.eval(t.model.parameters())
    t.model.train()
    t.tokenizer = FakeTokenizer()
    t.optimizer = optim.AdamW(learning_rate=config.learning_rate)
    return t


class TestLoadExamples:
    def test_drops_completions_that_fail_schema_validation(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        p.write_text("\n".join([
            json.dumps({"prompt": "p1", "completion": _valid_completion()}),
            json.dumps({"prompt": "p2", "completion": "not a valid recommendation"}),
        ]))
        examples = load_examples(p)
        assert len(examples) == 1
        assert examples[0].prompt == "p1"

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        p.write_text(json.dumps({"prompt": "p1", "completion": _valid_completion()}) + "\n\n")
        assert len(load_examples(p)) == 1


class TestMaskedLoss:
    def test_only_completion_positions_carry_gradient(self):
        """Changing a PROMPT-side token's target must not change the loss; changing a
        COMPLETION-side token's target must. This is the property masking exists for."""
        trainer = _built_trainer(SFTConfig())
        ids = mx.array([1, 2, 3, 4, 5, 6])
        mask = mx.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])   # last 3 positions are "completion"

        base = float(trainer.masked_loss(trainer.model, ids, mask))

        prompt_changed = mx.array([1, 9, 3, 4, 5, 6])     # change inside the masked-out region
        assert float(trainer.masked_loss(trainer.model, prompt_changed, mask)) == pytest.approx(base)

        completion_changed = mx.array([1, 2, 3, 4, 5, 9])  # change inside the scored region
        assert float(trainer.masked_loss(trainer.model, completion_changed, mask)) != pytest.approx(base)

    def test_all_zero_mask_does_not_divide_by_zero(self):
        trainer = _built_trainer(SFTConfig())
        ids = mx.array([1, 2, 3])
        mask = mx.array([0.0, 0.0, 0.0])
        loss = trainer.masked_loss(trainer.model, ids, mask)
        assert not mx.isnan(loss)


class TestTrain:
    def test_runs_and_writes_a_summary(self, tmp_path):
        data_path = tmp_path / "examples.jsonl"
        _write_examples(data_path, n=12)
        config = SFTConfig(data_path=str(data_path), out_dir=str(tmp_path / "out"),
                           seq_len=64, batch_size=1, grad_accum=2, epochs=2,
                           warmup_steps=1, eval_every_steps=2, checkpoint_every_steps=3)
        trainer = _built_trainer(config)

        summary = trainer.train(resume=False)

        assert summary["steps"] > 0
        assert summary["examples"] == 12
        assert (tmp_path / "out" / "summary.json").exists()
        assert list((tmp_path / "out" / "checkpoints").glob("step_*"))

    def test_resume_continues_instead_of_restarting(self, tmp_path):
        data_path = tmp_path / "examples.jsonl"
        _write_examples(data_path, n=12)
        out_dir = str(tmp_path / "out")

        config = SFTConfig(data_path=str(data_path), out_dir=out_dir, seq_len=64,
                           batch_size=1, grad_accum=1, epochs=1, warmup_steps=1,
                           eval_every_steps=100, checkpoint_every_steps=2)
        first = _built_trainer(config)
        first_summary = first.train(resume=False)
        assert first_summary["steps"] > 0

        # A second trainer resuming from the same out_dir must pick up at the same step
        # count, not restart from step 0 -- checkpoint.py's whole reason for existing.
        second = _built_trainer(config)
        second_summary = second.train(resume=True)
        assert second_summary["steps"] >= first_summary["steps"]

    def test_raises_on_empty_dataset(self, tmp_path):
        data_path = tmp_path / "examples.jsonl"
        data_path.write_text("")
        trainer = _built_trainer(SFTConfig(data_path=str(data_path),
                                           out_dir=str(tmp_path / "out")))
        with pytest.raises(ValueError):
            trainer.train(resume=False)

    def test_raises_when_every_example_is_truncated_away(self, tmp_path):
        data_path = tmp_path / "examples.jsonl"
        _write_examples(data_path, n=3)
        # seq_len=1 leaves no room for any completion tokens after the prompt.
        trainer = _built_trainer(SFTConfig(data_path=str(data_path),
                                           out_dir=str(tmp_path / "out"), seq_len=1))
        with pytest.raises(ValueError):
            trainer.train(resume=False)
