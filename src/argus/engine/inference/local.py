"""Local MLX generation.

Thin on purpose. The interesting decisions in this pipeline are which passages get
retrieved and whether a quote resolves; generation is plumbing, and the wrapper exists so
that (a) the extraction loop can be tested without a 14B model resident, and (b) the
base and instruct checkpoints can be swapped by name for the baseline comparison.

Greedy by default. Extraction is a copying task, and sampling temperature buys diversity
at the cost of exactly the thing being measured -- a model that paraphrases a quote by one
token has fabricated evidence, and temperature makes that strictly more likely.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)

BASE_MODEL = "mlx-community/Ministral-3-14B-Base-2512-4bit"
INSTRUCT_MODEL = "mlx-community/Ministral-3-14B-Instruct-2512-4bit"


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int = 900) -> str: ...


@dataclass
class GenConfig:
    model: str = BASE_MODEL
    max_tokens: int = 900
    temperature: float = 0.0
    stop: tuple[str, ...] = ()


@dataclass
class MLXGenerator:
    """Loads the model on first use and keeps it resident across calls."""

    config: GenConfig = field(default_factory=GenConfig)
    _model: object = None
    _tok: object = None
    total_tokens: int = 0
    total_seconds: float = 0.0

    @property
    def name(self) -> str:
        return self.config.model

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load
        t = time.time()
        log.info("loading %s", self.config.model)
        self._model, self._tok = load(self.config.model)
        log.info("loaded in %.1fs", time.time() - t)

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        self._load()
        n = max_tokens or self.config.max_tokens
        t = time.time()
        text = mlx_generate(
            self._model, self._tok, prompt=prompt, max_tokens=n, verbose=False,
            sampler=make_sampler(temp=self.config.temperature),
        )
        dt = time.time() - t
        self.total_seconds += dt
        self.total_tokens += len(self._tok.encode(text))
        return _truncate_at_stop(text, self.config.stop)


def _truncate_at_stop(text: str, stop: tuple[str, ...]) -> str:
    """Cut at the first stop sequence.

    mlx_lm 0.31.3's `generate` takes no stop-sequence argument, so this is applied after
    the fact. That costs tokens but not correctness -- and a base model continuing past
    its answer into a fresh few-shot block is the normal case, not an edge case.
    """
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


@dataclass
class ScriptedGenerator:
    """Returns canned outputs. Lets the extraction loop be tested end to end -- including
    the fabricated-quote path, which is hard to elicit on demand from a real model."""

    outputs: list[str]
    name: str = "scripted"
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, max_tokens: int = 900) -> str:
        self.calls.append(prompt)
        if not self.outputs:
            return ""
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
