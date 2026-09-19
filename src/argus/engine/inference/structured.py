"""Structured generation: model output -> validated Recommendation.

The SFT pass shapes the model to emit this format, but "shaped to" is not "guaranteed to".
Parsing is therefore defensive and retried, and a persistent failure is reported rather
than silently coerced -- format validity is a Phase 3 gate criterion, so quietly repairing
malformed output would inflate the very metric being measured.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from pydantic import ValidationError

from argus.contracts.recommendation import Recommendation

log = logging.getLogger(__name__)

# Models commonly wrap JSON in fences or prose. Reasoning models additionally emit
# <think> blocks, which must be stripped before parsing rather than confused for output.
_THINK = re.compile(r"(?is)<think>.*?</think>")
_FENCE = re.compile(r"(?s)```(?:json)?\s*(.*?)```")


@dataclass
class ParseResult:
    recommendation: Recommendation | None
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.recommendation is not None


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks.

    Relevant to model choice: the 8B distill emits these constantly. The selected
    Ministral base does not, but stripping is harmless and keeps the parser
    model-agnostic.
    """
    return _THINK.sub("", text).strip()


def extract_json(text: str) -> str | None:
    """Pull the most plausible JSON object out of a generation."""
    text = strip_reasoning(text)

    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()

    # Fall back to brace matching, which tolerates prose on either side.
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse(text: str) -> Recommendation | None:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return Recommendation.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None


def generate_structured(generate_fn, prompt: str, max_attempts: int = 3) -> ParseResult:
    """Generate until the output validates, or give up and say so.

    Each retry appends the validation error, so the model is told what was wrong rather
    than asked again identically. Failures are counted -- format_validity is a gate
    criterion, and a harness that silently retried forever would report 100%.
    """
    last_error: str | None = None
    current = prompt

    for attempt in range(1, max_attempts + 1):
        text = generate_fn(current)
        rec = parse(text)
        if rec is not None:
            return ParseResult(rec, attempt)

        raw = extract_json(text)
        if raw is None:
            last_error = "no JSON object found in output"
        else:
            try:
                Recommendation.model_validate(json.loads(raw))
            except ValidationError as e:
                last_error = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in e.errors()[:4])
            except json.JSONDecodeError as e:
                last_error = f"malformed JSON: {e}"

        log.warning("attempt %d/%d failed: %s", attempt, max_attempts, last_error)
        current = (f"{prompt}\n\nYour previous response was invalid: {last_error}\n"
                   f"Return ONLY a valid JSON object matching the schema.")

    return ParseResult(None, max_attempts, last_error)


SCHEMA_PROMPT = """Return ONLY a JSON object with this exact structure:

{
  "ticker": "NVDA",
  "sub_segment": "fabless",
  "as_of": "2026-03-14",
  "direction": "long" | "flat" | "short",
  "conviction": 0.0-1.0,
  "chosen": "bull" | "base" | "bear",
  "target_holding_days": 10,
  "summary": "one paragraph",
  "scenarios": [
    {"kind": "bull", "thesis": "...", "probability": 0.30,
     "levels": {"entry": 190.0, "target": 215.0, "invalidation": 178.0}},
    {"kind": "base", "thesis": "...", "probability": 0.50,
     "levels": {"entry": 190.0, "target": 205.0, "invalidation": 178.0}},
    {"kind": "bear", "thesis": "...", "probability": 0.20,
     "levels": {"entry": 190.0, "target": 170.0, "invalidation": 200.0}}
  ]
}

Rules:
- all three scenarios are required; probabilities must sum to ~1.0
- the chosen scenario must carry levels unless direction is "flat"
- "flat" is a legitimate answer. If no setup is attractive, say so rather than
  manufacturing a trade.
"""
