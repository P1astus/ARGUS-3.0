"""SFT dataset generation from real Briefings. Uses ScriptedGenerator throughout -- no
MLX, no real model -- to test the pipeline's own logic (prompt construction, evidence
filtering, acceptance/rejection accounting) independent of whether a real base checkpoint
is good at the underlying task.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from argus.contracts.briefing import Briefing, Claim, ClaimKind, Source, SourceType
from argus.contracts.quant import SubSegment
from argus.engine.inference.local import ScriptedGenerator
from argus.engine.training.sft_data import (GenerationStats, build_prompt,
                                            generate_sft_example, render_evidence)

VALID_REC = json.dumps({
    "ticker": "NVDA", "sub_segment": "fabless", "as_of": "2024-05-01",
    "direction": "long", "conviction": 0.6, "chosen": "base", "target_holding_days": 10,
    "summary": "Data center demand is driving durable revenue growth.",
    "scenarios": [
        {"kind": "bull", "thesis": "Demand accelerates further.", "probability": 0.3,
         "levels": {"entry": 100.0, "target": 130.0, "invalidation": 90.0}},
        {"kind": "base", "thesis": "Growth continues at the current pace.",
         "probability": 0.5,
         "levels": {"entry": 100.0, "target": 115.0, "invalidation": 90.0}},
        {"kind": "bear", "thesis": "Demand cools faster than expected.",
         "probability": 0.2,
         "levels": {"entry": 100.0, "target": 85.0, "invalidation": 108.0}},
    ],
})

WRONG_TICKER_REC = VALID_REC.replace('"ticker": "NVDA"', '"ticker": "AMD"')


def _source():
    return Source(source_id="S1", source_type=SourceType.SEC_FILING,
                 retrieved_at=datetime.now(timezone.utc), content_sha256="a" * 8)


def _briefing(claims=None, ticker="NVDA"):
    claims = claims if claims is not None else [
        Claim(text="Revenue grew 12% driven by data center demand.",
             kind=ClaimKind.SOURCED, source_id="S1", quote="Revenue grew 12%"),
        Claim(text="This suggests durable demand.", kind=ClaimKind.INFERENCE),
    ]
    return Briefing(ticker=ticker, sub_segment=SubSegment.FABLESS, as_of=date(2024, 5, 1),
                    claims=claims, sources=[_source()])


class TestRenderEvidence:
    def test_only_sourced_claims_are_shown(self):
        b = _briefing()
        text = render_evidence(b)
        assert "Revenue grew 12%" in text
        assert "durable demand" not in text   # the INFERENCE claim must not appear

    def test_no_sourced_claims_says_so_explicitly(self):
        b = _briefing(claims=[Claim(text="just a guess", kind=ClaimKind.INFERENCE)])
        assert "no sourced claims" in render_evidence(b)

    def test_respects_max_claims(self):
        claims = [Claim(text=f"Fact {i}.", kind=ClaimKind.SOURCED, source_id="S1",
                        quote=f"quote {i}") for i in range(30)]
        b = _briefing(claims=claims)
        text = render_evidence(b, max_claims=5)
        assert text.count("Fact ") == 5


class TestBuildPrompt:
    def test_ends_exactly_at_the_continuation_point(self):
        prompt = build_prompt("NVDA", SubSegment.FABLESS, date(2024, 5, 1), _briefing())
        assert prompt.rstrip().endswith("### RECOMMENDATION")

    def test_includes_ticker_date_and_segment_framing(self):
        prompt = build_prompt("NVDA", SubSegment.FABLESS, date(2024, 5, 1), _briefing())
        assert "TICKER: NVDA" in prompt
        assert "2024-05-01" in prompt
        assert "Fabless chip designers" in prompt

    def test_fewshot_uses_fictional_tickers_not_real_ones(self):
        # Same reasoning as extract/prompts.py's XSAS: a real ticker in a worked example
        # risks being read as a verified fact rather than a format demonstration.
        prompt = build_prompt("NVDA", SubSegment.FABLESS, date(2024, 5, 1), _briefing())
        assert "XSAS" in prompt and "XMEM" in prompt

    def test_states_the_exact_sub_segment_key_the_json_must_echo(self):
        # Measured: without this, the model only sees the human-readable segment NAME
        # ("Semiconductor capital equipment") and has to guess the machine key
        # ("equipment") from two unrelated few-shot examples -- it guessed wrong for
        # segments those examples didn't demonstrate.
        prompt = build_prompt("AMAT", SubSegment.EQUIPMENT, date(2019, 11, 1), _briefing())
        assert '"equipment"' in prompt


class TestGenerateSFTExample:
    def test_accepts_a_valid_generation(self):
        gen = ScriptedGenerator(outputs=[VALID_REC])
        stats = GenerationStats()
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1),
                                  _briefing(), stats=stats)
        assert ex is not None
        assert stats.accepted == 1 and stats.attempted == 1

    def test_rejects_when_briefing_has_no_sourced_claims(self):
        b = _briefing(claims=[Claim(text="just a guess", kind=ClaimKind.INFERENCE)])
        gen = ScriptedGenerator(outputs=[VALID_REC])
        stats = GenerationStats()
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1), b,
                                  stats=stats)
        assert ex is None
        assert stats.rejected_no_evidence == 1
        assert len(gen.calls) == 0, "must not call the model with nothing to ground it"

    def test_rejects_malformed_json_after_retries(self):
        gen = ScriptedGenerator(outputs=["not json at all"])
        stats = GenerationStats()
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1),
                                  _briefing(), max_attempts=2, stats=stats)
        assert ex is None
        assert stats.rejected_invalid == 1
        assert len(gen.calls) == 2, "must actually retry, not give up after one attempt"

    def test_rejects_when_model_drifts_off_the_prompted_ticker(self):
        gen = ScriptedGenerator(outputs=[WRONG_TICKER_REC])
        stats = GenerationStats()
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1),
                                  _briefing(), stats=stats)
        assert ex is None
        assert stats.rejected_invalid == 1

    def test_accepted_example_completion_is_the_serialised_recommendation(self):
        gen = ScriptedGenerator(outputs=[VALID_REC])
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1),
                                  _briefing())
        completion = json.loads(ex.completion)
        assert completion["ticker"] == "NVDA"
        assert completion["direction"] == "long"

    def test_recovers_after_one_bad_attempt_then_a_good_one(self):
        gen = ScriptedGenerator(outputs=["garbage", VALID_REC])
        stats = GenerationStats()
        ex = generate_sft_example(gen, "NVDA", SubSegment.FABLESS, date(2024, 5, 1),
                                  _briefing(), max_attempts=3, stats=stats)
        assert ex is not None
        assert stats.accepted == 1
        assert stats.total_attempts_used == 2


class TestGenerationStats:
    def test_render_does_not_crash_when_nothing_attempted(self):
        assert "attempted=0" in GenerationStats().render()

    def test_mean_attempts_computed_correctly(self):
        s = GenerationStats(attempted=2, accepted=2, total_attempts_used=5)
        assert "mean_attempts=2.50" in s.render()
