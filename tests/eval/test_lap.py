"""LAP scoring correctness. No real setups have been probed yet (no trained model
exists) -- this verifies the machinery itself, so it is ready rather than theoretical
when the pilot CPT run needs it (handoff §8).
"""

from __future__ import annotations

from datetime import date

from argus.engine.eval.lap import (LAPCase, probe_prompt, run_probe, run_probes,
                                   score, split_by_lap)
from argus.engine.inference.local import ScriptedGenerator


def _case(ticker="NVDA", up=True, as_of=date(2024, 1, 5)):
    return LAPCase(ticker=ticker, as_of=as_of, horizon_days=10, realized_up=up)


class TestProbePrompt:
    def test_contains_ticker_and_date_only_no_leakage_of_the_answer(self):
        c = _case()
        prompt = probe_prompt(c)
        assert "NVDA" in prompt
        assert "2024-01-05" in prompt
        assert "UP" in prompt and "DOWN" in prompt   # the forced-choice instruction
        # The realised outcome must never appear in what the model is shown -- that
        # would make the probe measure prompt leakage, not pretraining recall.
        assert "true" not in prompt.lower()


class TestRunProbe:
    def test_parses_up_answer(self):
        gen = ScriptedGenerator(outputs=["UP"])
        r = run_probe(gen, _case(up=True))
        assert r.predicted_up is True
        assert r.correct is True

    def test_parses_down_answer(self):
        gen = ScriptedGenerator(outputs=["DOWN"])
        r = run_probe(gen, _case(up=True))
        assert r.predicted_up is False
        assert r.correct is False

    def test_case_insensitive_and_tolerates_surrounding_text(self):
        gen = ScriptedGenerator(outputs=["I recall it went down over that period."])
        r = run_probe(gen, _case(up=False))
        assert r.predicted_up is False
        assert r.correct is True

    def test_unparseable_response_is_not_scored_as_wrong(self):
        gen = ScriptedGenerator(outputs=["I don't have that information."])
        r = run_probe(gen, _case())
        assert r.predicted_up is None
        assert r.correct is None
        assert not r.parsed


class TestScore:
    def test_all_correct_gives_high_excess_over_chance(self):
        results = [run_probe(ScriptedGenerator(outputs=["UP"]), _case(up=True))
                  for _ in range(10)]
        s = score(results)
        assert s.recall_accuracy == 1.0
        assert s.excess_over_chance == 0.5

    def test_chance_level_gives_zero_excess(self):
        cases = [_case(ticker=f"T{i}", up=(i % 2 == 0)) for i in range(20)]
        # Model always says UP -- correct exactly on the up-cases, i.e. 50%.
        results = [run_probe(ScriptedGenerator(outputs=["UP"]), c) for c in cases]
        s = score(results)
        assert s.recall_accuracy == 0.5
        assert s.excess_over_chance == 0.0

    def test_unparsed_responses_excluded_from_accuracy_denominator(self):
        gen_ok = ScriptedGenerator(outputs=["UP"])
        gen_bad = ScriptedGenerator(outputs=["unclear"])
        results = [run_probe(gen_ok, _case(up=True)),
                  run_probe(gen_ok, _case(up=True)),
                  run_probe(gen_bad, _case(up=True))]
        s = score(results)
        assert s.n == 3
        assert s.n_parsed == 2
        assert s.recall_accuracy == 1.0   # unparsed excluded, not counted as wrong

    def test_no_parsed_responses_returns_none_not_a_crash(self):
        gen = ScriptedGenerator(outputs=["???"])
        results = [run_probe(gen, _case())]
        s = score(results)
        assert s.recall_accuracy is None
        assert s.excess_over_chance is None

    def test_empty_results(self):
        s = score([])
        assert s.n == 0 and s.recall_accuracy is None

    def test_contaminated_keys_only_include_correct_parsed_answers(self):
        results = [
            run_probe(ScriptedGenerator(outputs=["UP"]), _case(ticker="A", up=True)),
            run_probe(ScriptedGenerator(outputs=["UP"]), _case(ticker="B", up=False)),
            run_probe(ScriptedGenerator(outputs=["???"]), _case(ticker="C", up=True)),
        ]
        s = score(results)
        assert ("A", "2024-01-05") in s.contaminated_keys
        assert ("B", "2024-01-05") not in s.contaminated_keys   # wrong guess
        assert ("C", "2024-01-05") not in s.contaminated_keys   # unparsed


class TestRunProbes:
    def test_runs_each_case_through_the_generator(self):
        gen = ScriptedGenerator(outputs=["UP", "DOWN", "UP"])
        cases = [_case(ticker=t) for t in ("A", "B", "C")]
        results = run_probes(gen, cases)
        assert len(results) == 3
        assert [r.case.ticker for r in results] == ["A", "B", "C"]


class _Setup:
    def __init__(self, ticker, as_of):
        self.ticker, self.as_of = ticker, as_of


class TestSplitByLAP:
    def test_partitions_setups_by_contamination(self):
        results = [
            run_probe(ScriptedGenerator(outputs=["UP"]), _case(ticker="HOT", up=True)),
            run_probe(ScriptedGenerator(outputs=["???"]), _case(ticker="COLD", up=True)),
        ]
        lap = score(results)
        setups = [_Setup("HOT", date(2024, 1, 5)), _Setup("COLD", date(2024, 1, 5))]
        high, low = split_by_lap(setups, lap)
        assert [s.ticker for s in high] == ["HOT"]
        assert [s.ticker for s in low] == ["COLD"]

    def test_every_setup_lands_in_exactly_one_bucket(self):
        results = [run_probe(ScriptedGenerator(outputs=["UP"]), _case(ticker=f"T{i}"))
                  for i in range(5)]
        lap = score(results)
        setups = [_Setup(f"T{i}", date(2024, 1, 5)) for i in range(5)]
        high, low = split_by_lap(setups, lap)
        assert len(high) + len(low) == len(setups)
