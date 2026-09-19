"""Phase 3 gate logic. No coverage existed for this before -- adding it here focuses on
the LAP conditioning this session added, plus the pre-existing guard behaviour it plugs
into, since a silent regression in either would let a contaminated result read as a clean
pass.
"""

from __future__ import annotations

from argus.contracts.provenance import Arm
from argus.engine.eval.gate import ArmResult, decide
from argus.engine.eval.lap import LAPScore


def _arm(arm, accuracy=0.6, n=200, ci=(0.53, 0.67), **kw):
    return ArmResult(arm=str(arm), n=n, accuracy=accuracy, accuracy_ci=ci, **kw)


class TestLAPConditioning:
    def test_missing_lap_probe_warns_that_contamination_is_unmeasured(self):
        d = decide({str(Arm.CPT_SFT): _arm(Arm.CPT_SFT),
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        assert any("UNMEASURED" in w for w in d.warnings)

    def test_low_lap_excess_does_not_warn(self):
        cpt = _arm(Arm.CPT_SFT, lap=LAPScore(100, 100, 0.52, 0.02))
        d = decide({str(Arm.CPT_SFT): cpt,
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        assert not any("LAP recall is elevated" in w for w in d.warnings)
        assert not any("UNMEASURED" in w for w in d.warnings)

    def test_high_lap_excess_warns_and_prefers_restricted_accuracy(self):
        cpt = _arm(Arm.CPT_SFT, lap=LAPScore(100, 100, 0.75, 0.25),
                  accuracy_low_lap=0.51)
        d = decide({str(Arm.CPT_SFT): cpt,
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        warning = next(w for w in d.warnings if "LAP recall is elevated" in w)
        assert "0.510" in warning and "0.600" in warning

    def test_high_lap_without_restricted_accuracy_still_warns(self):
        cpt = _arm(Arm.CPT_SFT, lap=LAPScore(100, 100, 0.75, 0.25))
        d = decide({str(Arm.CPT_SFT): cpt,
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        assert any("not re-measured" in w for w in d.warnings)

    def test_lap_warning_does_not_affect_pass_fail(self):
        # LAP is a diagnostic, not a gate criterion (see gate.py's own rationale) --
        # a contaminated-looking result must still show as PASS on the pre-registered
        # criteria; the warning is how a human catches what the criteria alone would not.
        clean = decide({str(Arm.CPT_SFT): _arm(Arm.CPT_SFT),
                        str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        contaminated = decide({
            str(Arm.CPT_SFT): _arm(Arm.CPT_SFT, lap=LAPScore(100, 100, 0.9, 0.4)),
            str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        assert clean.passed == contaminated.passed == True
        assert clean.criteria == contaminated.criteria


class TestGateBasics:
    def test_no_cpt_arm_fails_immediately(self):
        d = decide({str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        assert not d.passed
        assert "no CPT_SFT arm was evaluated" in d.warnings

    def test_uninformative_quant_arm_is_excluded_not_favourable(self):
        quant = _arm(Arm.QUANT_ONLY, accuracy=0.51, ci=(0.44, 0.58))  # near chance
        d = decide({str(Arm.CPT_SFT): _arm(Arm.CPT_SFT),
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5),
                   str(Arm.QUANT_ONLY): quant})
        assert "beats quant-only arm" not in d.criteria
        assert any("UNINFORMATIVE" in w for w in d.warnings)

    def test_missing_base_prompt_arm_warns_about_measuring_the_prompt(self):
        d = decide({str(Arm.CPT_SFT): _arm(Arm.CPT_SFT)})
        assert any("measures prompt engineering" in w for w in d.warnings)

    def test_render_includes_pass_and_criteria(self):
        d = decide({str(Arm.CPT_SFT): _arm(Arm.CPT_SFT),
                   str(Arm.BASE_PROMPT): _arm(Arm.BASE_PROMPT, accuracy=0.5)})
        text = d.render()
        assert "PASS" in text
        assert "cpt_sft" in text
