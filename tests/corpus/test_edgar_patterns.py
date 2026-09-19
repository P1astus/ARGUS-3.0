"""EDGAR exhibit-name filtering. `sources/edgar.py` had no test coverage at all before
this -- these are the pure-regex parts, cheap to test and load-bearing (a missed exhibit
silently loses the substance of a filing; an over-matched one pulls in noise).
"""

from __future__ import annotations

from argus.engine.corpus.sources.edgar import (EXHIBIT_PATTERN, FORTYF_EXHIBIT_PATTERN,
                                                FULL_SUBMISSION, SKIP_FILES)


class TestFortyFExhibitPattern:
    """Measured live against SHOP plus 5 independent Canadian 40-F filers (CNI, ENB, TD,
    BCE, BMO): most use the same ex99N convention 8-K already handles; SHOP is the
    outlier with descriptive exhibit names. This pattern must catch both without
    changing EXHIBIT_PATTERN itself (8-K's matching is already proven in production)."""

    def test_matches_standard_ex99_naming_used_by_most_40f_filers(self):
        for name in ("tm261145d1_ex99-3.htm", "a16-23211_1ex99d1.htm", "ex991.htm",
                    "d86374dex991.htm", "d938207dex991.htm"):
            assert FORTYF_EXHIBIT_PATTERN.search(name), name

    def test_matches_shops_descriptive_exhibit_naming(self):
        for name in ("exhibit13mdaq42023.htm", "exhibit11annualinformation.htm"):
            assert FORTYF_EXHIBIT_PATTERN.search(name), name

    def test_still_matches_everything_the_8k_pattern_matches(self):
        # A superset property: broadening for 40-F must not narrow what already works.
        samples = ["ex99.1.htm", "q1fy27cfocommentary.htm", "prepared-remarks.htm"]
        for name in samples:
            assert EXHIBIT_PATTERN.search(name)
            assert FORTYF_EXHIBIT_PATTERN.search(name)

    def test_excludes_xbrl_viewer_and_index_noise(self):
        for name in ("R1.htm", "FilingSummary.xml", "0001594805-24-000007-index.html"):
            assert not FORTYF_EXHIBIT_PATTERN.search(name) or SKIP_FILES.search(name) \
                or FULL_SUBMISSION.match(name), (
                f"{name} must be excluded by the pattern or one of the existing "
                "SKIP_FILES/FULL_SUBMISSION guards")

    def test_does_not_change_the_original_8k_pattern_object(self):
        # FORTYF_EXHIBIT_PATTERN is built by extending EXHIBIT_PATTERN's source string,
        # not mutating it -- confirms the 8-K pattern is unaffected by the 40-F addition.
        assert "exhibit" not in EXHIBIT_PATTERN.pattern
        assert not EXHIBIT_PATTERN.search("exhibit13mdaq42023.htm")
