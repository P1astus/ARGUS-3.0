"""Narrative section extraction -- no coverage existed for this despite it being the
gate every retained document passes through. `extract_narrative` decides what text
becomes chunkable and quotable; a mis-extracted section either loses real content or
(worse, and what most of this file guards against) silently splices unrelated text into
a section under the wrong label.
"""

from __future__ import annotations

from argus.engine.corpus.clean import (drop_numeric_tables, extract_narrative,
                                       find_sections, strip_html)

TENK = """<html><body>
<p>TABLE OF CONTENTS</p>
<p>Item 1. Business ... 3</p>
<p>Item 1A. Risk Factors ... 8</p>
<p>Item 7. Management's Discussion and Analysis ... 20</p>

<p>Item 1. Business</p>
<p>We design and sell semiconductor devices for the industrial and automotive markets,
with manufacturing facilities in three countries and a broad distributor network.</p>

<p>Item 1A. Risk Factors</p>
<p>Our results depend on cyclical demand in the semiconductor industry, which has
historically been volatile and difficult to predict over any given quarter.</p>

<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased 12% year over year, driven by strength in the automotive segment
and partially offset by softness in industrial demand during the back half of the year.</p>
</body></html>"""


class TestTenKExtraction:
    def test_finds_all_three_kept_sections(self):
        sections = extract_narrative(TENK, min_chars=10)
        assert set(sections) == {"business", "risk_factors", "mdna"}

    def test_business_section_is_the_real_content_not_the_toc(self):
        sections = extract_narrative(TENK, min_chars=10)
        assert "semiconductor devices" in sections["business"]
        assert "TABLE OF CONTENTS" not in sections["business"]

    def test_mdna_section_content(self):
        sections = extract_narrative(TENK, min_chars=10)
        assert "Revenue increased 12%" in sections["mdna"]

    def test_sections_are_literal_substrings_of_the_persisted_document(self):
        # The persisted "document" a chunk later slices into is drop_numeric_tables'
        # output, not the raw strip_html intermediate -- that intermediate is never
        # written to disk. Neither fixture has numeric-table lines to drop, so applying
        # it to the whole document is a valid baseline for this comparison.
        text = drop_numeric_tables(strip_html(TENK))
        for name, content in extract_narrative(TENK, min_chars=10).items():
            assert content in text, f"{name} is not verbatim from the persisted document"


# Reproduces the real bug found against TSM's actual 20-F (accession
# 0001628280-26-025362): the narrative cites its own item numbers in mixed case --
# "see Item 4. Information on the Company for further discussion" -- and those citations
# can sort AFTER the genuine section header, which the filing renders in full caps. The
# original "last occurrence" heuristic (correct for 10-K, where this doesn't happen) then
# picked a cross-reference sentence instead of the real header.
TWENTYF = """<html><body>
<p>ITEM 4. \n INFORMATION ON THE COMPANY \n 14</p>
<p>ITEM 5. OPERATING AND FINANCIAL REVIEWS AND PROSPECTS \n 40</p>

<p>ITEM 4. INFORMATION ON THE COMPANY</p>
<p>Our History and Structure. We were founded in 1987 as a joint venture and have
grown into a leading provider of semiconductor manufacturing services worldwide.</p>
<p>Item 4. Information on the Company Risk Management, see Item 16K. Cybersecurity for
a further discussion of our risk management program and governance structure.</p>

<p>ITEM 3. KEY INFORMATION</p>
<p>Risk Factors. We wish to caution readers that our results are subject to significant
fluctuation due to the highly cyclical nature of the semiconductor industry.</p>

<p>ITEM 5. OPERATING AND FINANCIAL REVIEWS AND PROSPECTS</p>
<p>Net revenue increased in the period, driven by strong demand for advanced process
nodes across our foundry customer base during the fiscal year.</p>

<p>ITEM 11. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISKS</p>
<p>We are exposed to foreign currency risk because a majority of our capital
expenditures are denominated in currencies other than our functional currency.</p>
</body></html>"""


class TestTwentyFExtraction:
    def test_finds_all_four_kept_sections(self):
        sections = extract_narrative(TWENTYF, form_type="20-F", min_chars=10)
        assert set(sections) == {"business", "risk_factors", "mdna",
                                 "quantitative_qualitative"}

    def test_business_section_starts_at_the_real_header_not_a_cross_reference(self):
        sections = extract_narrative(TWENTYF, form_type="20-F", min_chars=10)
        assert "Our History and Structure" in sections["business"]
        # The mixed-case cross-reference must not be what the span was anchored on --
        # this is the exact TSM bug: picking it instead loses the real section content.
        assert "joint venture" in sections["business"]

    def test_mdna_maps_item_5_not_item_7(self):
        sections = extract_narrative(TWENTYF, form_type="20-F", min_chars=10)
        assert "Net revenue increased" in sections["mdna"]

    def test_risk_factors_maps_item_3_not_item_1a(self):
        sections = extract_narrative(TWENTYF, form_type="20-F", min_chars=10)
        assert "cyclical nature" in sections["risk_factors"]

    def test_without_form_type_20f_headers_are_not_recognised(self):
        # Safety property: running a 20-F through the DEFAULT (10-K) patterns must not
        # silently mislabel content -- it should find nothing structured and fall back to
        # whole-document "full", never split on the wrong item numbers.
        sections = extract_narrative(TWENTYF, min_chars=10)
        assert "business" not in sections
        assert "risk_factors" not in sections

    def test_without_prefer_caps_the_cross_reference_bug_reproduces(self):
        # Documents the failure this fix addresses: with caps-preference off, the
        # cross-reference (later in the document) wins over the real header.
        text = strip_html(TWENTYF)
        from argus.engine.corpus.clean import SECTION_PATTERNS_20F
        spans_loose = find_sections(text, SECTION_PATTERNS_20F, prefer_caps=False)
        spans_strict = find_sections(text, SECTION_PATTERNS_20F, prefer_caps=True)
        assert spans_loose["business"][0] != spans_strict["business"][0]
        business_strict = text[spans_strict["business"][0]:spans_strict["business"][1]]
        assert "Our History and Structure" in business_strict


# Reproduces the real structure measured against Toronto-Dominion Bank's 40-F exhibit
# (CIK 947263, accession containing ex991.htm) -- a DELIBERATELY different, independent
# filer from SHOP, to check the headings generalise rather than being one company's
# convention. Confirmed live: "Risk Factors" and "Management's Discussion and Analysis"
# both appear multiple times (table of contents with page numbers, then a genuine
# cross-reference, THEN the real section) before the actual header; the real header is
# the LAST occurrence, same shape as the 10-K/20-F table-of-contents problem.
FORTYF_EXHIBIT = """<html><body>
<p>ANNUAL INFORMATION FORM</p>
<p>Table of Contents ... Approach to Sustainability 5 ... Risk Factors 5 ... Dividends ...
Description of the Business 8 ... Management's Discussion and Analysis 12</p>

<p>GENERAL DEVELOPMENT OF THE BUSINESS</p>
<p>Three Year History. Prior to October the Bank completed several acquisitions that
expanded its footprint across multiple jurisdictions and business lines significantly.</p>

<p>DESCRIPTION OF THE BUSINESS</p>
<p>The Bank and its subsidiaries provide a broad range of financial products and services
to more than twenty-seven million customers worldwide through several business segments
including personal and commercial banking, wealth management, and wholesale banking.</p>

<p>Examples of such risk factors include general business and economic conditions in the
regions in which the Bank operates, as referenced in the forward-looking statements
section of this annual information form and incorporated herein by reference.</p>

<p>Risk Factors</p>
<p>The Bank considers it critical to regularly assess its operating environment and the
risks that could affect its ability to achieve its strategic objectives and financial
targets, including credit risk, market risk, liquidity risk, and operational risk.</p>
</body></html>"""

FORTYF_MDNA_EXHIBIT = """<html><body>
<p>TD BANK GROUP ANNUAL REPORT</p>
<p>Page 1</p>
<p>Management's Discussion and Analysis</p>
<p>This Management's Discussion and Analysis (MD&A) is presented to enable readers to
assess material changes in the financial condition and results of operations of the
Bank for the year, compared with the prior year, and should be read in conjunction with
the audited consolidated financial statements for the year and related notes.</p>
</body></html>"""


class TestFortyFExtraction:
    def test_finds_business_and_risk_factors_from_the_aif_exhibit(self):
        sections = extract_narrative(FORTYF_EXHIBIT, form_type="40-F/EX", min_chars=10)
        assert "business" in sections
        assert "risk_factors" in sections

    def test_business_section_is_the_real_header_not_the_toc(self):
        sections = extract_narrative(FORTYF_EXHIBIT, form_type="40-F/EX", min_chars=10)
        assert "twenty-seven million customers" in sections["business"]

    def test_risk_factors_is_the_real_header_not_the_cross_reference(self):
        sections = extract_narrative(FORTYF_EXHIBIT, form_type="40-F/EX", min_chars=10)
        assert "credit risk, market risk, liquidity risk" in sections["risk_factors"]
        # The earlier cross-reference paragraph must not be what won the span.
        assert "forward-looking statements section" not in sections["risk_factors"]

    def test_mdna_extracted_from_a_separate_exhibit_document(self):
        # 40-F content is split across multiple exhibit FILES, not one document like
        # 10-K/20-F -- this fixture models the dedicated MD&A exhibit specifically.
        sections = extract_narrative(FORTYF_MDNA_EXHIBIT, form_type="40-F/EX", min_chars=10)
        assert "mdna" in sections
        assert "assess material changes in the financial condition" in sections["mdna"]

    def test_primary_40f_document_form_type_does_not_use_exhibit_patterns(self):
        # "40-F" (the primary, metadata-only document) is deliberately NOT a key in
        # PATTERNS_BY_FORM -- only "40-F/EX" is. Confirms it falls back to default
        # (10-K) patterns rather than picking up SECTION_PATTERNS_40F by accident.
        sections = extract_narrative(FORTYF_EXHIBIT, form_type="40-F", min_chars=10)
        assert "business" not in sections
        assert "risk_factors" not in sections


class TestExtractNarrativeFallback:
    def test_no_section_headers_falls_back_to_full_document(self):
        html = "<html><body><p>" + ("Earnings call commentary. " * 40) + "</p></body></html>"
        sections = extract_narrative(html, min_chars=10)
        assert list(sections) == ["full"]

    def test_below_min_chars_returns_nothing(self):
        assert extract_narrative("<p>too short</p>", min_chars=500) == {}

    def test_unknown_form_type_uses_default_patterns(self):
        sections = extract_narrative(TENK, form_type="8-K", min_chars=10)
        # 8-K has no item-1/1A/7 structure in practice, but if it did, an unregistered
        # form_type must not crash and must fall back to the default pattern set.
        assert "business" in sections
