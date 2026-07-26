from pathlib import Path


SUPPLEMENT_SOURCE = Path(__file__).resolve().parents[1] / "paper/latex" / "supplemental_material.tex"


def test_supplement_homepage_groups_later_tables_by_number_range():
    source = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")

    assert (
        r"Tables~\ref{tab:answer_overlap_definition}--\ref{tab:2wiki_hyde_query_mechanism} collect "
        "answer-overlap, gold-content, and 2Wiki query-composition diagnostics"
    ) in source
    assert (
        r"Table~\ref{tab:query_length_matched_hyde} reports query-length matched sensitivity"
    ) in source
    assert (
        r"Table~\ref{tab:single_annotator_error_taxonomy} gives the single-annotator error taxonomy"
    ) in source
    assert (
        r"Tables~\ref{tab:corpus_statistics}--\ref{tab:supporting_paragraph_audit} cover corpus statistics, "
        "supporting-title completeness, artifact availability, qualitative cases, and the supporting-paragraph audit"
    ) in source


def test_supplement_homepage_does_not_use_old_catchall_navigation_sentence():
    source = SUPPLEMENT_SOURCE.read_text(encoding="utf-8")

    assert "Later sections record prompt and API configuration, corpus-scale retrieval stress testing" not in source
    assert "After the first six lookup tables" not in source
