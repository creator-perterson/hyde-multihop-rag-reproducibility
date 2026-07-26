from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1] / "paper/latex"
CODE_ROOT = Path(__file__).resolve().parents[1]


def test_supplement_s8_describes_local_50k_100k_protocol():
    source = (PAPER_ROOT / "supplemental_material.tex").read_text(encoding="utf-8")

    assert "we compare the original retained-duplicate local index with expanded 50k and 100k stress-test indexes" in source
    assert "we build corpus-scale stress-test indexes at 5k, 50k, and 100k" not in source


def test_supplement_corpus_scale_table_uses_local_not_formal_5k_rows():
    source = (PAPER_ROOT / "table_corpus_scale_stress_full.tex").read_text(encoding="utf-8")

    assert "HotpotQA & Local & Question only & 0.9940 & 0.8820 & 0.9380" in source
    assert "2Wiki & Local & Question only & 0.9980 & 0.5100 & 0.7655" in source
    assert " & 5k & " not in source


def test_main_corpus_scale_table_disambiguates_single_rewrite_column():
    source = (PAPER_ROOT / "table_corpus_scale_stress.tex").read_text(encoding="utf-8")

    assert r"\multicolumn{2}{c}{Single rewrite}" in source
    assert r"\multicolumn{2}{c}{Rewrite}" not in source
    assert "Single rewrite denotes the Single rewritten query condition from Table~\\ref{tab:stronger_encoder_reproduction}" in source
    assert "not the question-plus-rewritten-query composition or the equal-budget direct-rewrite diagnostic" in source


def test_stress_script_defaults_to_expanded_50k_and_100k_only():
    source = (
        CODE_ROOT / "src" / "evaluation" / "analyze_corpus_scale_retrieval_stress.py"
    ).read_text(encoding="utf-8")

    assert "DEFAULT_CORPUS_SIZES = [50000, 100000]" in source
    assert "default=[5000, 50000, 100000]" not in source
