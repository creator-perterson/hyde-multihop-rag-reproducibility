from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1] / "paper/latex"


def test_related_work_merges_verification_into_agentic_context():
    source = (PAPER_ROOT / "related_work_draft.tex").read_text(encoding="utf-8")

    assert r"\subsection{Verification and Self-refinement}" not in source
    assert r"\label{sec:rw_verification_eval}" not in source
    assert "Post-retrieval evaluators and self-checking methods are adjacent to this adaptive-RAG literature" in source
    assert "not a separate method family in our main claim" in source


def test_figure1_keeps_verifier_branch_visually_secondary():
    source = (PAPER_ROOT / "figure1_method_pipeline_tikz.tex").read_text(encoding="utf-8")

    assert r"\draw[arrow, dashed, draw=black!35] (ans.south) -- (diag.north);" in source
    assert "Exploratory guarded" in source
    assert "small exploratory post-processing diagnostic" in source
