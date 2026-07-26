import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_topk_sensitivity import (
    DEFAULT_CONFIGS,
    first_support_ranks,
    summarize_rows,
    write_latex,
)


def test_first_support_ranks_use_earliest_title_match():
    row = {
        "supporting_facts": {"title": ["Alpha", "Beta"]},
        "retrieved": [
            {"title": "Noise"},
            {"title": "Beta"},
            {"title": "Noise 2"},
            {"title": "Alpha"},
            {"title": "Beta"},
        ],
    }

    assert first_support_ranks(row) == {"alpha": 4, "beta": 2}


def test_summarize_rows_computes_topk_and_capped_worst_rank():
    rows = [
        {
            "supporting_facts": {"title": ["Alpha", "Beta"]},
            "retrieved": [
                {"title": "Alpha"},
                {"title": "Noise"},
                {"title": "Noise 2"},
                {"title": "Beta"},
            ],
        },
        {
            "supporting_facts": {"title": ["Gamma", "Delta"]},
            "retrieved": [
                {"title": "Gamma"},
                {"title": "Noise"},
            ],
        },
    ]

    summary = summarize_rows(rows, ks=(1, 3), rank_depth=3)

    assert summary["n"] == 2
    assert summary["all_hit_at1"] == 0.0
    assert summary["recall_at1"] == 0.5
    assert summary["all_hit_at3"] == 0.0
    assert summary["recall_at3"] == 0.5
    assert summary["mean_worst_support_rank_capped"] == 4.0
    assert summary["median_worst_support_rank_capped"] == 4.0
    assert summary["missing_worst_rank_count"] == 2


def test_default_configs_exclude_depth_sensitive_hybrid_fusion():
    methods = [config["method"] for config in DEFAULT_CONFIGS]

    assert "Hybrid" not in methods


def test_latex_table_excludes_hybrid_rows(tmp_path):
    rows = [
        {
            "dataset": "HotpotQA",
            "method": "Dense",
            "all_hit_at5": 0.1,
            "all_hit_at10": 0.2,
            "all_hit_at20": 0.3,
            "recall_at5": 0.4,
            "recall_at10": 0.5,
            "recall_at20": 0.6,
            "mean_worst_support_rank_capped": 7.0,
            "median_worst_support_rank_capped": 8.0,
            "missing_worst_rank_count": 9,
        },
        {
            "dataset": "HotpotQA",
            "method": "Hybrid",
            "all_hit_at5": 0.1,
            "all_hit_at10": 0.2,
            "all_hit_at20": 0.3,
            "recall_at5": 0.4,
            "recall_at10": 0.5,
            "recall_at20": 0.6,
            "mean_worst_support_rank_capped": 7.0,
            "median_worst_support_rank_capped": 8.0,
            "missing_worst_rank_count": 9,
        },
    ]

    path = tmp_path / "table.tex"
    write_latex(path, rows, ks=(5, 10, 20))

    text = path.read_text(encoding="utf-8")
    assert "Dense" in text
    assert "Hybrid" not in text
