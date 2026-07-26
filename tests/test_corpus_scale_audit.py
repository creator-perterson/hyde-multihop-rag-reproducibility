import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_corpus_scale_audit import (
    audit_corpus_overlap,
    filter_added_gold_title_docs,
    select_hard_negative_indices,
    support_title_set,
)


def test_support_title_set_normalizes_and_deduplicates_titles():
    questions = [
        {"supporting_facts": {"title": ["Alpha", " alpha ", "Beta"]}},
        {"supporting_facts": {"title": "Gamma"}},
    ]

    assert support_title_set(questions) == {"alpha", "beta", "gamma"}


def test_audit_corpus_overlap_counts_gold_title_and_near_duplicate_added_docs():
    questions = [{"supporting_facts": {"title": ["Alpha"]}}]
    base_docs = [
        {
            "title": "Alpha",
            "text": "one two three four five six seven eight",
            "is_supporting": True,
        }
    ]
    expanded_docs = [
        base_docs[0],
        {
            "title": "Alpha",
            "text": "one two three four five six seven noise",
            "source_split": "train",
        },
        {
            "title": "Noise",
            "text": "completely different paragraph",
            "source_split": "train",
        },
    ]

    audit = audit_corpus_overlap(
        questions,
        base_docs,
        expanded_docs,
        near_duplicate_threshold=0.70,
    )

    assert audit["added_docs"] == 2
    assert audit["added_gold_title_docs"] == 1
    assert audit["near_duplicate_gold_paragraph_docs"] == 1
    assert audit["exact_gold_paragraph_duplicate_docs"] == 0


def test_filter_added_gold_title_docs_keeps_base_and_drops_only_added_overlaps():
    questions = [{"supporting_facts": {"title": ["Alpha"]}}]
    docs = [
        {"title": "Alpha", "text": "gold", "is_supporting": True},
        {"title": "Alpha", "text": "train same title", "source_split": "train"},
        {"title": "Beta", "text": "train other", "source_split": "train"},
    ]

    filtered = filter_added_gold_title_docs(docs, support_title_set(questions))

    assert [row["text"] for row in filtered] == ["gold", "train other"]


def test_select_hard_negative_indices_picks_highest_unique_scores():
    candidate_indices = [10, 11, 12, 13]
    max_scores = {10: 0.2, 11: 0.9, 12: 0.5, 13: 0.9}

    assert select_hard_negative_indices(candidate_indices, max_scores, 3) == [11, 13, 12]
