import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_corpus_scale_retrieval_stress import (
    bootstrap_delta_ci,
    build_expanded_corpus,
    context_to_document,
    document_key,
    paired_delta_summary,
    per_example_retrieval_metrics,
    retrieval_metrics,
)


class CorpusScaleRetrievalStressTests(unittest.TestCase):
    def test_document_key_normalizes_title_and_text(self):
        doc = {"title": "  The   Title ", "text": "Same\n text"}

        self.assertEqual(document_key(doc), ("the title", "same text"))

    def test_context_to_document_records_processed_source(self):
        row = {"question_id": "q1", "contexts": [{"idx": 3, "title": "A", "paragraph_text": "Text"}]}

        doc = context_to_document(row, row["contexts"][0], source_split="train")

        self.assertEqual(doc["doc_id"], "train:q1:3")
        self.assertEqual(doc["title"], "A")
        self.assertEqual(doc["text"], "Text")
        self.assertEqual(doc["source_question_id"], "q1")
        self.assertEqual(doc["source_split"], "train")

    def test_build_expanded_corpus_preserves_base_and_adds_unique_distractors(self):
        base_docs = [
            {"doc_id": "base-a", "title": "A", "text": "base", "source_question_id": "t1"},
            {"doc_id": "base-b", "title": "B", "text": "base", "source_question_id": "t2"},
        ]
        source_rows = [
            {
                "question_id": "train1",
                "contexts": [
                    {"idx": 0, "title": "A", "paragraph_text": "base"},
                    {"idx": 1, "title": "C", "paragraph_text": "new"},
                ],
            },
            {
                "question_id": "train2",
                "contexts": [
                    {"idx": 0, "title": "D", "paragraph_text": "new"},
                ],
            },
        ]

        docs, stats = build_expanded_corpus(
            base_docs=base_docs,
            source_rows=source_rows,
            target_size=4,
            source_split="train",
        )

        self.assertEqual([doc["doc_id"] for doc in docs], ["base-a", "base-b", "train:train1:1", "train:train2:0"])
        self.assertEqual(stats["base_docs"], 2)
        self.assertEqual(stats["expanded_docs"], 4)
        self.assertEqual(stats["added_distractors"], 2)
        self.assertEqual(stats["skipped_duplicates"], 1)

    def test_retrieval_metrics_report_any_all_and_recall(self):
        rows = [
            {
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}, {"title": "C"}],
            },
            {
                "supporting_facts": {"title": ["D"]},
                "retrieved": [{"title": "D"}],
            },
        ]

        metrics = retrieval_metrics(rows)

        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["any_hit@10"], 1.0)
        self.assertAlmostEqual(metrics["all_support_hit@10"], 0.5)
        self.assertAlmostEqual(metrics["supporting_title_recall@10"], 0.75)

    def test_per_example_metrics_support_paired_deltas(self):
        baseline = per_example_retrieval_metrics(
            [
                {"id": "q1", "supporting_facts": {"title": ["A", "B"]}, "retrieved": [{"title": "A"}]},
                {"id": "q2", "supporting_facts": {"title": ["C"]}, "retrieved": []},
            ]
        )
        target = per_example_retrieval_metrics(
            [
                {
                    "id": "q1",
                    "supporting_facts": {"title": ["A", "B"]},
                    "retrieved": [{"title": "A"}, {"title": "B"}],
                },
                {"id": "q2", "supporting_facts": {"title": ["C"]}, "retrieved": [{"title": "C"}]},
            ]
        )

        stats = paired_delta_summary(
            baseline,
            target,
            "all_support_hit@10",
            iterations=50,
            seed=7,
        )

        self.assertAlmostEqual(stats["all_support_hit@10_delta"], 1.0)
        self.assertEqual(stats["all_support_hit@10_ci_low"], 1.0)
        self.assertEqual(stats["all_support_hit@10_ci_high"], 1.0)

    def test_bootstrap_delta_ci_is_deterministic_for_constant_values(self):
        self.assertEqual(bootstrap_delta_ci([0.25, 0.25, 0.25], iterations=25, seed=3), (0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
