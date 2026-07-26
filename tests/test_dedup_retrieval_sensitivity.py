import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_dedup_retrieval_sensitivity import (
    deduplicate_documents,
    duplicate_slot_stats,
    normalize_dedup_text,
    retrieval_metrics,
)


class DedupRetrievalSensitivityTests(unittest.TestCase):
    def test_normalizes_title_paragraph_key_for_exact_deduplication(self):
        self.assertEqual(
            normalize_dedup_text("  The   Same\nTitle "),
            "the same title",
        )

    def test_deduplicate_documents_keeps_first_title_text_record(self):
        docs = [
            {"doc_id": "a", "title": "Title", "text": "Same text", "source_question_id": "q1"},
            {"doc_id": "b", "title": " title ", "text": "Same   text", "source_question_id": "q2"},
            {"doc_id": "c", "title": "Other", "text": "Same text", "source_question_id": "q3"},
        ]

        deduped, stats = deduplicate_documents(docs)

        self.assertEqual([doc["doc_id"] for doc in deduped], ["a", "c"])
        self.assertEqual(stats["input_docs"], 3)
        self.assertEqual(stats["dedup_docs"], 2)
        self.assertEqual(stats["removed_exact_duplicates"], 1)
        self.assertEqual(deduped[0]["duplicate_count"], 2)
        self.assertEqual(deduped[0]["duplicate_doc_ids"], ["a", "b"])

    def test_retrieval_metrics_collapse_duplicate_retrieved_titles(self):
        rows = [
            {
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}, {"title": "A"}, {"title": "C"}],
            },
            {
                "supporting_facts": {"title": ["D"]},
                "retrieved": [{"title": "D"}, {"title": "D"}],
            },
        ]

        metrics = retrieval_metrics(rows)

        self.assertEqual(metrics["n"], 2)
        self.assertAlmostEqual(metrics["any_hit"], 1.0)
        self.assertAlmostEqual(metrics["all_hit"], 0.5)
        self.assertAlmostEqual(metrics["support_recall"], 0.75)

    def test_duplicate_slot_stats_reports_title_and_exact_duplicate_slots(self):
        row = {
            "retrieved": [
                {"title": "A", "text": "same"},
                {"title": "A", "text": "same"},
                {"title": "A", "text": "different"},
                {"title": "B", "text": "other"},
            ]
        }

        stats = duplicate_slot_stats(row)

        self.assertEqual(stats["duplicate_title_slots"], 2)
        self.assertEqual(stats["duplicate_exact_slots"], 1)


if __name__ == "__main__":
    unittest.main()
