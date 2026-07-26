import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.retrieve_multiquery_topk import build_query_variants, fuse_multi_query_results


class RetrieveMultiqueryTopkTests(unittest.TestCase):
    def test_builds_distinct_query_variants(self):
        variants = build_query_variants(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            max_queries=3,
        )

        self.assertEqual(variants[0], "Were Scott Derrickson and Ed Wood of the same nationality?")
        self.assertEqual(len(variants), 3)
        self.assertEqual(len(set(variants)), 3)

    def test_fuses_multi_query_results_and_deduplicates_documents(self):
        docs = [
            {"doc_id": "d1", "title": "Original Hit", "text": "A", "source_question_id": "q1"},
            {"doc_id": "shared", "title": "Shared Hit", "text": "B", "source_question_id": "q1"},
            {"doc_id": "d2", "title": "Variant Hit", "text": "C", "source_question_id": "q1"},
        ]
        score_rows = [
            [0.90, 0.80],
            [0.95, 0.70],
        ]
        index_rows = [
            [0, 1],
            [1, 2],
        ]

        fused = fuse_multi_query_results(score_rows, index_rows, docs, top_k=3)

        self.assertEqual(fused[0]["doc_id"], "shared")
        self.assertEqual(len(fused), 3)
        self.assertEqual(len({doc["doc_id"] for doc in fused}), 3)
        self.assertEqual(fused[0]["matched_query_indices"], [0, 1])
        self.assertIn("rrf_score", fused[0])


if __name__ == "__main__":
    unittest.main()
