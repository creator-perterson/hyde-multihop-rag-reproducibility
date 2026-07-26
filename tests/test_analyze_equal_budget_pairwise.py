import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_equal_budget_pairwise import (
    analyze_pairwise_comparisons,
    per_example_retrieval_metrics,
)


def retrieval_row(item_id, support_titles, retrieved_titles):
    return {
        "id": item_id,
        "supporting_facts": {"title": support_titles},
        "retrieved": [{"title": title} for title in retrieved_titles],
    }


class EqualBudgetPairwiseTests(unittest.TestCase):
    def test_per_example_metrics_compute_complete_support_and_recall(self):
        rows = [
            retrieval_row("one", ["A", "B"], ["A"]),
            retrieval_row("two", ["C"], ["C"]),
        ]

        metrics = per_example_retrieval_metrics(rows)

        self.assertEqual(metrics["one"]["all_support_hit@10"], 0.0)
        self.assertEqual(metrics["one"]["supporting_title_recall@10"], 0.5)
        self.assertEqual(metrics["two"]["all_support_hit@10"], 1.0)

    def test_document_like_reports_paired_delta_and_ci_for_each_baseline(self):
        by_mode = {
            "direct_rewrite": per_example_retrieval_metrics(
                [
                    retrieval_row("one", ["A", "B"], ["A"]),
                    retrieval_row("two", ["C"], []),
                ]
            ),
            "keyword_expansion": per_example_retrieval_metrics(
                [
                    retrieval_row("one", ["A", "B"], ["A", "B"]),
                    retrieval_row("two", ["C"], []),
                ]
            ),
            "document_like_passage": per_example_retrieval_metrics(
                [
                    retrieval_row("one", ["A", "B"], ["A", "B"]),
                    retrieval_row("two", ["C"], ["C"]),
                ]
            ),
        }

        rows = analyze_pairwise_comparisons(by_mode, iterations=200, seed=13)

        direct = next(row for row in rows if row["baseline_query_mode"] == "direct_rewrite")
        keyword = next(row for row in rows if row["baseline_query_mode"] == "keyword_expansion")
        self.assertEqual(direct["n"], 2)
        self.assertEqual(direct["all_support_hit@10_delta"], 1.0)
        self.assertEqual(direct["supporting_title_recall@10_delta"], 0.75)
        self.assertEqual(keyword["all_support_hit@10_delta"], 0.5)
        self.assertLessEqual(keyword["all_support_hit@10_ci_low"], 0.5)
        self.assertGreaterEqual(keyword["all_support_hit@10_ci_high"], 0.5)

    def test_pairwise_analysis_rejects_misaligned_question_ids(self):
        by_mode = {
            "direct_rewrite": per_example_retrieval_metrics([retrieval_row("one", ["A"], ["A"])]),
            "document_like_passage": per_example_retrieval_metrics([retrieval_row("two", ["A"], ["A"])]),
        }

        with self.assertRaisesRegex(RuntimeError, "identical question ids"):
            analyze_pairwise_comparisons(by_mode, iterations=10)


if __name__ == "__main__":
    unittest.main()
