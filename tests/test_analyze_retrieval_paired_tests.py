import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_retrieval_paired_tests import (
    paired_retrieval_summary,
    support_metrics,
)


class RetrievalPairedTests(unittest.TestCase):
    def test_support_metrics_scores_complete_partial_and_empty_support(self):
        partial = {
            "id": "q1",
            "supporting_facts": {"title": ["A", "B"]},
            "retrieved": [{"title": "A"}, {"title": "C"}],
        }
        empty_gold = {
            "id": "q2",
            "supporting_facts": {"title": []},
            "retrieved": [{"title": "A"}],
        }

        self.assertEqual(
            support_metrics(partial),
            {"any_hit": 1.0, "all_hit": 0.0, "support_recall": 0.5},
        )
        self.assertEqual(
            support_metrics(empty_gold),
            {"any_hit": 0.0, "all_hit": 0.0, "support_recall": 0.0},
        )

    def test_paired_retrieval_summary_reports_deltas_ci_and_full_support_transitions(self):
        baseline = {
            "q1": {
                "id": "q1",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}],
            },
            "q2": {
                "id": "q2",
                "supporting_facts": {"title": ["C"]},
                "retrieved": [{"title": "C"}],
            },
            "q3": {
                "id": "q3",
                "supporting_facts": {"title": ["D", "E"]},
                "retrieved": [{"title": "X"}],
            },
        }
        target = {
            "q1": {
                "id": "q1",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}, {"title": "B"}],
            },
            "q2": {
                "id": "q2",
                "supporting_facts": {"title": ["C"]},
                "retrieved": [{"title": "Z"}],
            },
            "q3": {
                "id": "q3",
                "supporting_facts": {"title": ["D", "E"]},
                "retrieved": [{"title": "D"}],
            },
        }

        row = paired_retrieval_summary(
            dataset="Toy",
            baseline_label="Dense",
            target_label="HyDE",
            baseline_by_id=baseline,
            target_by_id=target,
            iterations=20,
            seed=7,
        )

        self.assertEqual(row["dataset"], "Toy")
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["baseline_all_hit"], 1 / 3)
        self.assertAlmostEqual(row["target_all_hit"], 1 / 3)
        self.assertAlmostEqual(row["delta_all_hit"], 0.0)
        self.assertAlmostEqual(row["baseline_support_recall"], 0.5)
        self.assertAlmostEqual(row["target_support_recall"], 0.5)
        self.assertEqual(row["target_only_full"], 1)
        self.assertEqual(row["baseline_only_full"], 1)
        self.assertIn("delta_all_hit_ci_low", row)
        self.assertIn("delta_support_recall_ci_high", row)


if __name__ == "__main__":
    unittest.main()
