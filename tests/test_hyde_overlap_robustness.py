import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_hyde_overlap_robustness import (
    paired_group_summary,
    support_metrics,
)


class HydeOverlapRobustnessTests(unittest.TestCase):
    def test_support_metrics_counts_complete_support(self):
        row = {
            "supporting_facts": {"title": ["A", "B"]},
            "retrieved": [{"title": "A"}, {"title": "C"}, {"title": "B"}],
        }

        metrics = support_metrics(row)

        self.assertEqual(metrics["any_hit"], 1.0)
        self.assertEqual(metrics["all_hit"], 1.0)
        self.assertEqual(metrics["support_recall"], 1.0)

    def test_paired_group_summary_reports_delta_and_transitions(self):
        baseline = {
            "q1": {
                "id": "q1",
                "gold_answer": "Paris",
                "prediction": "London",
                "supporting_facts": {"title": ["A"]},
                "retrieved": [{"title": "A"}],
            },
            "q2": {
                "id": "q2",
                "gold_answer": "yes",
                "prediction": "yes",
                "supporting_facts": {"title": ["B"]},
                "retrieved": [{"title": "B"}],
            },
        }
        target = {
            "q1": {
                "id": "q1",
                "gold_answer": "Paris",
                "prediction": "Paris",
                "supporting_facts": {"title": ["A"]},
                "retrieved": [{"title": "A"}],
            },
            "q2": {
                "id": "q2",
                "gold_answer": "yes",
                "prediction": "no",
                "supporting_facts": {"title": ["B"]},
                "retrieved": [{"title": "B"}],
            },
        }

        summary = paired_group_summary(
            "answer_not_in_hyde",
            ["q1", "q2"],
            baseline,
            target,
            baseline_label="Dense",
            target_label="HyDE",
        )

        self.assertEqual(summary["group"], "answer_not_in_hyde")
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["delta_em"], 0.0)
        self.assertAlmostEqual(summary["delta_f1"], 0.0)
        self.assertEqual(summary["wrong_to_correct"], 1)
        self.assertEqual(summary["correct_to_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
