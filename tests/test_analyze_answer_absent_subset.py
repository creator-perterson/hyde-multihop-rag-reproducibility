import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_answer_absent_subset import (
    answer_absent_ids,
    paired_subset_summary,
)


class AnswerAbsentSubsetTests(unittest.TestCase):
    def test_answer_absent_ids_filters_leakage_rows(self):
        leakage_by_id = {
            "q1": {"id": "q1", "answer_in_hyde": 0},
            "q2": {"id": "q2", "answer_in_hyde": 1},
            "q3": {"id": "q3", "answer_in_hyde": "0"},
        }

        self.assertEqual(answer_absent_ids(leakage_by_id), ["q1", "q3"])

    def test_paired_subset_summary_reports_ci_and_transitions(self):
        leakage_by_id = {
            "q1": {"id": "q1", "answer_in_hyde": 0},
            "q2": {"id": "q2", "answer_in_hyde": 0},
        }
        dense_retrieval = {
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
        }
        hyde_retrieval = {
            "q1": {
                "id": "q1",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}, {"title": "B"}],
            },
            "q2": {
                "id": "q2",
                "supporting_facts": {"title": ["C"]},
                "retrieved": [{"title": "D"}],
            },
        }
        dense_answers = {
            "q1": {"id": "q1", "gold_answer": "Paris", "prediction": "London"},
            "q2": {"id": "q2", "gold_answer": "yes", "prediction": "yes"},
        }
        hyde_answers = {
            "q1": {"id": "q1", "gold_answer": "Paris", "prediction": "Paris"},
            "q2": {"id": "q2", "gold_answer": "yes", "prediction": "no"},
        }

        row = paired_subset_summary(
            dataset="Toy",
            subset="answer_not_in_hyde",
            ids=["q1", "q2"],
            baseline_label="Dense",
            target_label="HyDE",
            baseline_retrieval_by_id=dense_retrieval,
            target_retrieval_by_id=hyde_retrieval,
            baseline_answers_by_id=dense_answers,
            target_answers_by_id=hyde_answers,
            iterations=20,
            seed=5,
        )

        self.assertEqual(row["dataset"], "Toy")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["baseline_all_hit"], 0.5)
        self.assertAlmostEqual(row["target_all_hit"], 0.5)
        self.assertAlmostEqual(row["delta_em"], 0.0)
        self.assertAlmostEqual(row["delta_f1"], 0.0)
        self.assertEqual(row["wrong_to_correct"], 1)
        self.assertEqual(row["correct_to_wrong"], 1)
        self.assertIn("delta_f1_ci_low", row)
        self.assertIn("delta_f1_ci_high", row)


if __name__ == "__main__":
    unittest.main()
