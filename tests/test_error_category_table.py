import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.build_error_analysis_table import classify_four_way_error


def make_row(prediction, gold="London", gold_titles=None, retrieved_titles=None):
    gold_titles = gold_titles or ["A", "B"]
    retrieved_titles = retrieved_titles or []
    return {
        "id": "q1",
        "question": "Q?",
        "gold_answer": gold,
        "prediction": prediction,
        "supporting_facts": {"title": gold_titles},
        "retrieved": [{"title": title, "text": f"{title} evidence"} for title in retrieved_titles],
    }


class ErrorCategoryTableTests(unittest.TestCase):
    def test_classifies_retrieval_miss_when_no_supporting_title_is_hit(self):
        row = make_row("Paris", retrieved_titles=["C", "D"])
        self.assertEqual(classify_four_way_error(row)["error_category"], "retrieval miss")

    def test_classifies_partial_evidence_missing_hop(self):
        row = make_row("Paris", retrieved_titles=["A", "C"])
        self.assertEqual(classify_four_way_error(row)["error_category"], "partial evidence / missing hop")

    def test_classifies_reader_reasoning_error_when_full_evidence_is_present(self):
        row = make_row("Paris", retrieved_titles=["A", "B"])
        self.assertEqual(classify_four_way_error(row)["error_category"], "reader reasoning error")

    def test_classifies_answer_format_alias_before_reader_reasoning(self):
        row = make_row(
            "United States",
            gold="American",
            gold_titles=["A"],
            retrieved_titles=["A"],
        )
        self.assertEqual(classify_four_way_error(row)["error_category"], "answer format / alias error")

    def test_classifies_numeric_word_format_error(self):
        row = make_row("fourth", gold="4", gold_titles=["A", "B"], retrieved_titles=["A", "B"])
        self.assertEqual(classify_four_way_error(row)["error_category"], "answer format / alias error")

    def test_classifies_correct_answer(self):
        row = make_row("London", retrieved_titles=["A", "B"])
        self.assertEqual(classify_four_way_error(row)["error_category"], "correct")


if __name__ == "__main__":
    unittest.main()
