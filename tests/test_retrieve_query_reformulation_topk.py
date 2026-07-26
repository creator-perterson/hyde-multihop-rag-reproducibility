import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.retrieve_query_reformulation_topk import build_reformulated_query_text


class RetrieveQueryReformulationTopkTests(unittest.TestCase):
    def test_uses_prediction_as_dense_retrieval_query(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "novel author",
        }

        query = build_reformulated_query_text(row)

        self.assertEqual(query, "novel author")

    def test_falls_back_to_question_when_prediction_is_blank(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "   ",
        }

        query = build_reformulated_query_text(row)

        self.assertEqual(query, "Who wrote the novel?")

    def test_truncates_long_reformulated_query(self):
        row = {
            "question": "Q?",
            "prediction": "a" * 20,
        }

        query = build_reformulated_query_text(row, max_query_chars=10)

        self.assertEqual(query, "aaaaaaaaaa...")

    def test_can_combine_original_question_with_reformulated_query(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "novel author",
        }

        query = build_reformulated_query_text(
            row,
            query_mode="question_plus_reformulation",
        )

        self.assertIn("Who wrote the novel?", query)
        self.assertIn("Rewritten retrieval query:", query)
        self.assertIn("novel author", query)


if __name__ == "__main__":
    unittest.main()
