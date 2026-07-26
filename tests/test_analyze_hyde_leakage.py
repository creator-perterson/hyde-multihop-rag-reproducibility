import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_hyde_leakage import (
    contains_normalized_answer,
    is_short_or_ambiguous_answer,
    normalize_text,
)


class AnalyzeHydeLeakageTests(unittest.TestCase):
    def test_normalizes_articles_and_punctuation(self):
        self.assertEqual(normalize_text("The Picture of Dorian Gray!"), "picture of dorian gray")

    def test_detects_gold_answer_inside_hypothetical_document(self):
        self.assertTrue(
            contains_normalized_answer(
                "The Picture of Dorian Gray",
                "Oscar Wilde wrote The Picture of Dorian Gray in 1890.",
            )
        )

    def test_marks_short_answers_as_ambiguous(self):
        self.assertTrue(is_short_or_ambiguous_answer("no"))
        self.assertTrue(is_short_or_ambiguous_answer("37"))
        self.assertFalse(is_short_or_ambiguous_answer("Wembley Stadium"))


if __name__ == "__main__":
    unittest.main()
