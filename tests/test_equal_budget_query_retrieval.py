import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.run_equal_budget_query_retrieval import (
    build_equal_budget_query_text,
    query_word_count,
    summarize_generation_lengths,
)


class EqualBudgetQueryRetrievalTests(unittest.TestCase):
    def test_all_modes_use_identical_query_serialization(self):
        row = {"question": "Who directed the film?", "prediction": "film director production"}

        query = build_equal_budget_query_text(row)

        self.assertEqual(
            query,
            "Who directed the film?\n\nGenerated retrieval text:\nfilm director production",
        )

    def test_query_word_count_counts_words_and_numbers(self):
        self.assertEqual(query_word_count("Wise Blood, 1979 film; John Huston."), 6)

    def test_summarize_generation_lengths_by_mode(self):
        rows = [
            {"equal_budget_query_mode": "keyword_expansion", "prediction": "a b c"},
            {"equal_budget_query_mode": "keyword_expansion", "prediction": "a b c d e"},
            {"equal_budget_query_mode": "direct_rewrite", "prediction": "a b"},
        ]

        summary = summarize_generation_lengths(rows)

        self.assertEqual(summary["keyword_expansion"]["n"], 2)
        self.assertEqual(summary["keyword_expansion"]["mean_words"], 4.0)
        self.assertEqual(summary["keyword_expansion"]["min_words"], 3)
        self.assertEqual(summary["keyword_expansion"]["max_words"], 5)
        self.assertEqual(summary["direct_rewrite"]["mean_words"], 2.0)

    def test_summarize_generation_lengths_uses_default_mode(self):
        rows = [
            {"prediction": "a b c"},
            {"prediction": "a b c d"},
        ]

        summary = summarize_generation_lengths(rows, default_mode="document_like_passage")

        self.assertEqual(summary["document_like_passage"]["n"], 2)
        self.assertEqual(summary["document_like_passage"]["mean_words"], 3.5)


if __name__ == "__main__":
    unittest.main()
