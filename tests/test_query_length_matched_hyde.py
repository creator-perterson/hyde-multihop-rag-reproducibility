import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_query_length_matched_hyde import (
    HYDE_SEPARATOR,
    build_length_matched_queries,
    build_query_length_row,
    token_count,
    truncate_hyde_to_serialized_query_budget,
    truncate_to_token_count,
)


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class QueryLengthMatchedHydeTests(unittest.TestCase):
    def test_truncate_to_token_count_uses_prefix_tokens(self):
        tokenizer = FakeTokenizer()

        truncated = truncate_to_token_count("alpha beta gamma delta", 2, tokenizer)

        self.assertEqual(truncated, "alpha beta")

    def test_truncate_to_token_count_keeps_empty_budget_empty(self):
        tokenizer = FakeTokenizer()

        truncated = truncate_to_token_count("alpha beta", 0, tokenizer)

        self.assertEqual(truncated, "")

    def test_build_query_length_row_reports_serialized_query_lengths(self):
        tokenizer = FakeTokenizer()
        row = {
            "id": "q1",
            "question": "who wrote it",
            "rewrite": "author name",
            "hyde": "author name wrote book",
        }

        stats = build_query_length_row(row, tokenizer, max_seq_length=5)

        self.assertEqual(stats["q_tokens"], 3)
        self.assertEqual(stats["r_tokens"], 2)
        self.assertEqual(stats["q_plus_r_tokens"], 8)
        self.assertEqual(stats["h_tokens"], 4)
        self.assertEqual(stats["q_plus_h_tokens"], 10)
        self.assertEqual(stats["q_plus_h_hits_cap"], 1)

    def test_question_plus_hyde_length_match_preserves_question(self):
        tokenizer = FakeTokenizer()
        row = {
            "id": "q1",
            "question": "who wrote it",
            "rewrite": "author name",
            "hyde": "author name wrote book in city",
        }

        queries, details = build_length_matched_queries(
            [row],
            tokenizer,
            mode="question_plus_hypothetical",
        )

        self.assertTrue(queries[0].startswith("who wrote it"))
        self.assertIn(HYDE_SEPARATOR.strip(), queries[0])
        self.assertEqual(details[0]["question_preserved"], 1)
        self.assertLessEqual(details[0]["matched_query_tokens"], details[0]["rewrite_query_tokens"])

    def test_truncate_hyde_to_serialized_query_budget_matches_total_budget(self):
        tokenizer = FakeTokenizer()
        question = "who wrote it"
        hyde = "alpha beta gamma delta"
        target_tokens = token_count(f"{question}\n\nRewritten retrieval query:\nalpha beta", tokenizer)

        matched = truncate_hyde_to_serialized_query_budget(question, hyde, target_tokens, tokenizer)
        serialized = f"{question}{HYDE_SEPARATOR}{matched}".strip()

        self.assertEqual(matched, "alpha beta")
        self.assertLessEqual(token_count(serialized, tokenizer), target_tokens)


if __name__ == "__main__":
    unittest.main()
