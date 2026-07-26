import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.build_equal_budget_query_prompts import (
    EQUAL_BUDGET_QUERY_MODES,
    build_equal_budget_prompt,
    build_prompt_row,
)


class EqualBudgetQueryPromptTests(unittest.TestCase):
    def test_all_modes_share_length_and_output_constraints(self):
        question = "Who directed the film whose writer was born in Paris?"

        prompts = {
            mode: build_equal_budget_prompt(question, mode=mode, target_words=40)
            for mode in EQUAL_BUDGET_QUERY_MODES
        }

        self.assertEqual(
            set(prompts),
            {"keyword_expansion", "direct_rewrite", "question_decomposition", "document_like_passage"},
        )
        for prompt in prompts.values():
            self.assertIn("Aim for 35-45 English words", prompt)
            self.assertIn("Do not answer the question", prompt)
            self.assertIn("Question:", prompt)
            self.assertIn(question, prompt)

    def test_mode_specific_prompt_names_generation_form(self):
        question = "Where was the author of Wise Blood born?"

        self.assertIn("keyword/entity list", build_equal_budget_prompt(question, "keyword_expansion"))
        self.assertIn("single dense-retrieval rewrite", build_equal_budget_prompt(question, "direct_rewrite"))
        self.assertIn("numbered subquestions", build_equal_budget_prompt(question, "question_decomposition"))
        self.assertIn("short encyclopedia-style passage", build_equal_budget_prompt(question, "document_like_passage"))

    def test_prompt_row_records_budget_metadata(self):
        question = {
            "id": "q1",
            "question": "Where was the author of Wise Blood born?",
            "answer": "Savannah, Georgia",
            "supporting_facts": {"title": ["Wise Blood", "Flannery O'Connor"]},
        }

        row = build_prompt_row(question, mode="document_like_passage", target_words=40)

        self.assertEqual(row["id"], "q1")
        self.assertEqual(row["gold_answer"], "Savannah, Georgia")
        self.assertEqual(row["equal_budget_query_mode"], "document_like_passage")
        self.assertEqual(row["budget"]["target_words"], 40)
        self.assertEqual(row["budget"]["max_tokens"], 64)
        self.assertEqual(row["retrieved"], [])


if __name__ == "__main__":
    unittest.main()
