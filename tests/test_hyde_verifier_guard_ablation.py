import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_hyde_verifier_guard_ablation import (
    apply_guard_variant,
    summarize_variant,
)


class HydeVerifierGuardAblationTests(unittest.TestCase):
    def test_apply_guard_variant_blocks_abstention_only_when_enabled(self):
        verifier_text = """{"verdict":"abstain","final_answer":"I don't know","reason":"insufficient"}"""

        self.assertEqual(
            apply_guard_variant("Nassau County", verifier_text, "Q?", "raw"),
            "I don't know",
        )
        self.assertEqual(
            apply_guard_variant("Nassau County", verifier_text, "Q?", "no_abstention_guard"),
            "Nassau County",
        )

    def test_apply_guard_variant_blocks_numeric_unit_only_when_enabled(self):
        verifier_text = '{"verdict":"correct","final_answer":"94 episodes","reason":"added unit"}'

        self.assertEqual(apply_guard_variant("94", verifier_text, "How many?", "raw"), "94 episodes")
        self.assertEqual(
            apply_guard_variant("94", verifier_text, "How many?", "numeric_unit_guard"),
            "94",
        )

    def test_apply_guard_variant_keeps_initial_on_malformed_json(self):
        self.assertEqual(
            apply_guard_variant("Nassau County", "not valid json", "Q?", "raw"),
            "Nassau County",
        )

    def test_summarize_variant_counts_verified_subset_transitions(self):
        base_rows = {
            "q1": {"id": "q1", "question": "Nationality?", "gold_answer": "American", "prediction": "United States"},
            "q2": {"id": "q2", "question": "County?", "gold_answer": "Nassau County", "prediction": "Nassau County"},
            "q3": {"id": "q3", "question": "How many episodes?", "gold_answer": "94", "prediction": "94"},
        }
        verifier_rows = {
            "q1": {"id": "q1", "prediction": '{"verdict":"correct","final_answer":"American","reason":"format"}'},
            "q2": {"id": "q2", "prediction": """{"verdict":"abstain","final_answer":"I don't know","reason":"unclear"}"""},
            "q3": {"id": "q3", "prediction": '{"verdict":"correct","final_answer":"94 episodes","reason":"unit"}'},
        }

        raw = summarize_variant("raw", base_rows, verifier_rows)
        both = summarize_variant("both_guards", base_rows, verifier_rows)

        self.assertEqual(raw["verified_n"], 3)
        self.assertEqual(raw["wrong_to_correct"], 1)
        self.assertEqual(raw["correct_to_wrong"], 2)
        self.assertEqual(both["wrong_to_correct"], 1)
        self.assertEqual(both["correct_to_wrong"], 0)
        self.assertGreater(both["final_em"], raw["final_em"])


if __name__ == "__main__":
    unittest.main()
