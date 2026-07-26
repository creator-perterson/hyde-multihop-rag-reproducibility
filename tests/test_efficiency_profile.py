import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_efficiency_profile import (
    MethodSpec,
    evidence_chars_from_prompt,
    prompt_stats,
    summarize_method,
)


class EfficiencyProfileTests(unittest.TestCase):
    def test_evidence_chars_from_prompt_extracts_evidence_block(self):
        prompt = "Question:\nQ?\n\nEvidence:\n[1] Title: A\nAlpha\n\nExact short answer:"

        self.assertEqual(evidence_chars_from_prompt(prompt), len("[1] Title: A\nAlpha"))

    def test_prompt_stats_counts_rows_chars_evidence_and_retrieved_docs(self):
        rows = [
            {"prompt": "Question:\nQ?\n\nEvidence:\nabc\n\nExact short answer:", "retrieved": [{"title": "A"}]},
            {"prompt": "Question:\nQ2?\n\nEvidence:\ndefgh\n\nExact short answer:", "retrieved": [{"title": "A"}, {"title": "B"}]},
        ]

        stats = prompt_stats(rows)

        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg_prompt_chars"], 49.5)
        self.assertAlmostEqual(stats["avg_evidence_chars"], 4.0)
        self.assertAlmostEqual(stats["avg_retrieved_docs"], 1.5)

    def test_summarize_method_counts_hyde_and_selective_verifier_calls(self):
        spec = MethodSpec(
            method="HyDE + verifier",
            reader_prompt_rows=[{"prompt": "Evidence:\na\n\nExact short answer:", "retrieved": [{}]}] * 2,
            hyde_prompt_rows=[{"prompt": "hyde"}] * 2,
            verifier_prompt_rows=[{"prompt": "Evidence:\nverify\n\nJSON:", "retrieved": [{}]}],
            answer_rows=[{"prediction": "A"}, {"prediction": "B"}],
            training_required="No",
        )

        row = summarize_method(spec)

        self.assertEqual(row["n_examples"], 2)
        self.assertEqual(row["reader_calls"], 2)
        self.assertEqual(row["hyde_generation_calls"], 2)
        self.assertEqual(row["verifier_calls"], 1)
        self.assertEqual(row["total_llm_calls"], 5)
        self.assertAlmostEqual(row["calls_per_question"], 2.5)
        self.assertAlmostEqual(row["verifier_coverage"], 0.5)
        self.assertAlmostEqual(row["avg_hyde_prompt_chars"], 4.0)
        self.assertEqual(row["training_required"], "No")


if __name__ == "__main__":
    unittest.main()
