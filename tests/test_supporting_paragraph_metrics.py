import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_supporting_paragraph_metrics import (
    build_support_index,
    paragraph_key,
    retrieval_metrics,
    write_latex,
)


class SupportingParagraphMetricsTests(unittest.TestCase):
    def test_paragraph_metrics_are_stricter_than_title_metrics(self):
        support_index = build_support_index(
            [
                {
                    "doc_id": "q1::0::A",
                    "source_question_id": "q1",
                    "title": "A",
                    "text": "gold A",
                    "is_supporting": True,
                },
                {
                    "doc_id": "q1::1::B",
                    "source_question_id": "q1",
                    "title": "B",
                    "text": "gold B",
                    "is_supporting": True,
                },
            ]
        )
        rows = [
            {
                "id": "q1",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [
                    {"doc_id": "q1::0::A", "title": "A", "text": "gold A"},
                    {"doc_id": "other::9::B", "title": "B", "text": "wrong B"},
                ],
            }
        ]

        metrics = retrieval_metrics(rows, support_index)

        self.assertAlmostEqual(metrics["title_all_hit_at10"], 1.0)
        self.assertAlmostEqual(metrics["paragraph_all_hit_at10"], 0.0)
        self.assertAlmostEqual(metrics["paragraph_recall_at10"], 0.5)

    def test_duplicate_paragraph_text_from_another_question_counts_as_evidence(self):
        support_index = build_support_index(
            [
                {
                    "doc_id": "q1::0::A",
                    "source_question_id": "q1",
                    "title": "A",
                    "text": "same supporting paragraph",
                    "is_supporting": True,
                }
            ]
        )
        rows = [
            {
                "id": "q1",
                "supporting_facts": {"title": ["A"]},
                "retrieved": [
                    {
                        "doc_id": "q2::7::A",
                        "title": " A ",
                        "text": "same   supporting paragraph",
                    }
                ],
            }
        ]

        metrics = retrieval_metrics(rows, support_index)

        self.assertAlmostEqual(metrics["paragraph_all_hit_at10"], 1.0)
        self.assertAlmostEqual(metrics["record_all_hit_at10"], 0.0)

    def test_paragraph_key_normalizes_title_and_text(self):
        self.assertEqual(
            paragraph_key({"title": " A\nTitle ", "text": " Some   Text "}),
            ("a title", "some text"),
        )

    def test_latex_table_keeps_all_audited_main_and_equal_budget_rows(self):
        def row(family, method):
            return {
                "dataset": "HotpotQA",
                "family": family,
                "method": method,
                "title_all_hit_at10": 0.5,
                "paragraph_all_hit_at10": 0.5,
                "paragraph_recall_at10": 0.5,
                "title_minus_paragraph_all": 0.0,
            }

        rows = [
            row("Main", "Multi-query"),
            row("Main", "Rule-based iterative"),
            row("Main", "Question + rewrite"),
            row("Main", "Hypothetical-only"),
            row("Equal-budget", "Decomposition"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.tex"
            write_latex(path, rows)
            text = path.read_text(encoding="utf-8")

        for method in [
            "Multi-query",
            "Rule-based iterative",
            "Question + rewrite",
            "Hypothetical-only",
            "Decomposition",
        ]:
            self.assertIn(method, text)


if __name__ == "__main__":
    unittest.main()
