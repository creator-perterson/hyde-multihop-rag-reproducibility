import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.summarize_equal_budget_reader import (
    format_float,
    paired_contrast,
    summarize_reader_outputs,
)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class SummarizeEqualBudgetReaderTests(unittest.TestCase):
    def test_format_float_suppresses_negative_zero(self):
        self.assertEqual(format_float(-0.000035), "0.0000")
        self.assertEqual(format_float(-0.002), "-0.0020")

    def test_cli_accepts_mode_filter_for_minimal_reader_check(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "evaluation" / "summarize_equal_budget_reader.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--mode"', source)
        self.assertIn("modes=modes", source)

    def test_requires_expected_rows_for_each_answer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            answers_dir = base / "reader_answers"
            answers_dir.mkdir()
            write_jsonl(
                answers_dir / "ircot_hotpotqa_test500_equal_budget_bge_base_direct_rewrite_top10_answers_qwenmax_500.jsonl",
                [{"id": "1", "prediction": "a", "gold_answer": "a"}],
            )

            with self.assertRaisesRegex(RuntimeError, "expected 2 rows"):
                summarize_reader_outputs(
                    answers_dir=answers_dir,
                    summary_csv=base / "summary.csv",
                    contrasts_csv=base / "contrasts.csv",
                    expected_n=2,
                    readers=[("qwenmax", "qwen3.7-max")],
                    datasets=[("HotpotQA", "ircot_hotpotqa_test500_equal_budget_bge_base")],
                    modes=["direct_rewrite"],
                )

    def test_writes_summary_and_document_like_vs_direct_contrast(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            answers_dir = base / "reader_answers"
            answers_dir.mkdir()
            prefix = "ircot_hotpotqa_test500_equal_budget_bge_base"
            rows = [
                {"id": "1", "prediction": "alpha", "gold_answer": "alpha"},
                {"id": "2", "prediction": "wrong", "gold_answer": "beta"},
            ]
            write_jsonl(answers_dir / f"{prefix}_direct_rewrite_top10_answers_qwenmax_500.jsonl", rows)
            write_jsonl(
                answers_dir / f"{prefix}_document_like_passage_top10_answers_qwenmax_500.jsonl",
                [
                    {"id": "1", "prediction": "alpha", "gold_answer": "alpha"},
                    {"id": "2", "prediction": "beta", "gold_answer": "beta"},
                ],
            )

            summary_rows, contrast_rows = summarize_reader_outputs(
                answers_dir=answers_dir,
                summary_csv=base / "summary.csv",
                contrasts_csv=base / "contrasts.csv",
                expected_n=2,
                readers=[("qwenmax", "qwen3.7-max")],
                datasets=[("HotpotQA", prefix)],
                modes=["direct_rewrite", "document_like_passage"],
            )

            self.assertEqual(len(summary_rows), 2)
            direct = [row for row in summary_rows if row["query_mode"] == "direct_rewrite"][0]
            doc = [row for row in summary_rows if row["query_mode"] == "document_like_passage"][0]
            self.assertEqual(direct["em"], 0.5)
            self.assertEqual(doc["f1"], 1.0)
            self.assertEqual(len(contrast_rows), 1)
            self.assertEqual(contrast_rows[0]["delta_f1"], 0.5)
            self.assertTrue((base / "summary.csv").exists())
            self.assertTrue((base / "contrasts.csv").exists())

    def test_writes_document_like_vs_keyword_contrast(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            answers_dir = base / "reader_answers"
            answers_dir.mkdir()
            prefix = "ircot_hotpotqa_test500_equal_budget_bge_base"
            write_jsonl(
                answers_dir / f"{prefix}_keyword_expansion_top10_answers_qwenturbo_500.jsonl",
                [
                    {"id": "1", "prediction": "alpha", "gold_answer": "alpha"},
                    {"id": "2", "prediction": "wrong", "gold_answer": "beta"},
                ],
            )
            write_jsonl(
                answers_dir / f"{prefix}_document_like_passage_top10_answers_qwenturbo_500.jsonl",
                [
                    {"id": "1", "prediction": "alpha", "gold_answer": "alpha"},
                    {"id": "2", "prediction": "beta", "gold_answer": "beta"},
                ],
            )

            _, contrast_rows = summarize_reader_outputs(
                answers_dir=answers_dir,
                summary_csv=base / "summary.csv",
                contrasts_csv=base / "contrasts.csv",
                expected_n=2,
                readers=[("qwenturbo", "qwen-turbo")],
                datasets=[("HotpotQA", prefix)],
                modes=["keyword_expansion", "document_like_passage"],
            )

            self.assertEqual(len(contrast_rows), 1)
            contrast = contrast_rows[0]
            self.assertEqual(contrast["baseline"], "keyword_expansion")
            self.assertEqual(contrast["target"], "document_like_passage")
            self.assertEqual(contrast["delta_f1"], 0.5)

    def test_paired_contrast_reports_ci_mcnemar_and_transitions(self):
        loaded = {
            ("HotpotQA", "qwenturbo", "direct_rewrite"): {
                "1": {"id": "1", "prediction": "wrong", "gold_answer": "alpha"},
                "2": {"id": "2", "prediction": "beta", "gold_answer": "beta"},
                "3": {"id": "3", "prediction": "wrong", "gold_answer": "gamma"},
                "4": {"id": "4", "prediction": "delta", "gold_answer": "delta"},
            },
            ("HotpotQA", "qwenturbo", "document_like_passage"): {
                "1": {"id": "1", "prediction": "alpha", "gold_answer": "alpha"},
                "2": {"id": "2", "prediction": "wrong", "gold_answer": "beta"},
                "3": {"id": "3", "prediction": "wrong", "gold_answer": "gamma"},
                "4": {"id": "4", "prediction": "delta", "gold_answer": "delta"},
            },
        }

        contrast = paired_contrast(
            "HotpotQA",
            "qwenturbo",
            "direct_rewrite",
            "document_like_passage",
            loaded,
            iterations=200,
            seed=7,
        )

        self.assertEqual(contrast["n"], 4)
        self.assertEqual(contrast["delta_em"], 0.0)
        self.assertEqual(contrast["wrong_to_correct"], 1)
        self.assertEqual(contrast["correct_to_wrong"], 1)
        self.assertEqual(contrast["both_correct"], 1)
        self.assertEqual(contrast["both_wrong"], 1)
        self.assertEqual(contrast["mcnemar_exact_p"], 1.0)
        self.assertIn("delta_f1_ci_low", contrast)
        self.assertIn("delta_f1_ci_high", contrast)
        self.assertLessEqual(contrast["delta_f1_ci_low"], contrast["delta_f1"])
        self.assertGreaterEqual(contrast["delta_f1_ci_high"], contrast["delta_f1"])


if __name__ == "__main__":
    unittest.main()
