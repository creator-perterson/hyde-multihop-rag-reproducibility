import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_retrieval_answer_correlation import (
    aggregate_bucket_summary,
    bucket_for_recall,
    bootstrap_mean_ci,
    is_primary_retrieval_strategy,
    mcnemar_exact_p,
    paired_metric_delta,
    per_example_record,
    summarize_buckets,
)


class RetrievalAnswerCorrelationTests(unittest.TestCase):
    def test_bucket_for_recall(self):
        self.assertEqual(bucket_for_recall(1.0), "full")
        self.assertEqual(bucket_for_recall(0.5), "partial")
        self.assertEqual(bucket_for_recall(0.0), "none")

    def test_per_example_record_counts_support_hits(self):
        row = {
            "id": "q1",
            "question": "Q?",
            "gold_answer": "Paris",
            "prediction": "Paris",
            "supporting_facts": {"title": ["A", "B"]},
            "retrieved": [{"title": "A"}, {"title": "C"}, {"title": "B"}],
        }
        record = per_example_record("Method", row)
        self.assertEqual(record["support_hit_count"], 2)
        self.assertEqual(record["support_gold_count"], 2)
        self.assertEqual(record["support_recall"], 1.0)
        self.assertEqual(record["bucket"], "full")
        self.assertEqual(record["em"], 1.0)

    def test_summarize_buckets_groups_by_evidence_completeness(self):
        records = [
            {"method": "M", "bucket": "full", "em": 1.0, "f1": 1.0, "support_recall": 1.0},
            {"method": "M", "bucket": "partial", "em": 0.0, "f1": 0.5, "support_recall": 0.5},
        ]
        summary = summarize_buckets(records)
        by_bucket = {row["bucket"]: row for row in summary}
        self.assertEqual(by_bucket["full"]["n"], 1)
        self.assertEqual(by_bucket["full"]["em"], 1.0)
        self.assertEqual(by_bucket["partial"]["mean_support_recall"], 0.5)

    def test_excludes_verifier_rows_from_retrieval_only_correlation(self):
        row = {
            "method": "HyDE-style RAG + Conservative Verifier",
            "top_k": "10",
            "all_hit": "0.8680",
            "mean_recall": "0.9280",
        }

        self.assertFalse(is_primary_retrieval_strategy(row))

    def test_bootstrap_mean_ci_is_deterministic_for_constant_values(self):
        ci = bootstrap_mean_ci([0.5, 0.5, 0.5], iterations=50, seed=7)

        self.assertEqual(ci["mean"], 0.5)
        self.assertEqual(ci["ci_low"], 0.5)
        self.assertEqual(ci["ci_high"], 0.5)

    def test_paired_metric_delta_counts_transitions(self):
        baseline = [
            {"id": "q1", "em": 1.0, "f1": 1.0},
            {"id": "q2", "em": 0.0, "f1": 0.5},
            {"id": "q3", "em": 1.0, "f1": 0.8},
        ]
        target = [
            {"id": "q1", "em": 1.0, "f1": 1.0},
            {"id": "q2", "em": 1.0, "f1": 1.0},
            {"id": "q3", "em": 0.0, "f1": 0.2},
        ]

        summary = paired_metric_delta("Base", "Target", baseline, target, iterations=50, seed=3)

        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["delta_em"], 0.0)
        self.assertAlmostEqual(summary["delta_f1"], -0.033333333333333326)
        self.assertEqual(summary["both_correct"], 1)
        self.assertEqual(summary["baseline_only_correct"], 1)
        self.assertEqual(summary["target_only_correct"], 1)
        self.assertEqual(summary["both_wrong"], 0)

    def test_mcnemar_exact_p_handles_symmetric_discordant_counts(self):
        self.assertEqual(mcnemar_exact_p(1, 1), 1.0)

    def test_aggregate_bucket_summary_pools_records_under_label(self):
        records = [
            {"method": "A", "bucket": "full", "em": 1.0, "f1": 1.0, "support_recall": 1.0},
            {"method": "B", "bucket": "full", "em": 0.0, "f1": 0.5, "support_recall": 1.0},
            {"method": "B", "bucket": "none", "em": 0.0, "f1": 0.0, "support_recall": 0.0},
        ]

        summary = aggregate_bucket_summary(records, method_label="All")
        by_bucket = {row["bucket"]: row for row in summary}

        self.assertEqual(by_bucket["full"]["method"], "All")
        self.assertEqual(by_bucket["full"]["n"], 2)
        self.assertEqual(by_bucket["full"]["em"], 0.5)
        self.assertEqual(by_bucket["none"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
