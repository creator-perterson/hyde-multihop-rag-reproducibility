import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_query_answer_overlap import build_audit_rows


class AnalyzeQueryAnswerOverlapTests(unittest.TestCase):
    def test_build_audit_rows_detects_answer_overlap_and_support_metrics(self):
        query_rows = [
            {
                "id": "q1",
                "question": "Who wrote it?",
                "gold_answer": "Ada Lovelace",
                "prediction": "Ada Lovelace author",
            }
        ]
        retrieval_by_id = {
            "q1": {
                "supporting_facts": {"title": ["Ada Lovelace"]},
                "retrieved": [{"title": "Ada Lovelace"}],
            }
        }

        rows = build_audit_rows(query_rows, retrieval_by_id)

        self.assertEqual(rows[0]["answer_in_query"], 1)
        self.assertEqual(rows[0]["all_hit"], 1)
        self.assertEqual(rows[0]["support_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
