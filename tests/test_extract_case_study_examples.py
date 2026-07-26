import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.extract_case_study_examples import (
    build_case_artifact_details,
    support_metrics,
    select_guard_case,
    select_hyde_success_case,
    select_alias_limitation_case,
)


class ExtractCaseStudyExamplesTests(unittest.TestCase):
    def test_support_metrics_detects_full_partial_none(self):
        row = {
            "supporting_facts": {"title": ["A", "B"]},
            "retrieved": [{"title": "A"}, {"title": "C"}],
        }

        metrics = support_metrics(row)

        self.assertEqual(metrics["support_hit_count"], 1)
        self.assertEqual(metrics["support_gold_count"], 2)
        self.assertEqual(metrics["bucket"], "partial")

    def test_select_hyde_success_prefers_answer_absent_full_support_gain(self):
        dense = {
            "q1": {
                "id": "q1",
                "question": "Q?",
                "gold_answer": "Paris",
                "prediction": "London",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}],
            }
        }
        hyde = {
            "q1": {
                "id": "q1",
                "question": "Q?",
                "gold_answer": "Paris",
                "prediction": "Paris",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A"}, {"title": "B"}],
            }
        }
        leakage = {"q1": {"answer_in_hyde": 0}}

        case = select_hyde_success_case(dense, hyde, leakage)

        self.assertEqual(case["id"], "q1")
        self.assertEqual(case["case_type"], "HyDE evidence acquisition")

    def test_select_guard_case_finds_raw_harm_prevented_by_guard(self):
        detail_rows = [
            {"id": "q1", "variant": "raw", "transition": "correct_to_wrong", "initial_prediction": "94", "final_answer": "94 episodes", "gold_answer": "94", "question": "How many?"},
            {"id": "q1", "variant": "both_guards", "transition": "correct_to_correct", "initial_prediction": "94", "final_answer": "94", "gold_answer": "94", "question": "How many?"},
        ]

        case = select_guard_case(detail_rows)

        self.assertEqual(case["id"], "q1")
        self.assertEqual(case["case_type"], "Verifier guard prevents harm")

    def test_select_alias_limitation_case_uses_preferred_alias_example(self):
        preferred_id = "5a728f015542991f9a20c4e4"
        hyde = {
            preferred_id: {
                "id": preferred_id,
                "question": "Q?",
                "gold_answer": "London",
                "prediction": "England",
                "supporting_facts": {"title": ["A"]},
                "retrieved": [{"title": "A"}],
            }
        }

        case = select_alias_limitation_case(hyde)

        self.assertEqual(case["id"], preferred_id)
        self.assertEqual(case["case_type"], "Evaluation / alias limitation")

    def test_build_case_artifact_details_compacts_retrieval_and_verifier_artifacts(self):
        cases = [
            {
                "case_type": "HyDE evidence acquisition",
                "id": "q1",
                "hyde_prediction": "Paris",
            },
            {
                "case_type": "Verifier guard prevents harm",
                "id": "q2",
                "initial_prediction": "37",
                "guarded_final_answer": "37",
            },
        ]
        dense_retrieval_by_id = {
            "q1": {
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A", "text": "Alpha evidence."}, {"title": "C", "text": "Distractor."}],
            }
        }
        hyde_retrieval_by_id = {
            "q1": {
                "hyde_document": "A hypothetical passage about Paris.",
                "supporting_facts": {"title": ["A", "B"]},
                "retrieved": [{"title": "A", "text": "Alpha evidence."}, {"title": "B", "text": "Beta evidence."}],
            }
        }
        hyde_answers_by_id = {"q1": {"prediction": "Paris"}}
        verifier_by_id = {"q2": {"prediction": '{"verdict":"incorrect","final_answer":"37 feature films"}'}}
        guard_rows = [
            {"id": "q2", "variant": "both_guards", "final_answer": "37"},
        ]

        details = build_case_artifact_details(
            cases,
            dense_retrieval_by_id,
            hyde_retrieval_by_id,
            hyde_answers_by_id,
            verifier_by_id,
            guard_rows,
        )

        self.assertEqual(details[0]["example_id"], "q1")
        self.assertIn("HyDE evidence acquisition", details[0]["case_type"])
        self.assertIn("A hypothetical passage", details[0]["generated_hypothetical_passage"])
        self.assertIn("Dense top: A; C", details[0]["dense_retrieval"])
        self.assertIn("supporting-title hits: A", details[0]["dense_retrieval"])
        self.assertIn("HyDE top: A; B", details[0]["hyde_retrieval"])
        self.assertIn("[A] Alpha evidence.", details[0]["relevant_evidence_excerpts"])
        self.assertEqual(details[0]["initial_reader_answer"], "Paris")
        self.assertEqual(details[0]["raw_verifier_json"], "not invoked for this example")
        self.assertEqual(details[0]["final_guarded_answer"], "Paris")
        self.assertEqual(details[1]["raw_verifier_json"], '{"verdict":"incorrect","final_answer":"37 feature films"}')
        self.assertEqual(details[1]["final_guarded_answer"], "37")


if __name__ == "__main__":
    unittest.main()
