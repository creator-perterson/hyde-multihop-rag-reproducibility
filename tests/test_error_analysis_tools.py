import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.analyze_qwen_errors import classify_error
from evaluation.compare_answer_sets import compare_pair
from verifier.evaluate_verification import extract_final_answer, transition_label
from verifier.evaluate_selective_verification import conservative_final_answer
from verifier.build_verification_prompts import is_risk_candidate, select_rows
from generator.generate_answers_openai_compatible import build_output_row


class ErrorAnalysisToolTests(unittest.TestCase):
    def test_compare_pair_marks_iterative_gain(self):
        dense = {"prediction": "wrong", "gold_answer": "right"}
        iterative = {"prediction": "right", "gold_answer": "right"}
        result = compare_pair("q1", dense, iterative)
        self.assertEqual(result["category"], "iterative_only_correct")

    def test_classify_retrieval_miss_before_reader_error(self):
        row = {
            "prediction": "Paris",
            "gold_answer": "London",
            "supporting_facts": {"title": ["Gold A", "Gold B"]},
            "retrieved": [{"title": "Gold A", "text": "Paris text."}],
        }
        result = classify_error(row)
        self.assertEqual(result["error_type"], "retrieval_miss")

    def test_classify_alias_or_format_error(self):
        row = {
            "prediction": "United States",
            "gold_answer": "American",
            "supporting_facts": {"title": ["A"]},
            "retrieved": [{"title": "A", "text": "Michael Fay is a United States citizen."}],
        }
        result = classify_error(row)
        self.assertEqual(result["error_type"], "answer_format_or_alias")

    def test_extract_final_answer_from_json(self):
        text = '{"verdict": "correct", "final_answer": "American", "reason": "nationality"}'
        self.assertEqual(extract_final_answer(text), "American")

    def test_transition_label(self):
        self.assertEqual(transition_label(False, True), "wrong_to_correct")
        self.assertEqual(transition_label(True, False), "correct_to_wrong")

    def test_build_output_row_preserves_initial_prediction(self):
        row = {
            "id": "q1",
            "question": "Q?",
            "gold_answer": "A",
            "supporting_facts": {},
            "retrieved": [],
            "prompt": "prompt",
            "initial_prediction": "wrong",
        }
        output = build_output_row(row, "{}", "model-x")
        self.assertEqual(output["initial_prediction"], "wrong")

    def test_risk_candidate_includes_abstained_answer(self):
        row = {
            "question": "What city is the writer based in?",
            "prediction": "I don't know",
        }
        self.assertTrue(is_risk_candidate(row))

    def test_risk_selection_caps_expanded_candidates(self):
        rows = [
            {"id": f"q{i}", "question": "Who directed the film?", "prediction": "John"}
            for i in range(160)
        ]
        selected = select_rows(rows, limit=None, strategy="risk", risk_target=120)
        self.assertEqual(len(selected), 120)

    def test_risk_selection_prioritizes_legacy_high_precision_rules(self):
        rows = [
            {"id": "abstain", "question": "What city is the writer based in?", "prediction": "I don't know"},
            {"id": "legacy", "question": "What nationality was the victim?", "prediction": "United States"},
        ]
        selected = select_rows(rows, limit=None, strategy="risk", risk_target=2)
        self.assertEqual(selected[0]["id"], "legacy")

    def test_conservative_final_answer_blocks_new_abstention(self):
        verifier_text = """{"verdict":"abstain","final_answer":"I don't know","reason":"insufficient"}"""
        self.assertEqual(conservative_final_answer("Nassau County", verifier_text), "Nassau County")

    def test_conservative_final_answer_blocks_single_word_numeric_unit(self):
        verifier_text = """{"verdict":"correct","final_answer":"94 episodes","reason":"added unit"}"""
        self.assertEqual(conservative_final_answer("94", verifier_text), "94")

    def test_conservative_final_answer_allows_multiword_numeric_unit(self):
        verifier_text = """{"verdict":"correct","final_answer":"12 member universities","reason":"added specific unit"}"""
        question = "How many member universities are there in this conference?"
        self.assertEqual(conservative_final_answer("12", verifier_text, question=question), "12 member universities")

    def test_conservative_final_answer_blocks_unasked_multiword_numeric_unit(self):
        verifier_text = """{"verdict":"correct","final_answer":"37 feature films","reason":"added specific unit"}"""
        question = "How many films were directed by the director of Wise Blood?"
        self.assertEqual(conservative_final_answer("37", verifier_text, question=question), "37")

    def test_conservative_final_answer_keeps_initial_on_malformed_json(self):
        self.assertEqual(conservative_final_answer("Nassau County", "not valid json"), "Nassau County")

    def test_conservative_final_answer_keeps_initial_when_final_answer_missing(self):
        verifier_text = """{"verdict":"correct","reason":"missing final answer"}"""
        self.assertEqual(conservative_final_answer("Nassau County", verifier_text), "Nassau County")


if __name__ == "__main__":
    unittest.main()
