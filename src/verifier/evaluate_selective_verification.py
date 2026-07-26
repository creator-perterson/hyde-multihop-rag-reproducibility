import argparse
from collections import Counter
from pathlib import Path
import re

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl, write_jsonl
from verifier.evaluate_verification import extract_final_answer, transition_label


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def is_abstention(answer):
    normalized = str(answer).strip().lower().rstrip(".")
    return normalized.startswith("i don") or normalized in {"unknown", "not enough information"}


def numeric_unit_suffix(initial, final):
    initial_text = str(initial).strip()
    final_text = str(final).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", initial_text):
        return ""
    match = re.fullmatch(rf"{re.escape(initial_text)}\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*)*)", final_text)
    return match.group(1) if match else ""


def should_block_numeric_unit_addition(initial, final, question=""):
    suffix = numeric_unit_suffix(initial, final)
    if not suffix:
        return False
    if len(suffix.split()) == 1:
        return True
    return suffix.lower() not in str(question).lower()


def conservative_final_answer(initial, verifier_text, question=""):
    final = extract_final_answer(verifier_text, fallback=initial)
    if not is_abstention(initial) and is_abstention(final):
        return initial
    if should_block_numeric_unit_addition(initial, final, question):
        return initial
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_answers", required=True)
    parser.add_argument("--verifier_answers", required=True)
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    base_rows = load_by_id(args.base_answers)
    verifier_rows = load_by_id(args.verifier_answers)
    rows = []
    for row_id, base in base_rows.items():
        initial = base["prediction"]
        verifier_row = verifier_rows.get(row_id)
        final = conservative_final_answer(initial, verifier_row["prediction"], base.get("question", "")) if verifier_row else initial
        initial_correct = exact_match(initial, base["gold_answer"])
        final_correct = exact_match(final, base["gold_answer"])
        rows.append({
            "id": row_id,
            "question": base["question"],
            "gold_answer": base["gold_answer"],
            "initial_prediction": initial,
            "final_answer": final,
            "verified": int(verifier_row is not None),
            "initial_em": int(initial_correct),
            "final_em": int(final_correct),
            "initial_f1": f1_score(initial, base["gold_answer"]),
            "final_f1": f1_score(final, base["gold_answer"]),
            "transition": transition_label(initial_correct, final_correct),
        })

    write_jsonl(args.out_jsonl, rows)
    total = len(rows)
    transitions = Counter(row["transition"] for row in rows if row["verified"])
    print(f"Questions: {total}")
    print(f"Verified subset: {sum(row['verified'] for row in rows)}")
    print(f"Initial EM: {sum(row['initial_em'] for row in rows) / total:.4f}")
    print(f"Final EM: {sum(row['final_em'] for row in rows) / total:.4f}")
    print(f"Initial F1: {sum(row['initial_f1'] for row in rows) / total:.4f}")
    print(f"Final F1: {sum(row['final_f1'] for row in rows) / total:.4f}")
    print("Verified-subset transitions:")
    for key, value in transitions.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
