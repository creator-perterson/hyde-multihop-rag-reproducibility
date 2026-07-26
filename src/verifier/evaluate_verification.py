import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl, write_jsonl


def extract_final_answer(text, fallback=None):
    text = text.strip()
    try:
        payload = json.loads(text)
        final_answer = str(payload.get("final_answer", "")).strip()
        return final_answer if final_answer else (fallback if fallback is not None else "")
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                final_answer = str(payload.get("final_answer", "")).strip()
                return final_answer if final_answer else (fallback if fallback is not None else "")
            except json.JSONDecodeError:
                pass
    return fallback if fallback is not None else text


def transition_label(initial_correct, final_correct):
    if initial_correct and final_correct:
        return "correct_to_correct"
    if initial_correct and not final_correct:
        return "correct_to_wrong"
    if not initial_correct and final_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def evaluate_row(row):
    initial_prediction = row.get("initial_prediction") or row.get("prediction", "")
    verifier_output = row["prediction"]
    final_answer = extract_final_answer(verifier_output, fallback=initial_prediction)
    gold = row["gold_answer"]
    initial_correct = exact_match(initial_prediction, gold)
    final_correct = exact_match(final_answer, gold)
    return {
        "id": row["id"],
        "question": row["question"],
        "gold_answer": gold,
        "initial_prediction": initial_prediction,
        "verifier_output": verifier_output,
        "final_answer": final_answer,
        "initial_em": int(initial_correct),
        "final_em": int(final_correct),
        "initial_f1": f1_score(initial_prediction, gold),
        "final_f1": f1_score(final_answer, gold),
        "transition": transition_label(initial_correct, final_correct),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    rows = [evaluate_row(row) for row in read_jsonl(args.answers)]
    write_jsonl(args.out_jsonl, rows)
    transitions = Counter(row["transition"] for row in rows)
    total = len(rows)
    print(f"Questions: {total}")
    print(f"Initial EM: {sum(row['initial_em'] for row in rows) / total:.4f}")
    print(f"Final EM: {sum(row['final_em'] for row in rows) / total:.4f}")
    print(f"Initial F1: {sum(row['initial_f1'] for row in rows) / total:.4f}")
    print(f"Final F1: {sum(row['final_f1'] for row in rows) / total:.4f}")
    print("Transitions:")
    for key, value in transitions.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
