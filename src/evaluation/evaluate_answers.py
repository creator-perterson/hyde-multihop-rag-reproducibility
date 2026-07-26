import argparse
import re
import string
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl


def normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction, gold):
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_score(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="results/hotpotqa_top5_answers.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    em_scores = []
    f1_scores = []
    for row in rows:
        em_scores.append(float(exact_match(row["prediction"], row["gold_answer"])))
        f1_scores.append(f1_score(row["prediction"], row["gold_answer"]))

    total = len(rows)
    print(f"Questions: {total}")
    print(f"Exact Match: {sum(em_scores) / total:.4f}")
    print(f"Token F1: {sum(f1_scores) / total:.4f}")
    print("\nFirst 5 examples:")
    for row, em, f1 in list(zip(rows, em_scores, f1_scores))[:5]:
        print("-" * 80)
        print(row["question"])
        print(f"gold: {row['gold_answer']}")
        print(f"pred: {row['prediction']}")
        print(f"EM={em:.0f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
