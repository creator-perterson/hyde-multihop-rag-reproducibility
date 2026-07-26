import argparse
import csv
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl, write_jsonl


def compare_pair(row_id, dense_row, iterative_row):
    dense_em = exact_match(dense_row["prediction"], dense_row["gold_answer"])
    iterative_em = exact_match(iterative_row["prediction"], iterative_row["gold_answer"])
    dense_f1 = f1_score(dense_row["prediction"], dense_row["gold_answer"])
    iterative_f1 = f1_score(iterative_row["prediction"], iterative_row["gold_answer"])

    if dense_em and iterative_em:
        category = "both_correct"
    elif dense_em and not iterative_em:
        category = "dense_only_correct"
    elif not dense_em and iterative_em:
        category = "iterative_only_correct"
    else:
        category = "both_wrong"

    return {
        "id": row_id,
        "question": iterative_row.get("question", dense_row.get("question", "")),
        "gold_answer": iterative_row["gold_answer"],
        "dense_prediction": dense_row["prediction"],
        "iterative_prediction": iterative_row["prediction"],
        "dense_em": int(dense_em),
        "iterative_em": int(iterative_em),
        "dense_f1": dense_f1,
        "iterative_f1": iterative_f1,
        "delta_f1": iterative_f1 - dense_f1,
        "category": category,
    }


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {
        "n": len(rows),
        "dense_em": sum(row["dense_em"] for row in rows) / len(rows),
        "iterative_em": sum(row["iterative_em"] for row in rows) / len(rows),
        "dense_f1": sum(row["dense_f1"] for row in rows) / len(rows),
        "iterative_f1": sum(row["iterative_f1"] for row in rows) / len(rows),
        "delta_f1": sum(row["delta_f1"] for row in rows) / len(rows),
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--iterative", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_jsonl", default=None)
    args = parser.parse_args()

    dense_rows = load_by_id(args.dense)
    iterative_rows = load_by_id(args.iterative)
    shared_ids = [row_id for row_id in iterative_rows if row_id in dense_rows]
    rows = [compare_pair(row_id, dense_rows[row_id], iterative_rows[row_id]) for row_id in shared_ids]

    write_csv(args.out_csv, rows)
    if args.out_jsonl:
        write_jsonl(args.out_jsonl, rows)

    summary = summarize(rows)
    print(f"Compared questions: {summary['n']}")
    print(f"Dense EM/F1: {summary['dense_em']:.4f} / {summary['dense_f1']:.4f}")
    print(f"Iterative EM/F1: {summary['iterative_em']:.4f} / {summary['iterative_f1']:.4f}")
    print(f"Delta F1: {summary['delta_f1']:.4f}")
    print("Categories:")
    for key, value in sorted(summary["categories"].items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
