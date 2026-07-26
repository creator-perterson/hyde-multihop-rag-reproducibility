import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl


MODES = [
    ("question_only", "Question only"),
    ("single_rewritten_query", "Single rewritten query"),
    ("question_plus_rewritten_query", "Question + rewritten query"),
    ("hypothetical_only", "Hypothetical passage only"),
    ("question_plus_hypothetical", "Question + hypothetical passage"),
]


def bootstrap_delta_ci(deltas, iterations=2000, seed=13, alpha=0.05):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    return lo, hi


def retrieval_metrics(row):
    gold_titles = set(row["supporting_facts"]["title"])
    retrieved_titles = {doc["title"] for doc in row["retrieved"]}
    hit_titles = gold_titles & retrieved_titles
    return {
        "any_hit@10": float(bool(hit_titles)),
        "all_support_hit@10": float(bool(gold_titles and gold_titles.issubset(retrieved_titles))),
        "supporting_title_recall@10": len(hit_titles) / len(gold_titles) if gold_titles else 0.0,
    }


def answer_metrics(row):
    return {
        "em": float(exact_match(row["prediction"], row["gold_answer"])),
        "f1": f1_score(row["prediction"], row["gold_answer"]),
    }


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def mean_metric(rows_by_id, ids, metric_fn, key):
    return sum(metric_fn(rows_by_id[qid])[key] for qid in ids) / len(ids)


def paired_delta(rows_a, rows_b, ids, metric_fn, key):
    return [metric_fn(rows_b[qid])[key] - metric_fn(rows_a[qid])[key] for qid in ids]


def summarize_mode(dataset, prefix, mode, label, results_dir):
    retrieval_path = results_dir / f"{prefix}_{mode}_top10_retrieval.jsonl"
    answers_path = results_dir / f"{prefix}_{mode}_top10_answers_qwenmax_500.jsonl"
    retrieval_by_id = load_by_id(retrieval_path)
    answers_by_id = load_by_id(answers_path)
    ids = [qid for qid in retrieval_by_id if qid in answers_by_id]
    if len(ids) != 500:
        raise RuntimeError(f"{dataset} {mode}: expected 500 paired rows, found {len(ids)}")
    row = {
        "dataset": dataset,
        "query_mode": mode,
        "query_input": label,
        "n": len(ids),
    }
    for key in ("any_hit@10", "all_support_hit@10", "supporting_title_recall@10"):
        row[key] = mean_metric(retrieval_by_id, ids, retrieval_metrics, key)
    row["em"] = mean_metric(answers_by_id, ids, answer_metrics, "em")
    row["f1"] = mean_metric(answers_by_id, ids, answer_metrics, "f1")
    return row, retrieval_by_id, answers_by_id, ids


def summarize_contrast(dataset, baseline_mode, target_mode, loaded, metric_group, key, seed):
    baseline = loaded[baseline_mode]
    target = loaded[target_mode]
    metric_fn = retrieval_metrics if metric_group == "retrieval" else answer_metrics
    ids = [qid for qid in baseline["ids"] if qid in target["ids"]]
    base_rows = baseline["retrieval"] if metric_group == "retrieval" else baseline["answers"]
    target_rows = target["retrieval"] if metric_group == "retrieval" else target["answers"]
    deltas = paired_delta(base_rows, target_rows, ids, metric_fn, key)
    lo, hi = bootstrap_delta_ci(deltas, seed=seed)
    return {
        "dataset": dataset,
        "baseline": baseline_mode,
        "target": target_mode,
        "metric_group": metric_group,
        "metric": key,
        "n": len(ids),
        "baseline_value": mean_metric(base_rows, ids, metric_fn, key),
        "target_value": mean_metric(target_rows, ids, metric_fn, key),
        "delta": sum(deltas) / len(deltas),
        "ci_low": lo,
        "ci_high": hi,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--contrasts_csv", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_rows = []
    loaded = {}
    for mode, label in MODES:
        row, retrieval, answers, ids = summarize_mode(
            args.dataset, args.prefix, mode, label, results_dir
        )
        summary_rows.append(row)
        loaded[mode] = {"retrieval": retrieval, "answers": answers, "ids": set(ids)}

    contrast_rows = []
    contrasts = [
        ("question_only", "question_plus_hypothetical"),
        ("question_plus_rewritten_query", "question_plus_hypothetical"),
        ("question_only", "hypothetical_only"),
    ]
    seed = 13
    for baseline_mode, target_mode in contrasts:
        for metric_group, keys in (
            ("retrieval", ("all_support_hit@10", "supporting_title_recall@10")),
            ("answer", ("em", "f1")),
        ):
            for key in keys:
                contrast_rows.append(
                    summarize_contrast(
                        args.dataset,
                        baseline_mode,
                        target_mode,
                        loaded,
                        metric_group,
                        key,
                        seed,
                    )
                )
                seed += 1

    write_csv(Path(args.summary_csv), summary_rows)
    write_csv(Path(args.contrasts_csv), contrast_rows)
    for row in summary_rows:
        print(
            f"{row['query_input']}: any={row['any_hit@10']:.4f}, "
            f"all={row['all_support_hit@10']:.4f}, "
            f"recall={row['supporting_title_recall@10']:.4f}, "
            f"EM={row['em']:.4f}, F1={row['f1']:.4f}"
        )
    print(f"Saved {args.summary_csv}")
    print(f"Saved {args.contrasts_csv}")


if __name__ == "__main__":
    main()
