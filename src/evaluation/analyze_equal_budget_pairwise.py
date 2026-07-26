import argparse
import csv
import random
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl


METRICS = ("all_support_hit@10", "supporting_title_recall@10")
BASELINE_MODES = (
    "direct_rewrite",
    "question_decomposition",
    "keyword_expansion",
)
DATASETS = {
    "hotpotqa": ("HotpotQA", "ircot_hotpotqa_test500_equal_budget_bge_base"),
    "2wiki": ("2WikiMultihopQA", "ircot_2wiki_test500_equal_budget_bge_base"),
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values, quantile):
    index = int(round(quantile * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def bootstrap_delta_ci(deltas, iterations=2000, seed=13, alpha=0.05):
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    estimates = []
    for _ in range(iterations):
        estimates.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return percentile(estimates, alpha / 2), percentile(estimates, 1 - alpha / 2)


def per_example_retrieval_metrics(rows):
    metrics = {}
    for row in rows:
        item_id = row["id"]
        if item_id in metrics:
            raise RuntimeError(f"Duplicate retrieval id: {item_id}")
        gold_titles = set(row["supporting_facts"]["title"])
        retrieved_titles = {doc["title"] for doc in row["retrieved"]}
        hit_titles = gold_titles & retrieved_titles
        metrics[item_id] = {
            "all_support_hit@10": float(bool(gold_titles) and gold_titles.issubset(retrieved_titles)),
            "supporting_title_recall@10": len(hit_titles) / len(gold_titles) if gold_titles else 0.0,
        }
    return metrics


def analyze_pairwise_comparisons(
    per_example_by_mode,
    target_mode="document_like_passage",
    baseline_modes=BASELINE_MODES,
    iterations=2000,
    seed=13,
):
    if target_mode not in per_example_by_mode:
        raise RuntimeError(f"Missing target mode: {target_mode}")
    target = per_example_by_mode[target_mode]
    rows = []
    for comparison_index, baseline_mode in enumerate(baseline_modes):
        if baseline_mode not in per_example_by_mode:
            continue
        baseline = per_example_by_mode[baseline_mode]
        if set(baseline) != set(target):
            raise RuntimeError(
                f"{baseline_mode} and {target_mode} must contain identical question ids for paired analysis."
            )
        item_ids = sorted(target)
        row = {
            "baseline_query_mode": baseline_mode,
            "target_query_mode": target_mode,
            "n": len(item_ids),
        }
        for metric_index, metric_name in enumerate(METRICS):
            deltas = [target[item_id][metric_name] - baseline[item_id][metric_name] for item_id in item_ids]
            ci_low, ci_high = bootstrap_delta_ci(
                deltas,
                iterations=iterations,
                seed=seed + comparison_index * 10 + metric_index,
            )
            row[f"{metric_name}_delta"] = mean(deltas)
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
        rows.append(row)
    return rows


def write_csv(path, rows):
    fieldnames = [
        "dataset",
        "baseline_query_mode",
        "target_query_mode",
        "n",
        "all_support_hit@10_delta",
        "all_support_hit@10_ci_low",
        "all_support_hit@10_ci_high",
        "supporting_title_recall@10_delta",
        "supporting_title_recall@10_ci_low",
        "supporting_title_recall@10_ci_high",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_metrics(retrieval_dir, prefix):
    by_mode = {}
    for mode in (*BASELINE_MODES, "document_like_passage"):
        path = Path(retrieval_dir) / f"{prefix}_{mode}_top10_retrieval.jsonl"
        by_mode[mode] = per_example_retrieval_metrics(list(read_jsonl(path)))
    return by_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=list(DATASETS))
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    all_rows = []
    for dataset_index, dataset_key in enumerate(args.datasets):
        dataset_name, prefix = DATASETS[dataset_key]
        rows = analyze_pairwise_comparisons(
            load_dataset_metrics(args.retrieval_dir, prefix),
            iterations=args.bootstrap_iterations,
            seed=args.seed + dataset_index * 100,
        )
        for row in rows:
            row["dataset"] = dataset_name
        all_rows.extend(rows)
    write_csv(args.out_csv, all_rows)
    for row in all_rows:
        print(
            f"{row['dataset']} document_like - {row['baseline_query_mode']}: "
            f"all={row['all_support_hit@10_delta']:+.4f} "
            f"[{row['all_support_hit@10_ci_low']:+.4f}, {row['all_support_hit@10_ci_high']:+.4f}], "
            f"recall={row['supporting_title_recall@10_delta']:+.4f} "
            f"[{row['supporting_title_recall@10_ci_low']:+.4f}, "
            f"{row['supporting_title_recall@10_ci_high']:+.4f}]"
        )


if __name__ == "__main__":
    main()
