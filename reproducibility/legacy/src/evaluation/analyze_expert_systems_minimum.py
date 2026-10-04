import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl


PRIMARY_CONTRASTS = [
    ("open_hyde_standard", "dense_bge"),
    ("open_query2doc_4shot", "dense_bge"),
    ("open_query2doc_4shot", "open_hyde_standard"),
    ("open_hyde_evidence_only", "open_hyde_standard"),
]


def support_metrics(row: dict) -> dict:
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved = row.get("retrieved", [])[:10]
    retrieved_titles = {doc.get("title", "") for doc in retrieved}
    hits = gold_titles & retrieved_titles
    ranks = {
        title: min(
            int(doc.get("rank", index))
            for index, doc in enumerate(retrieved, start=1)
            if doc.get("title") == title
        )
        for title in hits
    }
    missing_rank = len(retrieved) + 1
    worst_rank = (
        max(ranks.get(title, missing_rank) for title in gold_titles)
        if gold_titles
        else missing_rank
    )
    return {
        "any_hit@10": float(bool(hits)),
        "all_support_hit@10": float(
            bool(gold_titles) and gold_titles.issubset(retrieved_titles)
        ),
        "supporting_title_recall@10": (
            len(hits) / len(gold_titles) if gold_titles else 0.0
        ),
        "worst_supporting_title_rank": float(worst_rank),
    }


def index_unique(rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in indexed:
            raise ValueError(f"Duplicate ID in {label}: {row_id}")
        indexed[row_id] = row
    return indexed


def align_method_rows(method_rows: dict[str, list[dict]]) -> tuple[list[str], dict]:
    if not method_rows:
        raise ValueError("At least one retrieval method is required")
    indexed = {
        method: index_unique(rows, method) for method, rows in method_rows.items()
    }
    reference_method = (
        "dense_bge" if "dense_bge" in indexed else next(iter(indexed))
    )
    ordered_ids = [str(row["id"]) for row in method_rows[reference_method]]
    reference_ids = set(ordered_ids)
    for method, rows_by_id in indexed.items():
        missing = reference_ids - set(rows_by_id)
        extra = set(rows_by_id) - reference_ids
        if missing or extra:
            raise ValueError(
                f"ID mismatch for {method}: missing={len(missing)}, extra={len(extra)}"
            )
    return ordered_ids, indexed


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_bootstrap_ci(
    example_deltas: list[float], iterations: int = 10_000, seed: int = 13
) -> tuple[float, float]:
    if not example_deltas:
        raise ValueError("Paired bootstrap requires at least one example")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    n = len(example_deltas)
    estimates = []
    for _ in range(iterations):
        estimates.append(
            sum(example_deltas[rng.randrange(n)] for _ in range(n)) / n
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_mcnemar(reference: list[float], target: list[float]) -> dict:
    if len(reference) != len(target):
        raise ValueError("McNemar inputs must be paired")
    gains = sum(not bool(before) and bool(after) for before, after in zip(reference, target))
    losses = sum(bool(before) and not bool(after) for before, after in zip(reference, target))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "gains": gains,
        "losses": losses,
        "discordant_total": discordant,
        "p_value": float(p_value),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    previous = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[original_index]))
        previous = max(previous, candidate)
        adjusted[original_index] = previous
    return adjusted


def analyze_dataset(
    dataset: str,
    method_rows: dict[str, list[dict]],
    iterations: int = 10_000,
    seed: int = 13,
) -> tuple[list[dict], list[dict]]:
    required = {method for contrast in PRIMARY_CONTRASTS for method in contrast}
    missing_methods = required - set(method_rows)
    if missing_methods:
        raise ValueError(f"Missing required methods for {dataset}: {sorted(missing_methods)}")
    ids, indexed = align_method_rows(method_rows)
    metrics_by_method = {
        method: {row_id: support_metrics(rows[row_id]) for row_id in ids}
        for method, rows in indexed.items()
    }

    summaries = []
    for method in sorted(metrics_by_method):
        metric_rows = metrics_by_method[method]
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(ids),
                **{
                    metric: statistics.fmean(
                        metric_rows[row_id][metric] for row_id in ids
                    )
                    for metric in [
                        "any_hit@10",
                        "all_support_hit@10",
                        "supporting_title_recall@10",
                        "worst_supporting_title_rank",
                    ]
                },
            }
        )

    contrasts = []
    for contrast_index, (target, reference) in enumerate(PRIMARY_CONTRASTS):
        target_values = [
            metrics_by_method[target][row_id]["all_support_hit@10"] for row_id in ids
        ]
        reference_values = [
            metrics_by_method[reference][row_id]["all_support_hit@10"]
            for row_id in ids
        ]
        deltas = [target_value - reference_value for target_value, reference_value in zip(target_values, reference_values)]
        ci_low, ci_high = paired_bootstrap_ci(
            deltas, iterations=iterations, seed=seed + contrast_index
        )
        mcnemar = exact_mcnemar(reference_values, target_values)
        contrasts.append(
            {
                "dataset": dataset,
                "target": target,
                "reference": reference,
                "endpoint": "all-support hit@10",
                "n": len(ids),
                "target_value": statistics.fmean(target_values),
                "reference_value": statistics.fmean(reference_values),
                "paired_delta": statistics.fmean(deltas),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mcnemar_gains": mcnemar["gains"],
                "mcnemar_losses": mcnemar["losses"],
                "mcnemar_discordant": mcnemar["discordant_total"],
                "mcnemar_p": mcnemar["p_value"],
            }
        )
    return summaries, contrasts


def parse_scoped_paths(values: list[str]) -> dict[str, dict[str, Path]]:
    parsed = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise ValueError(f"Expected DATASET:METHOD=PATH, got: {value}")
        scope, raw_path = value.split("=", 1)
        dataset, method = scope.split(":", 1)
        if method in parsed.setdefault(dataset, {}):
            raise ValueError(f"Duplicate path specification: {dataset}:{method}")
        parsed[dataset][method] = Path(raw_path)
    return parsed


def generation_tradeoffs(
    summaries: list[dict],
    generation_paths: dict[str, dict[str, Path]],
    leakage_paths: dict[str, dict[str, Path]],
):
    summary_index = {(row["dataset"], row["method"]): row for row in summaries}
    for dataset, methods in generation_paths.items():
        for method, path in methods.items():
            rows = list(read_jsonl(path))
            index_unique(rows, f"{dataset}:{method} generations")
            lengths = [len(str(row.get("prediction", "")).split()) for row in rows]
            target = summary_index.get((dataset, method))
            if target is not None and lengths:
                target["mean_generated_words"] = statistics.fmean(lengths)
                target["median_generated_words"] = statistics.median(lengths)

    for dataset, methods in leakage_paths.items():
        for method, path in methods.items():
            rows = [
                row
                for row in read_jsonl(path)
                if row.get("condition", method) == method
            ]
            target = summary_index.get((dataset, method))
            if target is not None and rows:
                target["exact_answer_overlap_rate"] = statistics.fmean(
                    row["exact_answer_present"] for row in rows
                )
                target["any_gold_content_rate"] = statistics.fmean(
                    row["gold_content_group"] != "No identifiable gold content"
                    for row in rows
                )


def leakage_summary(
    leakage_paths: dict[str, dict[str, Path]]
) -> list[dict]:
    summaries = []
    for dataset, methods in leakage_paths.items():
        for method, path in methods.items():
            rows = [
                row
                for row in read_jsonl(path)
                if row.get("condition", method) == method
            ]
            if not rows:
                continue
            group_counts = {
                "count_exact_answer": sum(
                    row["gold_content_group"] == "Exact-answer present"
                    for row in rows
                ),
                "count_alias_proxy": sum(
                    row["gold_content_group"] == "Alias/paraphrase proxy present"
                    for row in rows
                ),
                "count_supporting_title_only": sum(
                    row["gold_content_group"]
                    == "Supporting entity/title present but answer proxy absent"
                    for row in rows
                ),
                "count_no_identifiable_gold_content": sum(
                    row["gold_content_group"] == "No identifiable gold content"
                    for row in rows
                ),
            }
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": len(rows),
                    "exact_answer_overlap_rate": statistics.fmean(
                        row["exact_answer_present"] for row in rows
                    ),
                    "alias_proxy_rate": statistics.fmean(
                        row["alias_paraphrase_proxy_present"] for row in rows
                    ),
                    "supporting_title_rate": statistics.fmean(
                        row["supporting_title_present"] for row in rows
                    ),
                    "new_supporting_title_rate": statistics.fmean(
                        row["supporting_title_not_in_question_present"] for row in rows
                    ),
                    **group_counts,
                }
            )
    return summaries


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="append", required=True)
    parser.add_argument("--generation", action="append", default=[])
    parser.add_argument("--leakage", action="append", default=[])
    parser.add_argument(
        "--out_dir", default="artifacts/summaries/expert_systems_minimum"
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    retrieval_paths = parse_scoped_paths(args.retrieval)
    generation_paths = parse_scoped_paths(args.generation)
    leakage_paths = parse_scoped_paths(args.leakage)
    all_summaries = []
    all_contrasts = []
    for dataset, methods in retrieval_paths.items():
        summaries, contrasts = analyze_dataset(
            dataset,
            {method: list(read_jsonl(path)) for method, path in methods.items()},
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        all_summaries.extend(summaries)
        all_contrasts.extend(contrasts)

    adjusted = holm_adjust([row["mcnemar_p"] for row in all_contrasts])
    for row, adjusted_p in zip(all_contrasts, adjusted):
        row["holm_adjusted_p"] = adjusted_p
    if len(retrieval_paths) == 2 and len(all_contrasts) != 8:
        raise ValueError(f"Expected eight primary contrasts, got {len(all_contrasts)}")

    generation_tradeoffs(all_summaries, generation_paths, leakage_paths)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "retrieval_summary.csv", all_summaries)
    write_csv(out_dir / "paired_contrasts.csv", all_contrasts)
    write_csv(out_dir / "leakage_summary.csv", leakage_summary(leakage_paths))
    print(f"Saved {len(all_contrasts)} primary contrasts to {out_dir}")


if __name__ == "__main__":
    main()
