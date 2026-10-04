from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ircot_fullwiki.fullwiki_manifest import read_jsonl


METRIC_KEYS = ("any_hit@10", "all_support_hit@10", "supporting_title_recall@10")


def normalize_title(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def support_metrics(row: dict) -> dict:
    gold_titles = [normalize_title(title) for title in row.get("supporting_facts", {}).get("title", [])]
    rank_by_title = {}
    for fallback_rank, document in enumerate(row.get("retrieved", [])[:10], start=1):
        title = normalize_title(document.get("title", ""))
        rank = int(document.get("rank", fallback_rank))
        if title and title not in rank_by_title:
            rank_by_title[title] = rank
    found_ranks = [rank_by_title[title] for title in gold_titles if title in rank_by_title]
    gold_count = len(gold_titles)
    all_found = gold_count > 0 and len(found_ranks) == gold_count
    return {
        "any_hit@10": float(bool(found_ranks)),
        "all_support_hit@10": float(all_found),
        "supporting_title_recall@10": len(found_ranks) / gold_count if gold_count else 0.0,
        "worst_supporting_title_rank": max(found_ranks) if all_found else None,
    }


def condition_summary(rows: list[dict], condition: str) -> dict:
    if not rows:
        raise ValueError("condition summary requires at least one row")
    metrics = [support_metrics(row) for row in rows]
    count = len(rows)
    return {
        "dataset": "hotpotqa",
        "condition": condition,
        "n": count,
        **{
            key: sum(float(item[key]) for item in metrics) / count
            for key in METRIC_KEYS
        },
        "mean_retrieval_calls": sum(int(row.get("retrieval_calls", 0)) for row in rows) / count,
        "mean_generation_calls": sum(int(row.get("generation_calls", 0)) for row in rows) / count,
        "mean_generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in rows) / count,
        "total_generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in rows),
        "hard_cap_count": sum(int(row.get("generation_calls", 0)) >= 10 for row in rows),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires non-empty values")
    index = int(round(probability * (len(ordered) - 1)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def paired_bootstrap(
    baseline: Sequence[float],
    target: Sequence[float],
    *,
    seed: int = 13,
    n_resamples: int = 10_000,
) -> dict:
    if len(baseline) != len(target) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    deltas = [float(right) - float(left) for left, right in zip(baseline, target)]
    rng = random.Random(seed)
    estimates = []
    n = len(deltas)
    for _ in range(n_resamples):
        estimates.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    return {
        "delta": sum(deltas) / n,
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def mcnemar_exact(baseline_only: int, target_only: int) -> float:
    discordant = baseline_only + target_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, target_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * float(p_values[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def index_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = str(row["id"])
        if identifier in result:
            raise ValueError(f"duplicate row ID: {identifier}")
        result[identifier] = row
    return result


def paired_comparison(
    baseline_rows: list[dict],
    target_rows: list[dict],
    baseline_condition: str,
    target_condition: str,
    *,
    seed: int = 13,
    n_resamples: int = 10_000,
) -> dict:
    baseline = index_rows(baseline_rows)
    target = index_rows(target_rows)
    if set(baseline) != set(target):
        raise ValueError("paired conditions do not contain identical IDs")
    identifiers = [str(row["id"]) for row in baseline_rows]
    output = {
        "baseline": baseline_condition,
        "target": target_condition,
        "n": len(identifiers),
    }
    metrics_by_condition = {
        "baseline": [support_metrics(baseline[identifier]) for identifier in identifiers],
        "target": [support_metrics(target[identifier]) for identifier in identifiers],
    }
    for offset, key in enumerate(METRIC_KEYS):
        left = [item[key] for item in metrics_by_condition["baseline"]]
        right = [item[key] for item in metrics_by_condition["target"]]
        result = paired_bootstrap(left, right, seed=seed + offset, n_resamples=n_resamples)
        prefix = key.replace("@", "_at_")
        output[f"baseline_{prefix}"] = sum(left) / len(left)
        output[f"target_{prefix}"] = sum(right) / len(right)
        output[f"delta_{prefix}"] = result["delta"]
        output[f"delta_{prefix}_ci_low"] = result["ci_low"]
        output[f"delta_{prefix}_ci_high"] = result["ci_high"]

    baseline_binary = [bool(item["all_support_hit@10"]) for item in metrics_by_condition["baseline"]]
    target_binary = [bool(item["all_support_hit@10"]) for item in metrics_by_condition["target"]]
    output["baseline_only_all_support"] = sum(left and not right for left, right in zip(baseline_binary, target_binary))
    output["target_only_all_support"] = sum(right and not left for left, right in zip(baseline_binary, target_binary))
    output["mcnemar_exact_p"] = mcnemar_exact(
        output["baseline_only_all_support"], output["target_only_all_support"]
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", type=Path, required=True)
    parser.add_argument("--hyde", type=Path, required=True)
    parser.add_argument("--ircot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    conditions = {
        "fullwiki_one_retrieval_bm25": read_jsonl(args.one),
        "fullwiki_hyde_bm25": read_jsonl(args.hyde),
        "fullwiki_ircot_qwen": read_jsonl(args.ircot),
    }
    pairs = [
        ("fullwiki_one_retrieval_bm25", "fullwiki_ircot_qwen"),
        ("fullwiki_hyde_bm25", "fullwiki_ircot_qwen"),
        ("fullwiki_one_retrieval_bm25", "fullwiki_hyde_bm25"),
    ]
    summaries = [
        paired_comparison(
            conditions[left],
            conditions[right],
            left,
            right,
            seed=args.seed + offset * 10,
            n_resamples=args.resamples,
        )
        for offset, (left, right) in enumerate(pairs)
    ]
    adjusted = holm_adjust([row["mcnemar_exact_p"] for row in summaries])
    for row, value in zip(summaries, adjusted):
        row["mcnemar_holm_p"] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_summaries = [
        condition_summary(rows, condition) for condition, rows in conditions.items()
    ]
    (args.output_dir / "condition_summary.json").write_text(
        json.dumps(condition_summaries, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "condition_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(condition_summaries[0]))
        writer.writeheader()
        writer.writerows(condition_summaries)
    (args.output_dir / "paired_comparisons.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
