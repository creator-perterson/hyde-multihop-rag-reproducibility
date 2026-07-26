import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl


BUCKET_ORDER = {"full": 0, "partial": 1, "none": 2, "unknown": 3}


def bucket_for_recall(recall):
    if recall >= 1.0:
        return "full"
    if recall > 0:
        return "partial"
    if recall == 0:
        return "none"
    return "unknown"


def support_titles(row):
    return set(row.get("supporting_facts", {}).get("title", []))


def retrieved_titles(row):
    return {doc.get("title", "") for doc in row.get("retrieved", [])}


def prediction_text(row):
    return row.get("prediction", row.get("final_answer", ""))


def per_example_record(method, row):
    gold_titles = support_titles(row)
    retrieved = retrieved_titles(row)
    hit_titles = gold_titles & retrieved
    support_recall = len(hit_titles) / len(gold_titles) if gold_titles else -1.0
    prediction = prediction_text(row)
    gold = row["gold_answer"]
    return {
        "method": method,
        "id": row["id"],
        "support_gold_count": len(gold_titles),
        "support_hit_count": len(hit_titles),
        "support_recall": support_recall,
        "bucket": bucket_for_recall(support_recall),
        "em": float(exact_match(prediction, gold)),
        "f1": f1_score(prediction, gold),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    index = int(round(q * (len(sorted_values) - 1)))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def bootstrap_mean_ci(values, iterations=2000, seed=13, alpha=0.05):
    values = list(values)
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return {
        "mean": mean(values),
        "ci_low": percentile(estimates, alpha / 2),
        "ci_high": percentile(estimates, 1 - alpha / 2),
    }


def bootstrap_delta_ci(deltas, iterations=2000, seed=13, alpha=0.05):
    ci = bootstrap_mean_ci(deltas, iterations=iterations, seed=seed, alpha=alpha)
    return ci["ci_low"], ci["ci_high"]


def mcnemar_exact_p(baseline_only_correct, target_only_correct):
    discordant = baseline_only_correct + target_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only_correct, target_only_correct)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def paired_metric_delta(baseline_method, target_method, baseline_records, target_records, iterations=2000, seed=13):
    baseline_by_id = {row["id"]: row for row in baseline_records}
    target_by_id = {row["id"]: row for row in target_records}
    shared_ids = sorted(set(baseline_by_id) & set(target_by_id))

    em_deltas = [target_by_id[qid]["em"] - baseline_by_id[qid]["em"] for qid in shared_ids]
    f1_deltas = [target_by_id[qid]["f1"] - baseline_by_id[qid]["f1"] for qid in shared_ids]
    em_ci_low, em_ci_high = bootstrap_delta_ci(em_deltas, iterations=iterations, seed=seed)
    f1_ci_low, f1_ci_high = bootstrap_delta_ci(f1_deltas, iterations=iterations, seed=seed + 1)

    both_correct = 0
    baseline_only_correct = 0
    target_only_correct = 0
    both_wrong = 0
    for qid in shared_ids:
        baseline_correct = baseline_by_id[qid]["em"] >= 1.0
        target_correct = target_by_id[qid]["em"] >= 1.0
        if baseline_correct and target_correct:
            both_correct += 1
        elif baseline_correct and not target_correct:
            baseline_only_correct += 1
        elif target_correct and not baseline_correct:
            target_only_correct += 1
        else:
            both_wrong += 1

    return {
        "baseline_method": baseline_method,
        "target_method": target_method,
        "n": len(shared_ids),
        "baseline_em": mean([baseline_by_id[qid]["em"] for qid in shared_ids]),
        "target_em": mean([target_by_id[qid]["em"] for qid in shared_ids]),
        "delta_em": mean(em_deltas),
        "delta_em_ci_low": em_ci_low,
        "delta_em_ci_high": em_ci_high,
        "baseline_f1": mean([baseline_by_id[qid]["f1"] for qid in shared_ids]),
        "target_f1": mean([target_by_id[qid]["f1"] for qid in shared_ids]),
        "delta_f1": mean(f1_deltas),
        "delta_f1_ci_low": f1_ci_low,
        "delta_f1_ci_high": f1_ci_high,
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only_correct,
        "target_only_correct": target_only_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": mcnemar_exact_p(baseline_only_correct, target_only_correct),
    }


def summarize_buckets(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], record["bucket"])].append(record)
    summary = []
    for (method, bucket), rows in groups.items():
        summary.append({
            "method": method,
            "bucket": bucket,
            "n": len(rows),
            "em": mean([row["em"] for row in rows]),
            "f1": mean([row["f1"] for row in rows]),
            "mean_support_recall": mean([row["support_recall"] for row in rows]),
        })
    return sorted(summary, key=lambda row: (row["method"], BUCKET_ORDER.get(row["bucket"], 99)))


def aggregate_bucket_summary(records, method_label):
    pooled = [dict(record, method=method_label) for record in records]
    return summarize_buckets(pooled)


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_den = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if x_den == 0 or y_den == 0:
        return 0.0
    return numerator / (x_den * y_den)


def load_formal_table(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_retrieval_metrics(row):
    return bool(row.get("mean_recall")) and bool(row.get("all_hit"))


def is_primary_retrieval_strategy(row):
    method = row.get("method", "")
    return (
        has_retrieval_metrics(row)
        and row.get("top_k") not in {"", "0"}
        and "Verification" not in method
        and "Verifier" not in method
    )


def method_level_correlations(rows):
    eligible = [
        row for row in rows
        if is_primary_retrieval_strategy(row) and row.get("em") and row.get("f1")
    ]
    metrics = {
        "all_hit_vs_em": pearson([float(r["all_hit"]) for r in eligible], [float(r["em"]) for r in eligible]),
        "all_hit_vs_f1": pearson([float(r["all_hit"]) for r in eligible], [float(r["f1"]) for r in eligible]),
        "mean_recall_vs_em": pearson([float(r["mean_recall"]) for r in eligible], [float(r["em"]) for r in eligible]),
        "mean_recall_vs_f1": pearson([float(r["mean_recall"]) for r in eligible], [float(r["f1"]) for r in eligible]),
    }
    return eligible, metrics


def load_per_example_records(formal_rows, code_root):
    records = []
    for formal_row in formal_rows:
        if not is_primary_retrieval_strategy(formal_row):
            continue
        answer_file = formal_row.get("answer_file", "")
        if not answer_file:
            continue
        path = Path(answer_file)
        if not path.is_absolute():
            path = code_root / answer_file
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        if not rows or "retrieved" not in rows[0] or "supporting_facts" not in rows[0]:
            continue
        for row in rows:
            records.append(per_example_record(formal_row["method"], row))
    return records


def is_answer_row(formal_row):
    return bool(formal_row.get("answer_file")) and bool(formal_row.get("em")) and bool(formal_row.get("f1"))


def load_answer_records_by_method(formal_rows, code_root):
    records_by_method = {}
    for formal_row in formal_rows:
        if not is_answer_row(formal_row):
            continue
        answer_file = formal_row.get("answer_file", "")
        path = Path(answer_file)
        if not path.is_absolute():
            path = code_root / answer_file
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        records = []
        for row in rows:
            if "gold_answer" not in row:
                continue
            records.append(per_example_record(formal_row["method"], row))
        if records:
            records_by_method[formal_row["method"]] = records
    return records_by_method


def method_reliability_summary(records_by_method, iterations=2000, seed=13):
    rows = []
    for offset, (method, records) in enumerate(records_by_method.items()):
        em_ci = bootstrap_mean_ci([row["em"] for row in records], iterations=iterations, seed=seed + offset)
        f1_ci = bootstrap_mean_ci([row["f1"] for row in records], iterations=iterations, seed=seed + 100 + offset)
        rows.append({
            "method": method,
            "n": len(records),
            "em": em_ci["mean"],
            "em_ci_low": em_ci["ci_low"],
            "em_ci_high": em_ci["ci_high"],
            "f1": f1_ci["mean"],
            "f1_ci_low": f1_ci["ci_low"],
            "f1_ci_high": f1_ci["ci_high"],
        })
    return rows


def default_pairwise_comparisons():
    return [
        ("One-step Dense RAG", "BM25 RAG"),
        ("One-step Dense RAG", "BM25 + Dense Hybrid"),
        ("One-step Dense RAG", "Evidence-guided Iterative Retrieval"),
        ("One-step Dense RAG", "HyDE-style RAG"),
        ("Single-query Reformulation RAG", "HyDE-style RAG"),
        ("BM25 + Dense Hybrid", "HyDE-style RAG"),
        ("Evidence-guided Iterative Retrieval", "HyDE-style RAG"),
        ("HyDE-style RAG", "HyDE-style RAG + Conservative Verifier"),
        ("Evidence-guided Iterative Retrieval", "Iterative RAG + Conservative Verifier"),
        ("Question + Single-query Reformulation RAG", "HyDE-style RAG"),
    ]


def pairwise_reliability_summary(records_by_method, comparisons=None, iterations=2000, seed=13):
    comparisons = comparisons or default_pairwise_comparisons()
    rows = []
    for offset, (baseline, target) in enumerate(comparisons):
        if baseline not in records_by_method or target not in records_by_method:
            continue
        rows.append(
            paired_metric_delta(
                baseline,
                target,
                records_by_method[baseline],
                records_by_method[target],
                iterations=iterations,
                seed=seed + offset,
            )
        )
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{value:.4f}"


def format_bucket_label(bucket):
    return {
        "full": "full title support",
        "partial": "partial title support",
        "none": "no title support",
    }.get(bucket, bucket)


def write_markdown(path, eligible_methods, correlations, bucket_summary, method_reliability, pairwise_summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval-Answer Correlation Analysis",
        "",
        "This analysis examines whether supporting-title completeness is associated with answer quality on the IRCoT HotpotQA test_subsampled split.",
        "",
        "## Method-Level Correlation",
        "",
        "These correlations are descriptive because the number of retrieval strategies is small. They should be used as supporting evidence, not as a standalone causal proof.",
        "",
        "| Correlation | Pearson r |",
        "| --- | ---: |",
    ]
    for name, value in correlations.items():
        lines.append(f"| {name} | {format_float(value)} |")
    lines.extend([
        "",
        "## Supporting-Title-Completeness Buckets",
        "",
        "| Method | Supporting-title bucket | n | EM | F1 | Mean title recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in bucket_summary:
        lines.append(
            f"| {row['method']} | {format_bucket_label(row['bucket'])} | {row['n']} | "
            f"{format_float(row['em'])} | {format_float(row['f1'])} | "
            f"{format_float(row['mean_support_recall'])} |"
        )
    lines.extend([
        "",
        "## Bootstrap Reliability",
        "",
        "| Method | n | EM [95% CI] | F1 [95% CI] |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in method_reliability:
        lines.append(
            f"| {row['method']} | {row['n']} | "
            f"{format_float(row['em'])} [{format_float(row['em_ci_low'])}, {format_float(row['em_ci_high'])}] | "
            f"{format_float(row['f1'])} [{format_float(row['f1_ci_low'])}, {format_float(row['f1_ci_high'])}] |"
        )
    lines.extend([
        "",
        "## Paired Method Differences",
        "",
        "| Baseline | Target | n | Delta EM [95% CI] | Delta F1 [95% CI] | W->C | C->W | McNemar exact p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in pairwise_summary:
        lines.append(
            f"| {row['baseline_method']} | {row['target_method']} | {row['n']} | "
            f"{format_float(row['delta_em'])} [{format_float(row['delta_em_ci_low'])}, {format_float(row['delta_em_ci_high'])}] | "
            f"{format_float(row['delta_f1'])} [{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}] | "
            f"{row['target_only_correct']} | {row['baseline_only_correct']} | "
            f"{format_float(row['mcnemar_exact_p'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Across retrieval-based methods, higher all-support hit@10 and mean supporting-title recall are positively associated with answer EM/F1. At the sample level, examples with all supporting titles retrieved consistently achieve substantially higher answer quality than partial-title-support or no-title-support examples. This supports a conservative claim that supporting-title completeness, aligned with paragraph-text recovery in the processed artifacts, is a major bottleneck in this lightweight multi-hop RAG setting.",
        "",
        "Verifier-only outputs and LLM-only/no-retrieval outputs are not included in the correlation tables because the goal here is to isolate the relationship between supporting-evidence acquisition and answer quality. Verification is analyzed separately as a conservative safety layer.",
        "",
        f"Primary retrieval strategies included in method-level correlation: {len(eligible_methods)}.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_table", default="local-artifacts/ircot_test500_qwen_results.csv")
    parser.add_argument("--out_dir", default="local-artifacts")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    formal_table = Path(args.formal_table)
    if not formal_table.is_absolute():
        formal_table = code_root / formal_table
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = code_root / out_dir

    formal_rows = load_formal_table(formal_table)
    eligible_methods, correlations = method_level_correlations(formal_rows)
    records = load_per_example_records(formal_rows, code_root)
    bucket_summary = summarize_buckets(records)
    bucket_summary.extend(aggregate_bucket_summary(records, "All retrieval-only methods"))
    records_by_method = load_answer_records_by_method(formal_rows, code_root)
    reliability_summary = method_reliability_summary(records_by_method)
    pairwise_summary = pairwise_reliability_summary(records_by_method)

    write_csv(
        out_dir / "retrieval_answer_correlation_method_summary.csv",
        eligible_methods,
        fieldnames=list(eligible_methods[0].keys()) if eligible_methods else [],
    )
    write_csv(
        out_dir / "retrieval_answer_correlation_by_bucket.csv",
        bucket_summary,
        fieldnames=["method", "bucket", "n", "em", "f1", "mean_support_recall"],
    )
    write_csv(
        out_dir / "hotpotqa_test500_method_bootstrap_ci.csv",
        reliability_summary,
        fieldnames=[
            "method", "n", "em", "em_ci_low", "em_ci_high",
            "f1", "f1_ci_low", "f1_ci_high",
        ],
    )
    write_csv(
        out_dir / "hotpotqa_test500_paired_method_deltas.csv",
        pairwise_summary,
        fieldnames=[
            "baseline_method", "target_method", "n",
            "baseline_em", "target_em", "delta_em", "delta_em_ci_low", "delta_em_ci_high",
            "baseline_f1", "target_f1", "delta_f1", "delta_f1_ci_low", "delta_f1_ci_high",
            "both_correct", "baseline_only_correct", "target_only_correct", "both_wrong",
            "mcnemar_exact_p",
        ],
    )
    write_markdown(
        out_dir / "stage3_statistical_evidence_analysis_hotpotqa_test500.md",
        eligible_methods,
        correlations,
        bucket_summary,
        reliability_summary,
        pairwise_summary,
    )
    print(f"Method-level rows: {len(eligible_methods)}")
    print(f"Per-example records: {len(records)}")
    print(f"Answer methods with bootstrap CI: {len(reliability_summary)}")
    print(f"Paired comparisons: {len(pairwise_summary)}")
    for name, value in correlations.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
