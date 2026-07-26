import argparse
import csv
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.analyze_retrieval_answer_correlation import mcnemar_exact_p
from utils import read_jsonl


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    index = int(round(q * (len(sorted_values) - 1)))
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def bootstrap_delta_ci(deltas, iterations=2000, seed=13, alpha=0.05):
    deltas = list(deltas)
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        estimates.append(mean(sample))
    estimates.sort()
    return percentile(estimates, alpha / 2), percentile(estimates, 1 - alpha / 2)


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hit_titles = gold_titles & retrieved_titles
    return {
        "any_hit": float(bool(hit_titles)),
        "all_hit": float(bool(gold_titles) and gold_titles.issubset(retrieved_titles)),
        "support_recall": len(hit_titles) / len(gold_titles) if gold_titles else 0.0,
    }


def metric_values(rows_by_id, ids, key):
    return [support_metrics(rows_by_id[qid])[key] for qid in ids]


def paired_retrieval_summary(
    dataset,
    baseline_label,
    target_label,
    baseline_by_id,
    target_by_id,
    iterations=2000,
    seed=13,
):
    shared_ids = sorted(set(baseline_by_id) & set(target_by_id))
    row = {
        "dataset": dataset,
        "baseline": baseline_label,
        "target": target_label,
        "n": len(shared_ids),
    }

    for offset, key in enumerate(["any_hit", "all_hit", "support_recall"]):
        baseline_values = metric_values(baseline_by_id, shared_ids, key)
        target_values = metric_values(target_by_id, shared_ids, key)
        deltas = [target - baseline for baseline, target in zip(baseline_values, target_values)]
        ci_low, ci_high = bootstrap_delta_ci(deltas, iterations=iterations, seed=seed + offset)
        row[f"baseline_{key}"] = mean(baseline_values)
        row[f"target_{key}"] = mean(target_values)
        row[f"delta_{key}"] = mean(deltas)
        row[f"delta_{key}_ci_low"] = ci_low
        row[f"delta_{key}_ci_high"] = ci_high

    both_full = 0
    baseline_only_full = 0
    target_only_full = 0
    neither_full = 0
    for qid in shared_ids:
        baseline_full = support_metrics(baseline_by_id[qid])["all_hit"] >= 1.0
        target_full = support_metrics(target_by_id[qid])["all_hit"] >= 1.0
        if baseline_full and target_full:
            both_full += 1
        elif baseline_full and not target_full:
            baseline_only_full += 1
        elif target_full and not baseline_full:
            target_only_full += 1
        else:
            neither_full += 1

    row["both_full"] = both_full
    row["baseline_only_full"] = baseline_only_full
    row["target_only_full"] = target_only_full
    row["neither_full"] = neither_full
    row["mcnemar_exact_p"] = mcnemar_exact_p(baseline_only_full, target_only_full)
    return row


def default_comparisons(code_root):
    results = code_root / "results"
    hyde_hotpot = results / "ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl"
    hyde_2wiki = results / "ircot_2wiki_test500_hyde_top10_retrieval.jsonl"
    return [
        {
            "dataset": "HotpotQA",
            "baseline": "Dense RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_hotpotqa_test500_top10_retrieval.jsonl",
            "target_path": hyde_hotpot,
        },
        {
            "dataset": "HotpotQA",
            "baseline": "Single-query Reformulation RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_hotpotqa_test500_single_query_reformulation_top10_retrieval.jsonl",
            "target_path": hyde_hotpot,
        },
        {
            "dataset": "HotpotQA",
            "baseline": "Question + rewritten query",
            "target": "Question + hypothetical passage",
            "baseline_path": results / "ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_retrieval.jsonl",
            "target_path": hyde_hotpot,
        },
        {
            "dataset": "HotpotQA",
            "baseline": "Hybrid RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_hotpotqa_test500_hybrid_top10_retrieval.jsonl",
            "target_path": hyde_hotpot,
        },
        {
            "dataset": "HotpotQA",
            "baseline": "Iterative RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl",
            "target_path": hyde_hotpot,
        },
        {
            "dataset": "2WikiMultihopQA",
            "baseline": "Dense RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_2wiki_test500_top10_retrieval.jsonl",
            "target_path": hyde_2wiki,
        },
        {
            "dataset": "2WikiMultihopQA",
            "baseline": "BM25 RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_2wiki_test500_bm25_top10_retrieval.jsonl",
            "target_path": hyde_2wiki,
        },
        {
            "dataset": "2WikiMultihopQA",
            "baseline": "Hybrid RAG",
            "target": "HyDE-style RAG",
            "baseline_path": results / "ircot_2wiki_test500_hybrid_top10_retrieval.jsonl",
            "target_path": hyde_2wiki,
        },
    ]


def assert_inputs_exist(comparisons):
    missing = []
    for comparison in comparisons:
        for key in ["baseline_path", "target_path"]:
            if not Path(comparison[key]).exists():
                missing.append(str(comparison[key]))
    if missing:
        raise FileNotFoundError("Missing input artifacts:\n" + "\n".join(missing))


def run_analysis(comparisons, iterations=2000, seed=13):
    rows = []
    cache = {}
    for offset, comparison in enumerate(comparisons):
        baseline_path = Path(comparison["baseline_path"])
        target_path = Path(comparison["target_path"])
        if baseline_path not in cache:
            cache[baseline_path] = load_by_id(baseline_path)
        if target_path not in cache:
            cache[target_path] = load_by_id(target_path)
        rows.append(
            paired_retrieval_summary(
                dataset=comparison["dataset"],
                baseline_label=comparison["baseline"],
                target_label=comparison["target"],
                baseline_by_id=cache[baseline_path],
                target_by_id=cache[target_path],
                iterations=iterations,
                seed=seed + offset * 10,
            )
        )
    return rows


def format_float(value):
    return f"{value:.4f}"


def format_p(value):
    if value < 0.0001:
        return "$<0.0001$"
    return format_float(value)


def latex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Paired Tests",
        "",
        "This no-new-LLM analysis compares retrieval metrics over paired examples using frozen top-10 retrieval artifacts. Bootstrap intervals are computed over per-example paired deltas. McNemar's exact test is computed on full-support hit@10 transitions.",
        "",
        "| Dataset | Baseline | Target | n | Baseline all-hit | Target all-hit | Delta all-hit [95% CI] | Baseline recall | Target recall | Delta recall [95% CI] | Full W->C / C->W | McNemar p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['baseline']} | {row['target']} | {row['n']} | "
            f"{format_float(row['baseline_all_hit'])} | {format_float(row['target_all_hit'])} | "
            f"{format_float(row['delta_all_hit'])} [{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}] | "
            f"{format_float(row['baseline_support_recall'])} | {format_float(row['target_support_recall'])} | "
            f"{format_float(row['delta_support_recall'])} [{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}] | "
            f"{row['target_only_full']} / {row['baseline_only_full']} | {format_p(row['mcnemar_exact_p']).replace('$', '')} |"
        )
    lines.extend([
        "",
        "W->C means the target retrieves all supporting titles when the baseline does not; C->W means the reverse.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Paired retrieval-side reliability tests. All comparisons use the same 500 paired examples per dataset. Intervals are paired bootstrap 95\\% confidence intervals over per-example deltas. McNemar's exact test is computed on all-support hit@10 transitions; W$\\rightarrow$C denotes target-only full-support retrieval and C$\\rightarrow$W denotes baseline-only full-support retrieval.}",
        "\\label{tab:retrieval_paired_tests}",
        "\\begin{tabularx}{\\textwidth}{lXrrrr}",
        "\\toprule",
        "Dataset & Baseline $\\rightarrow$ target & $\\Delta$ All@10 & $\\Delta$ Recall@10 & W$\\rightarrow$C / C$\\rightarrow$W & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        comparison = f"{latex_escape(row['baseline'])} $\\rightarrow$ {latex_escape(row['target'])}"
        lines.append(
            f"{latex_escape(row['dataset'])} & {comparison} & "
            f"{format_float(row['delta_all_hit'])} [{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}] & "
            f"{format_float(row['delta_support_recall'])} [{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}] & "
            f"{row['target_only_full']} / {row['baseline_only_full']} & "
            f"{format_p(row['mcnemar_exact_p'])} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--code-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    code_root = Path(args.code_root)
    comparisons = default_comparisons(code_root)
    assert_inputs_exist(comparisons)
    rows = run_analysis(comparisons, iterations=args.iterations, seed=args.seed)

    project_root = code_root.parent
    write_csv(project_root / "local-artifacts" / "retrieval_paired_tests.csv", rows)
    write_markdown(project_root / "local-artifacts" / "retrieval_paired_tests.md", rows)
    write_latex(project_root / "paper/latex" / "table_retrieval_paired_tests.tex", rows)

    for row in rows:
        print(
            f"{row['dataset']} {row['baseline']} -> {row['target']}: "
            f"delta all-hit={format_float(row['delta_all_hit'])} "
            f"[{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}], "
            f"delta recall={format_float(row['delta_support_recall'])} "
            f"[{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}], "
            f"W->C/C->W={row['target_only_full']}/{row['baseline_only_full']}, "
            f"p={format_float(row['mcnemar_exact_p']) if row['mcnemar_exact_p'] >= 0.0001 else '<0.0001'}"
        )


if __name__ == "__main__":
    main()
