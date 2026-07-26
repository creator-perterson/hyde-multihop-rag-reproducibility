import argparse
import csv
import math
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
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


def answer_absent_ids(leakage_by_id):
    return sorted(
        qid for qid, row in leakage_by_id.items()
        if int(row.get("answer_in_hyde", 0)) == 0
    )


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hit_titles = gold_titles & retrieved_titles
    return {
        "any_hit": float(bool(hit_titles)),
        "all_hit": float(gold_titles.issubset(retrieved_titles) if gold_titles else False),
        "support_recall": len(hit_titles) / len(gold_titles) if gold_titles else 0.0,
    }


def answer_metrics(row):
    gold = row.get("gold_answer", row.get("answer", ""))
    pred = row.get("prediction", row.get("final_answer", ""))
    return {
        "em": float(exact_match(pred, gold)),
        "f1": f1_score(pred, gold),
    }


def summarize_metric(rows_by_id, ids, metric_fn, key):
    return mean([metric_fn(rows_by_id[qid])[key] for qid in ids])


def paired_delta(rows_a, rows_b, ids, metric_fn, key):
    return [
        metric_fn(rows_b[qid])[key] - metric_fn(rows_a[qid])[key]
        for qid in ids
    ]


def paired_subset_summary(
    dataset,
    subset,
    ids,
    baseline_label,
    target_label,
    baseline_retrieval_by_id,
    target_retrieval_by_id,
    baseline_answers_by_id,
    target_answers_by_id,
    iterations=2000,
    seed=13,
):
    paired_ids = [
        qid for qid in ids
        if qid in baseline_retrieval_by_id
        and qid in target_retrieval_by_id
        and qid in baseline_answers_by_id
        and qid in target_answers_by_id
    ]
    support_keys = ["any_hit", "all_hit", "support_recall"]
    answer_keys = ["em", "f1"]
    row = {
        "dataset": dataset,
        "subset": subset,
        "baseline": baseline_label,
        "target": target_label,
        "n": len(paired_ids),
    }
    for key in support_keys:
        base_value = summarize_metric(baseline_retrieval_by_id, paired_ids, support_metrics, key)
        target_value = summarize_metric(target_retrieval_by_id, paired_ids, support_metrics, key)
        deltas = paired_delta(baseline_retrieval_by_id, target_retrieval_by_id, paired_ids, support_metrics, key)
        ci_low, ci_high = bootstrap_delta_ci(deltas, iterations=iterations, seed=seed + len(row))
        row[f"baseline_{key}"] = base_value
        row[f"target_{key}"] = target_value
        row[f"delta_{key}"] = mean(deltas)
        row[f"delta_{key}_ci_low"] = ci_low
        row[f"delta_{key}_ci_high"] = ci_high
    for key in answer_keys:
        base_value = summarize_metric(baseline_answers_by_id, paired_ids, answer_metrics, key)
        target_value = summarize_metric(target_answers_by_id, paired_ids, answer_metrics, key)
        deltas = paired_delta(baseline_answers_by_id, target_answers_by_id, paired_ids, answer_metrics, key)
        ci_low, ci_high = bootstrap_delta_ci(deltas, iterations=iterations, seed=seed + 100 + len(row))
        row[f"baseline_{key}"] = base_value
        row[f"target_{key}"] = target_value
        row[f"delta_{key}"] = mean(deltas)
        row[f"delta_{key}_ci_low"] = ci_low
        row[f"delta_{key}_ci_high"] = ci_high

    baseline_only_correct = 0
    target_only_correct = 0
    for qid in paired_ids:
        baseline_correct = answer_metrics(baseline_answers_by_id[qid])["em"] >= 1.0
        target_correct = answer_metrics(target_answers_by_id[qid])["em"] >= 1.0
        baseline_only_correct += int(baseline_correct and not target_correct)
        target_only_correct += int(target_correct and not baseline_correct)
    row["wrong_to_correct"] = target_only_correct
    row["correct_to_wrong"] = baseline_only_correct
    row["mcnemar_exact_p"] = mcnemar_exact_p(baseline_only_correct, target_only_correct)
    return row


def default_dataset_specs(code_root):
    results = code_root / "results"
    return [
        {
            "dataset": "HotpotQA",
            "leakage": results / "ircot_hotpotqa_test500_hyde_leakage_audit.jsonl",
            "baseline_retrieval": results / "ircot_hotpotqa_test500_top10_retrieval.jsonl",
            "target_retrieval": results / "ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl",
            "baseline_answers": results / "ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl",
            "target_answers": results / "ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
        },
        {
            "dataset": "2WikiMultihopQA",
            "leakage": results / "ircot_2wiki_test500_hyde_leakage_audit.jsonl",
            "baseline_retrieval": results / "ircot_2wiki_test500_top10_retrieval.jsonl",
            "target_retrieval": results / "ircot_2wiki_test500_hyde_top10_retrieval.jsonl",
            "baseline_answers": results / "ircot_2wiki_test500_dense_top10_extractive_answers_qwenmax_500.jsonl",
            "target_answers": results / "ircot_2wiki_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
        },
    ]


def assert_inputs_exist(specs):
    missing = []
    for spec in specs:
        for key in [
            "leakage",
            "baseline_retrieval",
            "target_retrieval",
            "baseline_answers",
            "target_answers",
        ]:
            if not Path(spec[key]).exists():
                missing.append(str(spec[key]))
    if missing:
        raise FileNotFoundError("Missing input artifacts:\n" + "\n".join(missing))


def run_analysis(specs, iterations=2000, seed=13):
    rows = []
    for offset, spec in enumerate(specs):
        leakage_by_id = load_by_id(spec["leakage"])
        ids = answer_absent_ids(leakage_by_id)
        rows.append(
            paired_subset_summary(
                dataset=spec["dataset"],
                subset="answer_not_in_hyde",
                ids=ids,
                baseline_label="Dense RAG",
                target_label="HyDE-style RAG",
                baseline_retrieval_by_id=load_by_id(spec["baseline_retrieval"]),
                target_retrieval_by_id=load_by_id(spec["target_retrieval"]),
                baseline_answers_by_id=load_by_id(spec["baseline_answers"]),
                target_answers_by_id=load_by_id(spec["target_answers"]),
                iterations=iterations,
                seed=seed + offset * 100,
            )
        )
    return rows


def format_float(value):
    return f"{value:.4f}"


def format_p(value):
    if value < 0.0001:
        return "$<0.0001$"
    return format_float(value)


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
        "# Answer-absent Subset Paired Analysis",
        "",
        "This diagnostic restricts each dataset to examples where the normalized gold-answer string does not appear in the generated HyDE hypothetical passage. It reuses existing retrieval and reader answer artifacts; no new LLM calls are made.",
        "",
        "| Dataset | N | Dense all-hit | HyDE all-hit | Dense recall | HyDE recall | Dense F1 | HyDE F1 | Delta F1 [95% CI] | W->C | C->W | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['n']} | "
            f"{format_float(row['baseline_all_hit'])} | {format_float(row['target_all_hit'])} | "
            f"{format_float(row['baseline_support_recall'])} | {format_float(row['target_support_recall'])} | "
            f"{format_float(row['baseline_f1'])} | {format_float(row['target_f1'])} | "
            f"{format_float(row['delta_f1'])} [{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}] | "
            f"{row['wrong_to_correct']} | {row['correct_to_wrong']} | {format_float(row['mcnemar_exact_p'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A positive HyDE-over-Dense difference on this subset weakens the strict explanation that the observed HyDE gain is solely due to exact answer-string overlap in the hypothetical passage. This diagnostic does not prove the absence of parametric memory, paraphrased answer leakage, or benchmark contamination; it only removes exact normalized answer-string overlap from the analyzed subset.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Answer-absent subset paired analysis.}",
        "\\label{tab:answer_absent_subset}",
        "\\scriptsize",
        "\\begin{threeparttable}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X r r r r r r r c c c}",
        "\\toprule",
        "Dataset & $N$ & Dense all-hit & HyDE all-hit & Dense recall & HyDE recall & Dense F1 & HyDE F1 & $\\Delta$F1 [95\\% CI] & W$\\rightarrow$C / C$\\rightarrow$W & McNemar $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['n']} & "
            f"{format_float(row['baseline_all_hit'])} & {format_float(row['target_all_hit'])} & "
            f"{format_float(row['baseline_support_recall'])} & {format_float(row['target_support_recall'])} & "
            f"{format_float(row['baseline_f1'])} & {format_float(row['target_f1'])} & "
            f"{format_float(row['delta_f1'])} [{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}] & "
            f"{row['wrong_to_correct']} / {row['correct_to_wrong']} & {format_p(row['mcnemar_exact_p'])} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item The subset contains examples where the normalized gold-answer string is absent from the generated HyDE hypothetical passage. All rows reuse frozen retrieval and reader answer artifacts. The diagnostic reduces the exact answer-string overlap explanation, but it does not rule out paraphrased answer leakage or model memorization.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out_csv", default="local-artifacts/answer_absent_subset_paired_ci.csv")
    parser.add_argument("--out_md", default="local-artifacts/answer_absent_subset_paired_ci.md")
    parser.add_argument("--out_tex", default="paper/latex/table_answer_absent_subset.tex")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    specs = default_dataset_specs(code_root)
    assert_inputs_exist(specs)
    rows = run_analysis(specs, iterations=args.iterations, seed=args.seed)

    write_csv(code_root / args.out_csv, rows)
    write_markdown(code_root / args.out_md, rows)
    write_latex(code_root / args.out_tex, rows)
    for row in rows:
        print(
            f"{row['dataset']}: n={row['n']} "
            f"Dense F1={format_float(row['baseline_f1'])} "
            f"HyDE F1={format_float(row['target_f1'])} "
            f"Delta F1={format_float(row['delta_f1'])} "
            f"CI=[{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}]"
        )


if __name__ == "__main__":
    main()
