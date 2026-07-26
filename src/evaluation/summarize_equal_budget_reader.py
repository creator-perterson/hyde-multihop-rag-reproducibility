import argparse
import csv
import math
import random
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl


DEFAULT_DATASETS = [
    ("HotpotQA", "ircot_hotpotqa_test500_equal_budget_bge_base"),
    ("2WikiMultihopQA", "ircot_2wiki_test500_equal_budget_bge_base"),
]

DEFAULT_MODES = [
    "keyword_expansion",
    "direct_rewrite",
    "question_decomposition",
    "document_like_passage",
]

DEFAULT_READERS = [
    ("qwenmax", "qwen3.7-max"),
    ("qwenturbo", "qwen-turbo"),
]


def read_answer_rows(path, expected_n):
    rows = list(read_jsonl(path))
    if len(rows) != expected_n:
        raise RuntimeError(f"{path}: expected {expected_n} rows, found {len(rows)}")
    return rows


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


def mcnemar_exact_p(baseline_only_correct, target_only_correct):
    discordant = baseline_only_correct + target_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only_correct, target_only_correct)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def summarize_rows(rows):
    em_scores = [float(exact_match(row["prediction"], row["gold_answer"])) for row in rows]
    f1_scores = [f1_score(row["prediction"], row["gold_answer"]) for row in rows]
    return {
        "n": len(rows),
        "em": mean(em_scores),
        "f1": mean(f1_scores),
    }


def load_by_id(rows):
    return {row["id"]: row for row in rows}


def answer_metrics(row):
    return {
        "em": float(exact_match(row["prediction"], row["gold_answer"])),
        "f1": f1_score(row["prediction"], row["gold_answer"]),
    }


def paired_contrast(
    dataset,
    reader_label,
    baseline_mode,
    target_mode,
    loaded,
    iterations=2000,
    seed=13,
):
    baseline = loaded[(dataset, reader_label, baseline_mode)]
    target = loaded[(dataset, reader_label, target_mode)]
    ids = sorted(set(baseline) & set(target))
    if not ids:
        raise RuntimeError(f"No paired ids for {dataset} {reader_label} {baseline_mode}->{target_mode}")
    delta_em = []
    delta_f1 = []
    both_correct = 0
    baseline_only_correct = 0
    target_only_correct = 0
    both_wrong = 0
    for qid in ids:
        base_metrics = answer_metrics(baseline[qid])
        target_metrics = answer_metrics(target[qid])
        delta_em.append(target_metrics["em"] - base_metrics["em"])
        delta_f1.append(target_metrics["f1"] - base_metrics["f1"])
        baseline_correct = base_metrics["em"] >= 1.0
        target_correct = target_metrics["em"] >= 1.0
        if baseline_correct and target_correct:
            both_correct += 1
        elif baseline_correct and not target_correct:
            baseline_only_correct += 1
        elif target_correct and not baseline_correct:
            target_only_correct += 1
        else:
            both_wrong += 1
    delta_em_ci_low, delta_em_ci_high = bootstrap_delta_ci(
        delta_em, iterations=iterations, seed=seed
    )
    delta_f1_ci_low, delta_f1_ci_high = bootstrap_delta_ci(
        delta_f1, iterations=iterations, seed=seed + 1
    )
    return {
        "dataset": dataset,
        "reader": reader_label,
        "baseline": baseline_mode,
        "target": target_mode,
        "n": len(ids),
        "delta_em": mean(delta_em),
        "delta_em_ci_low": delta_em_ci_low,
        "delta_em_ci_high": delta_em_ci_high,
        "delta_f1": mean(delta_f1),
        "delta_f1_ci_low": delta_f1_ci_low,
        "delta_f1_ci_high": delta_f1_ci_high,
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only_correct,
        "target_only_correct": target_only_correct,
        "both_wrong": both_wrong,
        "wrong_to_correct": target_only_correct,
        "correct_to_wrong": baseline_only_correct,
        "mcnemar_exact_p": mcnemar_exact_p(baseline_only_correct, target_only_correct),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    value = float(value)
    if abs(value) < 0.00005:
        value = 0.0
    return f"{value:.4f}"


def format_p(value):
    value = float(value)
    if value < 0.0001:
        return "$<0.0001$"
    return format_float(value)


def mode_label(mode):
    labels = {
        "keyword_expansion": "Keyword/entity expansion",
        "direct_rewrite": "Direct rewrite",
        "question_decomposition": "Question decomposition",
        "document_like_passage": "Document-like passage",
    }
    return labels.get(mode, str(mode).replace("_", " "))


def dataset_label(dataset):
    if dataset == "2WikiMultihopQA":
        return "2Wiki"
    return dataset


def latex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def summary_lookup(summary_rows):
    return {
        (row["dataset"], row["reader"], row["query_mode"]): row
        for row in summary_rows
    }


def write_markdown(path, summary_rows, contrast_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = summary_lookup(summary_rows)
    lines = [
        "# Equal-budget Qwen-Turbo Reader Paired Statistics",
        "",
        "This no-new-LLM diagnostic reuses fixed top-10 equal-budget BGE-base evidence and Qwen-Turbo answer files. Bootstrap intervals are paired 95% confidence intervals over per-example answer-score deltas. McNemar's exact test uses EM correctness transitions.",
        "",
        "| Dataset | Baseline | Baseline EM/F1 | Doc-like EM/F1 | Delta EM | Delta F1 [95% CI] | W->C / C->W | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in contrast_rows:
        baseline = lookup[(row["dataset"], row["reader"], row["baseline"])]
        target = lookup[(row["dataset"], row["reader"], row["target"])]
        lines.append(
            f"| {dataset_label(row['dataset'])} | {mode_label(row['baseline'])} | "
            f"{format_float(baseline['em'])}/{format_float(baseline['f1'])} | "
            f"{format_float(target['em'])}/{format_float(target['f1'])} | "
            f"{format_float(row['delta_em'])} | "
            f"{format_float(row['delta_f1'])} [{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}] | "
            f"{row['wrong_to_correct']} / {row['correct_to_wrong']} | "
            f"{format_p(row['mcnemar_exact_p']).replace('$', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, summary_rows, contrast_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = summary_lookup(summary_rows)
    ordered_rows = sorted(
        contrast_rows,
        key=lambda row: (
            0 if row["dataset"] == "HotpotQA" else 1,
            0 if row["baseline"] == "direct_rewrite" else 1,
        ),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Targeted Qwen-Turbo reader paired check on fixed equal-budget evidence.}",
        "\\label{tab:equal_budget_reader_check}",
        "\\scriptsize",
        "\\begin{threeparttable}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{lXrrrrcc}",
        "\\toprule",
        "Dataset & Baseline $\\rightarrow$ doc-like & Baseline EM/F1 & Doc-like EM/F1 & $\\Delta$EM & $\\Delta$F1 [95\\% CI] & W$\\rightarrow$C / C$\\rightarrow$W & McNemar $p$ \\\\",
        "\\midrule",
    ]
    for row in ordered_rows:
        baseline = lookup[(row["dataset"], row["reader"], row["baseline"])]
        target = lookup[(row["dataset"], row["reader"], row["target"])]
        lines.append(
            f"{latex_escape(dataset_label(row['dataset']))} & "
            f"{latex_escape(mode_label(row['baseline']))} $\\rightarrow$ doc-like & "
            f"{format_float(baseline['em'])}/{format_float(baseline['f1'])} & "
            f"{format_float(target['em'])}/{format_float(target['f1'])} & "
            f"{format_float(row['delta_em'])} & "
            f"{format_float(row['delta_f1'])} [{format_float(row['delta_f1_ci_low'])}, {format_float(row['delta_f1_ci_high'])}] & "
            f"{row['wrong_to_correct']} / {row['correct_to_wrong']} & "
            f"{format_p(row['mcnemar_exact_p'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\begin{tablenotes}",
            "\\footnotesize",
            "\\item All rows reuse the fixed top-10 evidence from the strict equal-budget BGE-base retrieval diagnostic and regenerate only reader answers with \\texttt{qwen-turbo}. $\\Delta$ values are document-like passage minus the named baseline over 500 paired examples. F1 intervals are paired bootstrap 95\\% confidence intervals. W$\\rightarrow$C and C$\\rightarrow$W are EM transitions for document-like versus the baseline; McNemar's exact test is computed on these discordant EM counts.",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_reader_outputs(
    answers_dir,
    summary_csv,
    contrasts_csv,
    contrasts_md=None,
    latex_table=None,
    expected_n=500,
    readers=DEFAULT_READERS,
    datasets=DEFAULT_DATASETS,
    modes=DEFAULT_MODES,
    iterations=2000,
    seed=13,
):
    answers_dir = Path(answers_dir)
    summary_rows = []
    loaded = {}
    for reader_label, reader_model in readers:
        for dataset, prefix in datasets:
            for mode in modes:
                path = answers_dir / f"{prefix}_{mode}_top10_answers_{reader_label}_500.jsonl"
                rows = read_answer_rows(path, expected_n)
                metrics = summarize_rows(rows)
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "query_mode": mode,
                        "reader": reader_label,
                        "reader_model": reader_model,
                        "n": metrics["n"],
                        "em": metrics["em"],
                        "f1": metrics["f1"],
                    }
                )
                loaded[(dataset, reader_label, mode)] = load_by_id(rows)

    contrast_rows = []
    if "document_like_passage" in modes:
        baselines = [mode for mode in ("direct_rewrite", "keyword_expansion") if mode in modes]
        for baseline_offset, baseline_mode in enumerate(baselines):
            for reader_label, _ in readers:
                for dataset_offset, (dataset, _) in enumerate(datasets):
                    contrast_rows.append(
                        paired_contrast(
                            dataset,
                            reader_label,
                            baseline_mode,
                            "document_like_passage",
                            loaded,
                            iterations=iterations,
                            seed=seed + baseline_offset * 100 + dataset_offset * 10,
                        )
                    )

    write_csv(summary_csv, summary_rows)
    write_csv(contrasts_csv, contrast_rows)
    if contrasts_md:
        write_markdown(contrasts_md, summary_rows, contrast_rows)
    if latex_table:
        write_latex(latex_table, summary_rows, contrast_rows)
    return summary_rows, contrast_rows


def parse_reader(value):
    if "=" in value:
        label, model = value.split("=", 1)
        return label, model
    return value, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_dir", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--contrasts_csv", required=True)
    parser.add_argument("--contrasts_md")
    parser.add_argument("--latex_table")
    parser.add_argument("--expected_n", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--reader",
        action="append",
        dest="readers",
        default=[],
        help="Reader as label=model, e.g. qwenmax=qwen3.7-max. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        default=[],
        choices=DEFAULT_MODES,
        help="Query mode to summarize. Can be repeated.",
    )
    args = parser.parse_args()

    readers = [parse_reader(item) for item in args.readers] if args.readers else DEFAULT_READERS
    modes = args.modes if args.modes else DEFAULT_MODES
    summary_rows, contrast_rows = summarize_reader_outputs(
        answers_dir=args.answers_dir,
        summary_csv=args.summary_csv,
        contrasts_csv=args.contrasts_csv,
        contrasts_md=args.contrasts_md,
        latex_table=args.latex_table,
        expected_n=args.expected_n,
        readers=readers,
        modes=modes,
        iterations=args.iterations,
        seed=args.seed,
    )
    for row in summary_rows:
        print(
            f"{row['dataset']} {row['reader']} {row['query_mode']}: "
            f"EM={row['em']:.4f}, F1={row['f1']:.4f}"
        )
    for row in contrast_rows:
        print(
            f"{row['dataset']} {row['reader']} {row['target']}-{row['baseline']}: "
            f"dEM={row['delta_em']:.4f}, "
            f"dF1={row['delta_f1']:.4f} "
            f"[{row['delta_f1_ci_low']:.4f}, {row['delta_f1_ci_high']:.4f}], "
            f"W->C/C->W={row['wrong_to_correct']}/{row['correct_to_wrong']}, "
            f"p={row['mcnemar_exact_p']:.4f}"
        )


if __name__ == "__main__":
    main()
