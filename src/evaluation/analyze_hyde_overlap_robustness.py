import argparse
import csv
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl


def mean(values):
    return sum(values) / len(values) if values else 0.0


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hits = gold_titles & retrieved_titles
    return {
        "any_hit": float(bool(hits)),
        "all_hit": float(gold_titles.issubset(retrieved_titles) if gold_titles else False),
        "support_recall": len(hits) / len(gold_titles) if gold_titles else 0.0,
    }


def answer_metrics(row):
    gold = row.get("gold_answer", row.get("answer", ""))
    pred = row.get("prediction", "")
    return {
        "em": float(exact_match(pred, gold)),
        "f1": f1_score(pred, gold),
    }


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def metric_summary(rows_by_id, ids):
    rows = [rows_by_id[qid] for qid in ids if qid in rows_by_id]
    support = [support_metrics(row) for row in rows]
    answers = [answer_metrics(row) for row in rows]
    return {
        "any_hit": mean([row["any_hit"] for row in support]),
        "all_hit": mean([row["all_hit"] for row in support]),
        "support_recall": mean([row["support_recall"] for row in support]),
        "em": mean([row["em"] for row in answers]),
        "f1": mean([row["f1"] for row in answers]),
    }


def paired_group_summary(group, ids, baseline_by_id, target_by_id, baseline_label, target_label):
    paired_ids = [qid for qid in ids if qid in baseline_by_id and qid in target_by_id]
    baseline = metric_summary(baseline_by_id, paired_ids)
    target = metric_summary(target_by_id, paired_ids)
    wrong_to_correct = 0
    correct_to_wrong = 0
    for qid in paired_ids:
        base_em = answer_metrics(baseline_by_id[qid])["em"] == 1.0
        target_em = answer_metrics(target_by_id[qid])["em"] == 1.0
        wrong_to_correct += int((not base_em) and target_em)
        correct_to_wrong += int(base_em and (not target_em))
    return {
        "group": group,
        "baseline": baseline_label,
        "target": target_label,
        "n": len(paired_ids),
        "baseline_all_hit": baseline["all_hit"],
        "target_all_hit": target["all_hit"],
        "delta_all_hit": target["all_hit"] - baseline["all_hit"],
        "baseline_recall": baseline["support_recall"],
        "target_recall": target["support_recall"],
        "delta_recall": target["support_recall"] - baseline["support_recall"],
        "baseline_em": baseline["em"],
        "target_em": target["em"],
        "delta_em": target["em"] - baseline["em"],
        "baseline_f1": baseline["f1"],
        "target_f1": target["f1"],
        "delta_f1": target["f1"] - baseline["f1"],
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
    }


def leakage_groups(leakage_by_id):
    all_ids = list(leakage_by_id)
    answer_in = [
        qid for qid, row in leakage_by_id.items()
        if int(row.get("answer_in_hyde", 0)) == 1
    ]
    answer_not_in = [
        qid for qid, row in leakage_by_id.items()
        if int(row.get("answer_in_hyde", 0)) == 0
    ]
    nontrivial_in = [
        qid for qid, row in leakage_by_id.items()
        if int(row.get("answer_in_hyde", 0)) == 1
        and int(row.get("short_or_ambiguous_answer", 0)) == 0
    ]
    return [
        ("all_examples", all_ids),
        ("answer_in_hyde", answer_in),
        ("answer_not_in_hyde", answer_not_in),
        ("nontrivial_answer_in_hyde", nontrivial_in),
    ]


def format_float(value):
    return f"{value:.4f}"


def write_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# HyDE Answer-Overlap Robustness Analysis",
        "",
        "This analysis stratifies HotpotQA examples by whether the generated HyDE hypothetical passage contains the normalized gold answer string. It then compares Dense RAG and HyDE-style RAG within each group using existing answer and retrieval files; no new LLM calls are used.",
        "",
        "| Group | n | Dense all-hit | HyDE all-hit | Delta all-hit | Dense F1 | HyDE F1 | Delta F1 | W->C | C->W |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['n']} | {format_float(row['baseline_all_hit'])} | "
            f"{format_float(row['target_all_hit'])} | {format_float(row['delta_all_hit'])} | "
            f"{format_float(row['baseline_f1'])} | {format_float(row['target_f1'])} | "
            f"{format_float(row['delta_f1'])} | {row['wrong_to_correct']} | {row['correct_to_wrong']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The answer-not-in-HyDE group is the most important robustness slice. A positive HyDE gain in this group indicates that the method is not solely explained by explicit answer-string overlap in the hypothetical passage. The answer-in-HyDE group remains a leakage-sensitive subgroup and should be interpreted as query-side answer-bearing expansion.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Answer-overlap robustness analysis on HotpotQA.}",
        "\\label{tab:hyde_overlap_robustness}",
        "\\scriptsize",
        "\\begin{threeparttable}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X r r r r r r c}",
        "\\toprule",
        "Group & $N$ & Dense all-hit & HyDE all-hit & $\\Delta$ all-hit & Dense F1 & HyDE F1 & W$\\rightarrow$C / C$\\rightarrow$W \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['group'].replace('_', ' ')} & {row['n']} & "
            f"{format_float(row['baseline_all_hit'])} & {format_float(row['target_all_hit'])} & "
            f"{format_float(row['delta_all_hit'])} & {format_float(row['baseline_f1'])} & "
            f"{format_float(row['target_f1'])} & {row['wrong_to_correct']} / {row['correct_to_wrong']} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item Groups are defined by normalized string overlap between the gold answer and the generated hypothetical passage. The answer-not-in-HyDE row is the robustness slice most relevant to answer-memory concerns; it uses existing answer files and does not require new LLM calls.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leakage", default="results/ircot_hotpotqa_test500_hyde_leakage_audit.jsonl")
    parser.add_argument("--baseline_answers", default="results/ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl")
    parser.add_argument("--target_answers", default="results/ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl")
    parser.add_argument("--baseline_label", default="Dense RAG")
    parser.add_argument("--target_label", default="HyDE-style RAG")
    parser.add_argument("--out_csv", default="local-artifacts/hyde_overlap_robustness_hotpotqa_test500.csv")
    parser.add_argument("--out_md", default="local-artifacts/hyde_overlap_robustness_hotpotqa_test500.md")
    parser.add_argument("--out_tex", default="paper/latex/table_hyde_overlap_robustness.tex")
    args = parser.parse_args()

    leakage_by_id = load_by_id(args.leakage)
    baseline_by_id = load_by_id(args.baseline_answers)
    target_by_id = load_by_id(args.target_answers)

    rows = [
        paired_group_summary(
            group,
            ids,
            baseline_by_id,
            target_by_id,
            args.baseline_label,
            args.target_label,
        )
        for group, ids in leakage_groups(leakage_by_id)
    ]

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)
    write_latex(args.out_tex, rows)
    for row in rows:
        print(
            f"{row['group']}: n={row['n']} "
            f"Dense F1={format_float(row['baseline_f1'])} "
            f"HyDE F1={format_float(row['target_f1'])} "
            f"Delta F1={format_float(row['delta_f1'])}"
        )


if __name__ == "__main__":
    main()
