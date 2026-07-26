import argparse
import csv
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl, write_jsonl
from verifier.evaluate_verification import extract_final_answer, transition_label
from verifier.evaluate_selective_verification import (
    is_abstention,
    should_block_numeric_unit_addition,
)


VARIANTS = [
    ("reader_only", "HyDE reader only"),
    ("raw", "HyDE + verifier raw"),
    ("no_abstention_guard", "HyDE + no-abstention guard"),
    ("numeric_unit_guard", "HyDE + numeric-unit guard"),
    ("both_guards", "HyDE + both guards"),
]


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def apply_guard_variant(initial, verifier_text, question, variant):
    final = extract_final_answer(verifier_text, fallback=initial)
    if variant == "raw":
        return final
    if variant == "no_abstention_guard":
        if not is_abstention(initial) and is_abstention(final):
            return initial
        return final
    if variant == "numeric_unit_guard":
        if should_block_numeric_unit_addition(initial, final, question):
            return initial
        return final
    if variant == "both_guards":
        if not is_abstention(initial) and is_abstention(final):
            return initial
        if should_block_numeric_unit_addition(initial, final, question):
            return initial
        return final
    if variant == "reader_only":
        return initial
    raise ValueError(f"Unknown guard variant: {variant}")


def evaluate_final(initial, final, gold):
    initial_correct = exact_match(initial, gold)
    final_correct = exact_match(final, gold)
    return {
        "initial_em": float(initial_correct),
        "final_em": float(final_correct),
        "initial_f1": f1_score(initial, gold),
        "final_f1": f1_score(final, gold),
        "transition": transition_label(initial_correct, final_correct),
    }


def variant_rows(variant, base_rows_by_id, verifier_rows_by_id):
    rows = []
    for qid, base in base_rows_by_id.items():
        initial = base["prediction"]
        verifier = verifier_rows_by_id.get(qid)
        verified = verifier is not None and variant != "reader_only"
        if variant == "reader_only" or not verifier:
            final = initial
            verifier_final = ""
        else:
            verifier_final = extract_final_answer(verifier["prediction"], fallback=initial)
            final = apply_guard_variant(initial, verifier["prediction"], base.get("question", ""), variant)
        metrics = evaluate_final(initial, final, base["gold_answer"])
        changed = int(final.strip() != initial.strip())
        raw_changed = (
            int(verifier_final.strip() != initial.strip())
            if verifier and variant != "reader_only"
            else 0
        )
        rows.append({
            "id": qid,
            "variant": variant,
            "question": base.get("question", ""),
            "gold_answer": base["gold_answer"],
            "initial_prediction": initial,
            "verifier_final_answer": verifier_final,
            "final_answer": final,
            "verified": int(verified),
            "changed": changed,
            "raw_changed": raw_changed,
            **metrics,
        })
    return rows


def summarize_variant(variant, base_rows_by_id, verifier_rows_by_id):
    rows = variant_rows(variant, base_rows_by_id, verifier_rows_by_id)
    total = len(rows)
    verified_rows = [row for row in rows if row["verified"]]
    transitions = Counter(row["transition"] for row in verified_rows)
    final_em = sum(row["final_em"] for row in rows) / total if total else 0.0
    final_f1 = sum(row["final_f1"] for row in rows) / total if total else 0.0
    initial_em = sum(row["initial_em"] for row in rows) / total if total else 0.0
    initial_f1 = sum(row["initial_f1"] for row in rows) / total if total else 0.0
    verified_n = len(verified_rows)
    changed_n = sum(row["changed"] for row in verified_rows)
    raw_changed_n = sum(row["raw_changed"] for row in verified_rows)
    return {
        "variant": variant,
        "label": dict(VARIANTS)[variant],
        "n": total,
        "verified_n": verified_n,
        "changed_n": changed_n,
        "raw_changed_n": raw_changed_n,
        "initial_em": initial_em,
        "final_em": final_em,
        "delta_em": final_em - initial_em,
        "initial_f1": initial_f1,
        "final_f1": final_f1,
        "delta_f1": final_f1 - initial_f1,
        "correct_to_correct": transitions.get("correct_to_correct", 0),
        "wrong_to_wrong": transitions.get("wrong_to_wrong", 0),
        "wrong_to_correct": transitions.get("wrong_to_correct", 0),
        "correct_to_wrong": transitions.get("correct_to_wrong", 0),
    }


def run_ablation(base_answers, verifier_answers):
    base_rows_by_id = load_by_id(base_answers)
    verifier_rows_by_id = load_by_id(verifier_answers)
    summaries = [
        summarize_variant(variant, base_rows_by_id, verifier_rows_by_id)
        for variant, _ in VARIANTS
    ]
    detail_rows = []
    for variant, _ in VARIANTS:
        detail_rows.extend(variant_rows(variant, base_rows_by_id, verifier_rows_by_id))
    return summaries, detail_rows


def format_float(value):
    return f"{value:.4f}"


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
        "# HyDE Verifier Guard Ablation",
        "",
        "This no-new-LLM analysis reconstructs verifier variants from the same HyDE reader answers and the same risk-selected verifier outputs. It tests whether the final guard policy matters, without changing retrieval, reader prompts, or verifier generations.",
        "",
        "| Variant | N | Verified | Changed | EM | F1 | Delta F1 | W->C | C->W |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['verified_n']} | {row['changed_n']} | "
            f"{format_float(row['final_em'])} | {format_float(row['final_f1'])} | "
            f"{format_float(row['delta_f1'])} | {row['wrong_to_correct']} | {row['correct_to_wrong']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The raw verifier row applies the verifier final answer without post-processing. The no-abstention and numeric-unit rows isolate the two deterministic guards, and the both-guards row corresponds to the final conservative policy. The table should be read as a safety ablation: verification is intentionally a small canonicalization layer, and the guards are meant to prevent harmful transitions rather than to create the main performance gain.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{HyDE verifier guard ablation reconstructed from existing verifier outputs.}",
        "\\label{tab:hyde_verifier_guard_ablation}",
        "\\small",
        "\\begin{threeparttable}",
        "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X r r r r r r c}",
        "\\toprule",
        "Variant & $N$ & Verified & Changed & EM & F1 & $\\Delta$F1 & W$\\rightarrow$C / C$\\rightarrow$W \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['n']} & {row['verified_n']} & {row['changed_n']} & "
            f"{format_float(row['final_em'])} & {format_float(row['final_f1'])} & "
            f"{format_float(row['delta_f1'])} & {row['wrong_to_correct']} / {row['correct_to_wrong']} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item All rows reuse the same HyDE reader answers and the same 120 risk-selected verifier outputs. The raw row applies verifier final answers directly. The final row applies both deterministic guards: block new abstentions and block unsupported numeric-unit additions. The ablation is intended to justify the verifier's safety boundary, not to present verification as the main performance driver.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_answers",
        default="results/ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
    )
    parser.add_argument(
        "--verifier_answers",
        default="results/qwenmax_hyde_verification_answers_conservative_risk120.jsonl",
    )
    parser.add_argument("--out_csv", default="local-artifacts/hyde_verifier_guard_ablation.csv")
    parser.add_argument("--out_md", default="local-artifacts/hyde_verifier_guard_ablation.md")
    parser.add_argument("--out_tex", default="paper/latex/table_hyde_verifier_guard_ablation.tex")
    parser.add_argument("--out_details", default="results/qwenmax_hyde_verifier_guard_ablation_details.jsonl")
    args = parser.parse_args()

    summaries, detail_rows = run_ablation(args.base_answers, args.verifier_answers)
    code_root = Path(__file__).resolve().parents[2]
    write_csv(code_root / args.out_csv, summaries)
    write_markdown(code_root / args.out_md, summaries)
    write_latex(code_root / args.out_tex, summaries)
    write_jsonl(code_root / args.out_details, detail_rows)
    for row in summaries:
        print(
            f"{row['label']}: EM={format_float(row['final_em'])} "
            f"F1={format_float(row['final_f1'])} "
            f"W->C/C->W={row['wrong_to_correct']}/{row['correct_to_wrong']}"
        )


if __name__ == "__main__":
    main()
