"""Build open-generator LaTeX tables from complete verified summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("hotpotqa", "2wiki")
GENERATED = (
    "open_hyde_standard",
    "open_hyde_evidence_only",
    "open_query2doc_4shot",
)
RETRIEVAL_METHODS = ("dense_bge",) + GENERATED
SCRUB_SUFFIXES = ("answer_scrubbed", "gold_content_scrubbed")
DISPLAY_DATASETS = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki"}
DISPLAY_METHODS = {
    "dense_bge": "Dense",
    "open_hyde_standard": "Open HyDE",
    "open_hyde_evidence_only": "Evidence-only HyDE",
    "open_query2doc_4shot": "Query2doc-style (4-shot adaptation)",
}
REGISTERED_REFERENCES = {
    "open_hyde_standard": "dense_bge",
    "open_query2doc_4shot": "dense_bge",
    "open_hyde_evidence_only": "open_hyde_standard",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required summary: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Summary is empty: {path}")
    return rows


def required_rows(rows: list[dict[str, str]], dataset: str, methods: tuple[str, ...]):
    index = {(row.get("dataset"), row.get("method")): row for row in rows}
    selected = {}
    for method in methods:
        row = index.get((dataset, method))
        if row is None:
            raise ValueError(f"Missing complete row for {dataset}:{method}")
        if row.get("n") != "500":
            raise ValueError(f"Expected n=500 for {dataset}:{method}, got {row.get('n')}")
        selected[method] = row
    return selected


def contrast_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row.get("dataset"), row.get("target"), row.get("reference")): row
        for row in rows
    }


def fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def latex_escape(value: str) -> str:
    return str(value).replace("_", r"\_")


def build_main_table(
    retrieval_rows: list[dict[str, str]],
    contrast_rows: list[dict[str, str]],
    leakage_rows: list[dict[str, str]],
) -> str:
    contrast = contrast_index(contrast_rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Open-weight retrieval-stage confirmatory results. Deltas and 95\% paired-bootstrap confidence intervals use each condition's pre-registered reference: Dense for standard HyDE and the Query2doc-style adaptation, and standard HyDE for evidence-only HyDE.}",
        r"\label{tab:open_generator_query2doc}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrlrrrr}",
        r"\toprule",
        r"Dataset & Condition & All-support & Recall & Reference & Delta & 95\% CI & Exact-answer & Mean words \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        selected = required_rows(retrieval_rows, dataset, RETRIEVAL_METHODS)
        for method in RETRIEVAL_METHODS:
            row = selected[method]
            if method == "dense_bge":
                reference = r"--"
                delta = r"--"
                interval = r"--"
            else:
                reference_method = REGISTERED_REFERENCES[method]
                paired = contrast[(dataset, method, reference_method)]
                reference = DISPLAY_METHODS[reference_method]
                delta = fmt(paired["paired_delta"])
                interval = f"[{fmt(paired['ci_low'])}, {fmt(paired['ci_high'])}]"
            exact_answer = row.get("exact_answer_overlap_rate", "") or "--"
            mean_words = row.get("mean_generated_words", "") or "--"
            if exact_answer != "--":
                exact_answer = fmt(exact_answer)
            if mean_words != "--":
                mean_words = fmt(mean_words, digits=1)
            lines.append(
                f"{DISPLAY_DATASETS[dataset]} & {DISPLAY_METHODS[method]} & "
                f"{fmt(row['all_support_hit@10'])} & "
                f"{fmt(row['supporting_title_recall@10'])} & "
                f"{reference} & {delta} & {interval} & {exact_answer} & "
                f"{mean_words} \\\\" 
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def build_leakage_table(
    retrieval_rows: list[dict[str, str]],
    contrast_rows: list[dict[str, str]],
    leakage_rows: list[dict[str, str]],
) -> str:
    retrieval = {(row.get("dataset"), row.get("method")): row for row in retrieval_rows}
    contrasts = contrast_index(contrast_rows)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Open-generator content strata and gold-aware retrieval diagnostics. Answer-scrubbed and Gold-content-scrubbed rows are diagnostic only. McNemar and Holm-adjusted $p$-values use each condition's pre-registered reference.}",
        r"\label{tab:open_generator_leakage}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Condition & Ref. & Exact & Alias & Title-only & None & Original & Ans.-scrub. & Gold-scrub. & McNemar $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for condition in GENERATED:
            leakage = next(
                (
                    row
                    for row in leakage_rows
                    if row.get("dataset") == dataset
                    and row.get("method", row.get("condition")) == condition
                ),
                None,
            )
            if leakage is None or leakage.get("n") != "500":
                raise ValueError(f"Missing complete leakage row for {dataset}:{condition}")
            original = retrieval.get((dataset, condition))
            answer_scrubbed = retrieval.get((dataset, f"{condition}__answer_scrubbed"))
            gold_scrubbed = retrieval.get((dataset, f"{condition}__gold_content_scrubbed"))
            if not original or not answer_scrubbed or not gold_scrubbed:
                raise ValueError(f"Missing scrubbed retrieval rows for {dataset}:{condition}")
            reference = REGISTERED_REFERENCES[condition]
            paired = contrasts[(dataset, condition, reference)]
            lines.append(
                f"{DISPLAY_DATASETS[dataset]} & {DISPLAY_METHODS[condition]} & "
                f"{DISPLAY_METHODS[reference]} & "
                f"{leakage['count_exact_answer']} & {leakage['count_alias_proxy']} & "
                f"{leakage['count_supporting_title_only']} & "
                f"{leakage['count_no_identifiable_gold_content']} & "
                f"{fmt(original['all_support_hit@10'])} & "
                f"{fmt(answer_scrubbed['all_support_hit@10'])} & "
                f"{fmt(gold_scrubbed['all_support_hit@10'])} & "
                f"{fmt(paired['mcnemar_p'])} & "
                f"{fmt(paired['holm_adjusted_p'])} \\\\" 
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="artifacts/summaries/expert_systems_minimum")
    parser.add_argument("--leakage_csv", default=None)
    parser.add_argument("--out_main", default="paper/latex/table_open_generator_query2doc.tex")
    parser.add_argument("--out_leakage", default="paper/latex/table_open_generator_leakage.tex")
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    retrieval_rows = read_csv(summary_dir / "retrieval_summary.csv")
    contrast_rows = read_csv(summary_dir / "paired_contrasts.csv")
    leakage_rows = read_csv(
        Path(args.leakage_csv)
        if args.leakage_csv
        else summary_dir / "leakage_summary.csv"
    )
    main_text = build_main_table(retrieval_rows, contrast_rows, leakage_rows)
    leakage_text = build_leakage_table(
        retrieval_rows, contrast_rows, leakage_rows
    )
    for output, text in ((Path(args.out_main), main_text), (Path(args.out_leakage), leakage_text)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out_main} and {args.out_leakage}")


if __name__ == "__main__":
    main()
