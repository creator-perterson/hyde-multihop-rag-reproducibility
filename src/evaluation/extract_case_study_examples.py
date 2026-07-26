import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from utils import read_jsonl


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hit_titles = gold_titles & retrieved_titles
    recall = len(hit_titles) / len(gold_titles) if gold_titles else 0.0
    if recall >= 1.0:
        bucket = "full"
    elif recall > 0:
        bucket = "partial"
    else:
        bucket = "none"
    return {
        "support_gold_count": len(gold_titles),
        "support_hit_count": len(hit_titles),
        "support_recall": recall,
        "bucket": bucket,
        "hit_titles": sorted(hit_titles),
        "missing_titles": sorted(gold_titles - retrieved_titles),
    }


def answer_metrics(row):
    return {
        "em": float(exact_match(row.get("prediction", ""), row.get("gold_answer", ""))),
        "f1": f1_score(row.get("prediction", ""), row.get("gold_answer", "")),
    }


def compact_question(text, max_chars=145):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def select_hyde_success_case(dense_by_id, hyde_by_id, leakage_by_id):
    candidates = []
    for qid in sorted(set(dense_by_id) & set(hyde_by_id)):
        dense = dense_by_id[qid]
        hyde = hyde_by_id[qid]
        dense_support = support_metrics(dense)
        hyde_support = support_metrics(hyde)
        dense_answer = answer_metrics(dense)
        hyde_answer = answer_metrics(hyde)
        leakage = leakage_by_id.get(qid, {})
        if int(leakage.get("answer_in_hyde", 1)) != 0:
            continue
        if dense_support["support_recall"] >= 1.0 or hyde_support["support_recall"] < 1.0:
            continue
        if dense_answer["em"] >= 1.0 or hyde_answer["em"] < 1.0:
            continue
        candidates.append((hyde_answer["f1"] - dense_answer["f1"], qid, dense, hyde, dense_support, hyde_support))
    if not candidates:
        raise ValueError("No HyDE success case matched the strict selection rule.")
    _, qid, dense, hyde, dense_support, hyde_support = sorted(candidates, reverse=True)[0]
    return {
        "case_type": "HyDE evidence acquisition",
        "id": qid,
        "question": dense["question"],
        "gold_answer": dense["gold_answer"],
        "dense_prediction": dense["prediction"],
        "hyde_prediction": hyde["prediction"],
        "dense_support": dense_support,
        "hyde_support": hyde_support,
        "lesson": "HyDE retrieves the missing supporting title without an exact normalized answer-string match in the hypothetical passage, turning a partial-evidence Dense failure into a correct answer.",
    }


def select_guard_case(detail_rows):
    by_id = {}
    for row in detail_rows:
        by_id.setdefault(row["id"], {})[row["variant"]] = row
    candidates = []
    for qid, variants in by_id.items():
        raw = variants.get("raw")
        guarded = variants.get("both_guards")
        if not raw or not guarded:
            continue
        if raw.get("transition") == "correct_to_wrong" and guarded.get("transition") == "correct_to_correct":
            candidates.append((qid, raw, guarded))
    if not candidates:
        raise ValueError("No guard case matched raw correct-to-wrong and guarded correct-to-correct.")
    qid, raw, guarded = sorted(candidates)[0]
    return {
        "case_type": "Verifier guard prevents harm",
        "id": qid,
        "question": raw["question"],
        "gold_answer": raw["gold_answer"],
        "initial_prediction": raw["initial_prediction"],
        "raw_final_answer": raw["final_answer"],
        "guarded_final_answer": guarded["final_answer"],
        "lesson": "The raw verifier adds an unsupported numeric unit and would flip a correct answer to wrong; the guard keeps the original answer.",
    }


def select_alias_limitation_case(hyde_by_id):
    preferred_id = "5a728f015542991f9a20c4e4"
    if preferred_id not in hyde_by_id:
        raise ValueError("Preferred alias case is missing from HyDE answers.")
    row = hyde_by_id[preferred_id]
    support = support_metrics(row)
    return {
        "case_type": "Evaluation / alias limitation",
        "id": preferred_id,
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "hyde_prediction": row["prediction"],
        "hyde_support": support,
        "lesson": "The reader returns the professional name Marty Ingels, whereas the benchmark gold answer uses the birth name Martin Ingerman. The two names refer to the same person, illustrating that exact-match evaluation can count semantically valid aliases as errors.",
    }


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def support_phrase(metrics):
    return f"{metrics['bucket']} ({metrics['support_hit_count']}/{metrics['support_gold_count']})"


def compact_text(text, max_chars=220):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def ordered_top_titles(row, limit=3):
    titles = []
    seen = set()
    for doc in row.get("retrieved", []) if row else []:
        title = doc.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def retrieval_artifact_summary(row, label):
    if not row:
        return f"{label} top: not available; supporting-title hits: not available"
    top_titles = ordered_top_titles(row)
    metrics = support_metrics(row)
    top_text = "; ".join(top_titles) if top_titles else "none"
    hit_text = "; ".join(metrics["hit_titles"]) if metrics["hit_titles"] else "none"
    missing_text = "; ".join(metrics["missing_titles"]) if metrics["missing_titles"] else "none"
    return f"{label} top: {top_text}; supporting-title hits: {hit_text}; missing: {missing_text}"


def relevant_evidence_excerpts(*rows, max_docs=2, max_chars=170):
    selected = []
    selected_titles = set()
    for row in rows:
        if not row:
            continue
        gold_titles = set(row.get("supporting_facts", {}).get("title", []))
        for doc in row.get("retrieved", []):
            title = doc.get("title", "")
            if title not in gold_titles or title in selected_titles:
                continue
            text = doc.get("text", "")
            if text:
                selected.append(f"[{title}] {compact_text(text, max_chars=max_chars)}")
                selected_titles.add(title)
            if len(selected) >= max_docs:
                return "; ".join(selected)
    return "not available"


def build_case_artifact_details(
    cases,
    dense_retrieval_by_id,
    hyde_retrieval_by_id,
    hyde_answers_by_id,
    verifier_by_id,
    guard_rows,
):
    guard_by_id = {}
    for row in guard_rows:
        guard_by_id.setdefault(row["id"], {})[row["variant"]] = row

    details = []
    for case in cases:
        qid = case["id"]
        dense_row = dense_retrieval_by_id.get(qid, {})
        hyde_row = hyde_retrieval_by_id.get(qid, {})
        hyde_answer = hyde_answers_by_id.get(qid, {})
        guard_variants = guard_by_id.get(qid, {})
        guarded = guard_variants.get("both_guards", {})
        verifier = verifier_by_id.get(qid, {})

        initial_answer = (
            case.get("initial_prediction")
            or hyde_answer.get("prediction")
            or case.get("hyde_prediction")
            or case.get("dense_prediction")
            or ""
        )
        final_answer = guarded.get("final_answer") or case.get("guarded_final_answer") or initial_answer
        raw_verifier = verifier.get("prediction") or "not invoked for this example"

        details.append(
            {
                "case_type": case["case_type"],
                "example_id": qid,
                "generated_hypothetical_passage": compact_text(hyde_row.get("hyde_document", ""), max_chars=260)
                or "not available",
                "dense_retrieval": retrieval_artifact_summary(dense_row, "Dense"),
                "hyde_retrieval": retrieval_artifact_summary(hyde_row, "HyDE"),
                "relevant_evidence_excerpts": relevant_evidence_excerpts(hyde_row, dense_row),
                "initial_reader_answer": initial_answer,
                "raw_verifier_json": compact_text(raw_verifier, max_chars=260),
                "final_guarded_answer": final_answer,
            }
        )
    return details


def case_prediction_summary(case):
    if case["case_type"] == "HyDE evidence acquisition":
        return (
            f"Dense: {case['dense_prediction']}; "
            f"HyDE: {case['hyde_prediction']}; Gold: {case['gold_answer']}"
        )
    if case["case_type"] == "Verifier guard prevents harm":
        return (
            f"Reader: {case['initial_prediction']}; "
            f"Raw verifier: {case['raw_final_answer']}; "
            f"Guarded: {case['guarded_final_answer']}; Gold: {case['gold_answer']}"
        )
    return f"HyDE: {case['hyde_prediction']}; Gold: {case['gold_answer']}"


def case_support_summary(case):
    if case["case_type"] == "HyDE evidence acquisition":
        return f"Dense support {support_phrase(case['dense_support'])}; HyDE support {support_phrase(case['hyde_support'])}"
    if case["case_type"] == "Verifier guard prevents harm":
        return "Same retrieved evidence and verifier output; only deterministic guard policy changes."
    return f"HyDE support {support_phrase(case['hyde_support'])}"


def write_markdown(path, cases):
    lines = [
        "# Artifact-grounded Qualitative Case Studies",
        "",
        "These cases were selected from rule-filtered candidates using frozen per-example artifacts. They are illustrative and do not add new evaluation examples.",
        "",
        "| Case | ID | Question | Predictions | Evidence state | Lesson |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            f"| {markdown_escape(case['case_type'])} | `{case['id']}` | {markdown_escape(compact_question(case['question']))} | "
            f"{markdown_escape(case_prediction_summary(case))} | {markdown_escape(case_support_summary(case))} | {markdown_escape(case['lesson'])} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_escape(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_latex(path, cases):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Artifact-grounded qualitative case studies.}",
        r"\label{tab:qualitative_cases}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.17\textwidth}",
        r"                            >{\raggedright\arraybackslash}p{0.27\textwidth}",
        r"                            >{\raggedright\arraybackslash}p{0.23\textwidth}",
        r"                            >{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Case type & Question summary & Prediction / evidence state & Interpretation \\",
        r"\midrule",
    ]
    for case in cases:
        lines.append(
            f"{latex_escape(case['case_type'])} & "
            f"{latex_escape(compact_question(case['question'], max_chars=135))} & "
            f"{latex_escape(case_prediction_summary(case))}; {latex_escape(case_support_summary(case))} & "
            f"{latex_escape(case['lesson'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item Cases were selected from rule-filtered candidates using frozen HotpotQA per-example artifacts: Dense and HyDE reader outputs, HyDE answer-overlap audit rows, and verifier guard-ablation details. They are illustrative rather than additional evaluation examples.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact_markdown(path, details):
    lines = [
        "# Supplementary Artifact Details for Qualitative Cases",
        "",
        "These compact blocks expose the frozen artifacts underlying Table 7 / Table S18.",
        "",
    ]
    for item in details:
        lines.extend(
            [
                f"## {markdown_escape(item['case_type'])} (`{item['example_id']}`)",
                "",
                f"- Generated hypothetical passage: {markdown_escape(item['generated_hypothetical_passage'])}",
                f"- Dense retrieved supporting titles: {markdown_escape(item['dense_retrieval'])}",
                f"- HyDE retrieved supporting titles: {markdown_escape(item['hyde_retrieval'])}",
                f"- Relevant evidence excerpts: {markdown_escape(item['relevant_evidence_excerpts'])}",
                f"- Initial reader answer: {markdown_escape(item['initial_reader_answer'])}",
                f"- Raw verifier JSON: {markdown_escape(item['raw_verifier_json'])}",
                f"- Final guarded answer: {markdown_escape(item['final_guarded_answer'])}",
                "",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_artifact_latex(path, details):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Compact artifact blocks for the qualitative case studies.}",
        r"\label{tab:qualitative_case_artifacts}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.15\textwidth}",
        r"                            >{\raggedright\arraybackslash}p{0.28\textwidth}",
        r"                            >{\raggedright\arraybackslash}p{0.28\textwidth}",
        r"                            >{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Example & Generated / retrieved artifacts & Evidence excerpts & Reader / verifier artifacts \\",
        r"\midrule",
    ]
    for item in details:
        lines.append(
            f"{latex_escape(item['case_type'])}\\newline{{\\tiny\\texttt{{{latex_escape(item['example_id'])}}}}} & "
            f"HyDE passage: {latex_escape(item['generated_hypothetical_passage'])}\\newline "
            f"{latex_escape(item['dense_retrieval'])}\\newline "
            f"{latex_escape(item['hyde_retrieval'])} & "
            f"{latex_escape(item['relevant_evidence_excerpts'])} & "
            f"Initial reader answer: {latex_escape(item['initial_reader_answer'])}\\newline "
            f"Raw verifier JSON: {latex_escape(item['raw_verifier_json'])}\\newline "
            f"Final guarded answer: {latex_escape(item['final_guarded_answer'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Each block is a compact projection of frozen per-example artifacts. ``Not invoked'' means the verifier was not run for that example under the risk-selected verification protocol.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_cases(dense_answers, hyde_answers, leakage_audit, guard_details):
    dense_by_id = load_by_id(dense_answers)
    hyde_by_id = load_by_id(hyde_answers)
    leakage_by_id = load_by_id(leakage_audit)
    guard_rows = list(read_jsonl(guard_details))
    return [
        select_hyde_success_case(dense_by_id, hyde_by_id, leakage_by_id),
        select_guard_case(guard_rows),
        select_alias_limitation_case(hyde_by_id),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_answers", default="results/ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl")
    parser.add_argument("--hyde_answers", default="results/ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl")
    parser.add_argument("--dense_retrieval", default="results/ircot_hotpotqa_test500_top10_retrieval.jsonl")
    parser.add_argument("--hyde_retrieval", default="results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl")
    parser.add_argument("--leakage_audit", default="results/ircot_hotpotqa_test500_hyde_leakage_audit.jsonl")
    parser.add_argument("--guard_details", default="results/qwenmax_hyde_verifier_guard_ablation_details.jsonl")
    parser.add_argument("--verifier_answers", default="results/qwenmax_hyde_verification_answers_conservative_risk120.jsonl")
    parser.add_argument("--out_md", default="local-artifacts/artifact_grounded_case_studies.md")
    parser.add_argument("--out_tex", default="paper/latex/table_qualitative_case_study.tex")
    parser.add_argument("--out_artifact_md", default="local-artifacts/artifact_grounded_case_details.md")
    parser.add_argument("--out_artifact_tex", default="paper/latex/table_qualitative_case_artifacts.tex")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    cases = extract_cases(args.dense_answers, args.hyde_answers, args.leakage_audit, args.guard_details)
    details = build_case_artifact_details(
        cases,
        load_by_id(args.dense_retrieval),
        load_by_id(args.hyde_retrieval),
        load_by_id(args.hyde_answers),
        load_by_id(args.verifier_answers),
        list(read_jsonl(args.guard_details)),
    )
    write_markdown(code_root / args.out_md, cases)
    write_latex(code_root / args.out_tex, cases)
    write_artifact_markdown(code_root / args.out_artifact_md, details)
    write_artifact_latex(code_root / args.out_artifact_tex, details)
    for case in cases:
        print(f"{case['case_type']}: {case['id']}")


if __name__ == "__main__":
    main()
