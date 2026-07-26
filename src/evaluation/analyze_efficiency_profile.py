import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl


ANSWER_MARKERS = [
    "\n\nExact short answer:",
    "\n\nShort answer:",
    "\n\nJSON:",
]


@dataclass
class MethodSpec:
    method: str
    reader_prompt_rows: list = field(default_factory=list)
    hyde_prompt_rows: list = field(default_factory=list)
    query_side_prompt_rows: list = field(default_factory=list)
    verifier_prompt_rows: list = field(default_factory=list)
    answer_rows: list = field(default_factory=list)
    retrieval_rows: list = field(default_factory=list)
    retrieval_query_executions_per_question: float = 0.0
    training_required: str = "No"
    notes: str = ""


def evidence_chars_from_prompt(prompt):
    text = str(prompt)
    marker = "Evidence:\n"
    if marker not in text:
        return 0
    evidence = text.split(marker, 1)[1]
    for answer_marker in ANSWER_MARKERS:
        if answer_marker in evidence:
            evidence = evidence.split(answer_marker, 1)[0]
            break
    return len(evidence.strip())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def prompt_stats(rows):
    rows = list(rows)
    prompt_lengths = [len(str(row.get("prompt", ""))) for row in rows]
    evidence_lengths = [evidence_chars_from_prompt(row.get("prompt", "")) for row in rows]
    retrieved_counts = [len(row.get("retrieved", [])) for row in rows]
    return {
        "count": len(rows),
        "total_prompt_chars": sum(prompt_lengths),
        "avg_prompt_chars": mean(prompt_lengths),
        "avg_evidence_chars": mean(evidence_lengths),
        "avg_retrieved_docs": mean(retrieved_counts),
    }


def response_stats(rows):
    rows = list(rows)
    lengths = []
    for row in rows:
        value = row.get("prediction", row.get("final_answer", ""))
        lengths.append(len(str(value)))
    return {
        "count": len(rows),
        "avg_response_chars": mean(lengths),
    }


def retrieval_stats(rows):
    rows = list(rows)
    retrieved_counts = [len(row.get("retrieved", [])) for row in rows]
    raw_chars = []
    for row in rows:
        raw_chars.append(sum(len(str(doc.get("text", ""))) for doc in row.get("retrieved", [])))
    return {
        "count": len(rows),
        "avg_retrieved_docs": mean(retrieved_counts),
        "avg_raw_retrieved_chars": mean(raw_chars),
    }


def summarize_method(spec):
    reader = prompt_stats(spec.reader_prompt_rows)
    query_side_rows = spec.query_side_prompt_rows or spec.hyde_prompt_rows
    query_side = prompt_stats(query_side_rows)
    verifier = prompt_stats(spec.verifier_prompt_rows)
    answers = response_stats(spec.answer_rows)
    retrieval = retrieval_stats(spec.retrieval_rows)

    n_examples = (
        reader["count"]
        or answers["count"]
        or retrieval["count"]
        or query_side["count"]
        or verifier["count"]
    )
    reader_calls = reader["count"] if reader["count"] else answers["count"]
    query_side_calls = query_side["count"]
    verifier_calls = verifier["count"]
    total_calls = reader_calls + query_side_calls + verifier_calls
    total_prompt_chars = (
        reader["total_prompt_chars"]
        + query_side["total_prompt_chars"]
        + verifier["total_prompt_chars"]
    )
    calls_per_question = total_calls / n_examples if n_examples else 0.0
    verifier_coverage = verifier_calls / n_examples if n_examples else 0.0
    avg_total_prompt_chars_per_question = total_prompt_chars / n_examples if n_examples else 0.0

    return {
        "method": spec.method,
        "n_examples": n_examples,
        "reader_calls": reader_calls,
        "hyde_generation_calls": query_side_calls,
        "query_side_llm_calls": query_side_calls,
        "retrieval_query_executions_per_question": spec.retrieval_query_executions_per_question,
        "verifier_calls": verifier_calls,
        "total_llm_calls": total_calls,
        "calls_per_question": calls_per_question,
        "verifier_coverage": verifier_coverage,
        "avg_reader_prompt_chars": reader["avg_prompt_chars"],
        "avg_hyde_prompt_chars": query_side["avg_prompt_chars"],
        "avg_query_side_prompt_chars": query_side["avg_prompt_chars"],
        "avg_verifier_prompt_chars_verified": verifier["avg_prompt_chars"],
        "avg_total_prompt_chars_per_question": avg_total_prompt_chars_per_question,
        "approx_total_prompt_tokens_per_question": avg_total_prompt_chars_per_question / 4,
        "avg_reader_evidence_chars": reader["avg_evidence_chars"],
        "avg_verifier_evidence_chars_verified": verifier["avg_evidence_chars"],
        "avg_retrieved_docs": reader["avg_retrieved_docs"] or retrieval["avg_retrieved_docs"],
        "avg_raw_retrieved_chars": retrieval["avg_raw_retrieved_chars"],
        "avg_response_chars": answers["avg_response_chars"],
        "training_required": spec.training_required,
        "notes": spec.notes,
    }


def load_rows(path, code_root):
    if not path:
        return []
    path = Path(path)
    if not path.is_absolute():
        path = code_root / path
    if not path.exists():
        return []
    return list(read_jsonl(path))


def hotpotqa_default_specs(code_root):
    rows = [
        {
            "method": "LLM-only / No Retrieval",
            "reader_prompts": "results/ircot_hotpotqa_test500_llm_only_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_llm_only_answers_qwen_500.jsonl",
            "retrieval_query_executions_per_question": 0.0,
            "notes": "No retrieval; lower-bound answer-only prompt.",
        },
        {
            "method": "One-step Dense RAG",
            "reader_prompts": "results/ircot_hotpotqa_test500_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 1.0,
            "notes": "One reader call per question.",
        },
        {
            "method": "Multi-query RAG",
            "reader_prompts": "results/ircot_hotpotqa_test500_multiquery_top10_decay050_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_multiquery_top10_decay050_extractive_answers_qwen_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_multiquery_top10_decay050_retrieval.jsonl",
            "retrieval_query_executions_per_question": 3.0,
            "notes": "Three rule-based dense retrieval queries; one reader call per question.",
        },
        {
            "method": "Single-query Reformulation RAG",
            "query_side_prompts": "results/ircot_hotpotqa_test500_single_query_reformulation_prompts.jsonl",
            "reader_prompts": "results/ircot_hotpotqa_test500_single_query_reformulation_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_single_query_reformulation_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 1.0,
            "notes": "One query-rewrite call plus one dense retrieval query and one reader call per question.",
        },
        {
            "method": "BM25 RAG",
            "reader_prompts": "results/ircot_hotpotqa_test500_bm25_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_bm25_top10_extractive_answers_qwen_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_bm25_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 1.0,
            "notes": "Lexical retrieval; one reader call per question.",
        },
        {
            "method": "BM25 + Dense Hybrid",
            "reader_prompts": "results/ircot_hotpotqa_test500_hybrid_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_hybrid_top10_extractive_answers_qwen_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_hybrid_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 2.0,
            "notes": "Dense plus BM25 retrieval followed by rank fusion; one reader call per question.",
        },
        {
            "method": "Evidence-guided Iterative Retrieval",
            "reader_prompts": "results/ircot_hotpotqa_test500_iterative_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 4.0,
            "notes": "First dense query plus three evidence-expanded retrieval queries; one final reader call per question.",
        },
        {
            "method": "HyDE-style RAG",
            "query_side_prompts": "results/ircot_hotpotqa_test500_hyde_generation_prompts.jsonl",
            "reader_prompts": "results/ircot_hotpotqa_test500_hyde_top10_extractive_prompts.jsonl",
            "answers": "results/ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 1.0,
            "notes": "One hypothetical-passage call plus one dense retrieval query and one reader call per question.",
        },
        {
            "method": "HyDE-style RAG + Conservative Verifier",
            "query_side_prompts": "results/ircot_hotpotqa_test500_hyde_generation_prompts.jsonl",
            "reader_prompts": "results/ircot_hotpotqa_test500_hyde_top10_extractive_prompts.jsonl",
            "verifier_prompts": "results/qwenmax_hyde_verification_prompts_conservative_risk120.jsonl",
            "answers": "results/qwenmax_hyde_selective_verification_eval_risk120_numeric_guarded_500.jsonl",
            "retrieval": "results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl",
            "retrieval_query_executions_per_question": 1.0,
            "notes": "HyDE retrieval plus selective verifier over risk cases only.",
        },
    ]
    specs = []
    for row in rows:
        specs.append(MethodSpec(
            method=row["method"],
            reader_prompt_rows=load_rows(row.get("reader_prompts"), code_root),
            query_side_prompt_rows=load_rows(row.get("query_side_prompts"), code_root),
            verifier_prompt_rows=load_rows(row.get("verifier_prompts"), code_root),
            answer_rows=load_rows(row.get("answers"), code_root),
            retrieval_rows=load_rows(row.get("retrieval"), code_root),
            retrieval_query_executions_per_question=row.get("retrieval_query_executions_per_question", 0.0),
            training_required="No",
            notes=row.get("notes", ""),
        ))
    return specs


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "n_examples",
        "reader_calls",
        "query_side_llm_calls",
        "retrieval_query_executions_per_question",
        "verifier_calls",
        "total_llm_calls",
        "calls_per_question",
        "verifier_coverage",
        "avg_reader_prompt_chars",
        "avg_query_side_prompt_chars",
        "avg_verifier_prompt_chars_verified",
        "avg_total_prompt_chars_per_question",
        "approx_total_prompt_tokens_per_question",
        "avg_reader_evidence_chars",
        "avg_verifier_evidence_chars_verified",
        "avg_retrieved_docs",
        "avg_raw_retrieved_chars",
        "avg_response_chars",
        "training_required",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row[key]) for key in fieldnames})


def write_latex(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Full lightweight call and retrieval-query accounting on the released IRCoT HotpotQA \\texttt{test\\_subsampled} split.}",
        "\\label{tab:efficiency_profile_full}",
        "\\scriptsize",
        "\\begin{threeparttable}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X r r r r r r r}",
        "\\toprule",
        "Method & Q-LLM/q & Ret.-q/q & Reader/q & Verifier/q & Total LLM/q & Prompt chars/q & Evidence chars/q \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"].replace("&", "\\&")
        note = row["notes"].replace("&", "\\&")
        query_side_calls_per_question = row["query_side_llm_calls"] / row["n_examples"] if row["n_examples"] else 0.0
        reader_calls_per_question = row["reader_calls"] / row["n_examples"] if row["n_examples"] else 0.0
        lines.append(
            f"{method} & "
            f"{query_side_calls_per_question:.2f} & "
            f"{row['retrieval_query_executions_per_question']:.2f} & "
            f"{reader_calls_per_question:.2f} & "
            f"{row['verifier_coverage']:.2f} & "
            f"{row['calls_per_question']:.2f} & "
            f"{row['avg_total_prompt_chars_per_question']:.0f} & "
            f"{row['avg_reader_evidence_chars']:.0f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabularx}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item Q-LLM/q counts query-side generation calls for matched-call reformulation or HyDE-style hypothetical-passage generation. Ret.-q/q counts retrieval query executions per question: Hybrid executes one dense and one BM25 retrieval, Multi-query executes three dense query variants, and Iterative Retrieval executes one first-round dense query plus three evidence-expanded dense queries. Total LLM/q counts query-side generation, reader answering, and selective verifier calls; it does not include retrieval query executions. Prompt chars/q reports average prompt characters per question from actual JSONL prompt files. Evidence chars/q reports serialized reader evidence length. The current logs do not store provider-side token usage, wall-clock latency, or billing. No method uses task-specific fine-tuning or model training.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 4 Lightweight Efficiency Profile",
        "",
        "This profile estimates lightweight cost from existing HotpotQA 500 prompt, answer, retrieval, and verifier files. Character counts come from actual JSONL prompts. Approximate token counts use a coarse 4 characters/token heuristic and should be treated as a readability aid rather than billing data. Retrieval query executions are counted separately from LLM calls.",
        "",
        "| Method | Query-side LLM/q | Retrieval queries/q | Reader/q | Verifier/q | Total LLM/q | Total LLM calls | Avg reader evidence chars | Avg total prompt chars/q | Approx prompt tokens/q | Training required |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        query_side_calls_per_question = row["query_side_llm_calls"] / row["n_examples"] if row["n_examples"] else 0.0
        reader_calls_per_question = row["reader_calls"] / row["n_examples"] if row["n_examples"] else 0.0
        lines.append(
            f"| {row['method']} | {query_side_calls_per_question:.4f} | "
            f"{row['retrieval_query_executions_per_question']:.4f} | "
            f"{reader_calls_per_question:.4f} | {row['verifier_coverage']:.4f} | "
            f"{row['calls_per_question']:.4f} | {row['total_llm_calls']} | "
            f"{row['avg_reader_evidence_chars']:.1f} | "
            f"{row['avg_total_prompt_chars_per_question']:.1f} | "
            f"{row['approx_total_prompt_tokens_per_question']:.1f} | "
            f"{row['training_required']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "HyDE-style RAG is not cost-free, but remains lightweight in this protocol: it adds one hypothetical-passage generation call and one dense retrieval query per question, and it does not require task-specific training or an agentic tool loop. Multi-query and Iterative Retrieval do not add LLM calls, but they do execute more retrieval queries per question. Conservative verification is selective: the final HyDE verifier uses 120 verifier calls over 500 examples, adding 0.24 calls per question rather than a verifier call for every answer.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", default="local-artifacts/lightweight_efficiency_profile_hotpotqa_test500.csv")
    parser.add_argument("--out_tex", default="paper/latex/table_efficiency_profile_full.tex")
    parser.add_argument("--out_md", default="local-artifacts/stage4_lightweight_efficiency_profile_hotpotqa_test500.md")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    specs = hotpotqa_default_specs(code_root)
    rows = [summarize_method(spec) for spec in specs]

    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_md = Path(args.out_md)
    if not out_csv.is_absolute():
        out_csv = code_root / out_csv
    if not out_tex.is_absolute():
        out_tex = code_root / out_tex
    if not out_md.is_absolute():
        out_md = code_root / out_md

    write_csv(out_csv, rows)
    write_latex(out_tex, rows)
    write_markdown(out_md, rows)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_tex}")
    print(f"Wrote {out_md}")
    print(f"Methods: {len(rows)}")


if __name__ == "__main__":
    main()
