import argparse
import csv
import random
import statistics
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.analyze_retrieval_paired_tests import paired_retrieval_summary
from retriever.retrieve_query_reformulation_topk import build_reformulated_query_text
from utils import read_jsonl, write_jsonl


HYDE_SEPARATOR = "\n\nHypothetical supporting passage:\n"
REWRITE_SEPARATOR = "\n\nRewritten retrieval query:\n"


def token_count(text, tokenizer):
    return len(tokenizer.tokenize(str(text or "")))


def truncate_to_token_count(text, token_budget, tokenizer):
    if token_budget <= 0:
        return ""
    tokens = tokenizer.tokenize(str(text or ""))
    if len(tokens) <= token_budget:
        return str(text or "").strip()
    return tokenizer.convert_tokens_to_string(tokens[:token_budget]).strip()


def truncate_hyde_to_serialized_query_budget(question, hyde, target_tokens, tokenizer):
    """Return the longest HyDE prefix whose serialized q+h query fits target_tokens."""
    hyde_tokens = tokenizer.tokenize(str(hyde or ""))
    lo, hi = 0, len(hyde_tokens)
    best = ""
    question = str(question or "").strip()
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = tokenizer.convert_tokens_to_string(hyde_tokens[:mid]).strip()
        serialized = f"{question}{HYDE_SEPARATOR}{candidate}".strip()
        if token_count(serialized, tokenizer) <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_query_text(row, kind):
    question = row["question"].strip()
    rewrite = row.get("rewrite", "").strip()
    hyde = row.get("hyde", "").strip()
    if kind == "q":
        return question
    if kind == "r":
        return rewrite
    if kind == "q+r":
        return f"{question}{REWRITE_SEPARATOR}{rewrite}".strip()
    if kind == "h":
        return hyde
    if kind == "q+h":
        return f"{question}{HYDE_SEPARATOR}{hyde}".strip()
    raise ValueError(f"Unsupported query kind: {kind}")


def build_query_length_row(row, tokenizer, max_seq_length):
    stats = {"id": row["id"]}
    for kind in ["q", "r", "q+r", "h", "q+h"]:
        key = kind.replace("+", "_plus_")
        count = token_count(build_query_text(row, kind), tokenizer)
        stats[f"{key}_tokens"] = count
        stats[f"{key}_hits_cap"] = int(count >= max_seq_length)
    return stats


def summarize_lengths(length_rows):
    summary = []
    for kind, label in [
        ("q", "q"),
        ("r", "r"),
        ("q_plus_r", "q+r"),
        ("h", "h"),
        ("q_plus_h", "q+h"),
    ]:
        values = [row[f"{kind}_tokens"] for row in length_rows]
        values_sorted = sorted(values)
        p95_index = int(round(0.95 * (len(values_sorted) - 1))) if values_sorted else 0
        summary.append(
            {
                "query": label,
                "mean_tokens": statistics.mean(values) if values else 0.0,
                "median_tokens": statistics.median(values) if values else 0.0,
                "p95_tokens": values_sorted[p95_index] if values_sorted else 0,
                "hit_256_token_cap": sum(row[f"{kind}_hits_cap"] for row in length_rows),
            }
        )
    return summary


def format_retrieved_docs(score_row, index_row, docs):
    retrieved = []
    for score, doc_idx in zip(score_row, index_row):
        doc = docs[int(doc_idx)]
        retrieved.append(
            {
                "score": float(score),
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "source_question_id": doc["source_question_id"],
            }
        )
    return retrieved


def load_query_rows(rewrite_path, hyde_path, max_query_chars=300, max_hyde_chars=900):
    rewrites = {row["id"]: row for row in read_jsonl(rewrite_path)}
    hydes = {row["id"]: row for row in read_jsonl(hyde_path)}
    rows = []
    for qid in sorted(set(rewrites) & set(hydes)):
        rewrite_query = build_reformulated_query_text(
            rewrites[qid],
            max_query_chars=max_query_chars,
            query_mode="reformulation_only",
        )
        hyde_document = hydes[qid].get("prediction", "").strip()
        if len(hyde_document) > max_hyde_chars:
            hyde_document = hyde_document[:max_hyde_chars].rstrip() + "..."
        rows.append(
            {
                "id": qid,
                "question": rewrites[qid]["question"],
                "answer": rewrites[qid]["gold_answer"],
                "supporting_facts": rewrites[qid].get("supporting_facts", {}),
                "rewrite": rewrite_query,
                "hyde": hyde_document,
            }
        )
    return rows


def dense_retrieval_rows(rows, query_texts, docs, index, model, top_k, batch_size, strategy_name):
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    scores, indices = index.search(query_embeddings, top_k)
    output = []
    for row, query_text, score_row, index_row in zip(rows, query_texts, scores, indices):
        output.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "supporting_facts": row.get("supporting_facts", {}),
                "retrieved": format_retrieved_docs(score_row, index_row, docs),
                "query_text": query_text,
                "retrieval_strategy": {
                    "name": strategy_name,
                    "top_k": top_k,
                },
            }
        )
    return output


def build_length_matched_queries(rows, tokenizer, mode):
    query_texts = []
    detail_rows = []
    for row in rows:
        rewrite_tokens = token_count(row["rewrite"], tokenizer)
        rewrite_query_text = build_query_text(row, "q+r")
        rewrite_query_tokens = token_count(rewrite_query_text, tokenizer)
        if mode == "hypothetical_only":
            matched_hyde = truncate_to_token_count(row["hyde"], rewrite_tokens, tokenizer)
            query_text = matched_hyde
            target_tokens = rewrite_tokens
        elif mode == "question_plus_hypothetical":
            target_tokens = rewrite_query_tokens
            matched_hyde = truncate_hyde_to_serialized_query_budget(
                row["question"], row["hyde"], target_tokens, tokenizer
            )
            query_text = f"{row['question'].strip()}{HYDE_SEPARATOR}{matched_hyde}".strip()
        else:
            raise ValueError(f"Unsupported length-matched mode: {mode}")
        query_tokens = token_count(query_text, tokenizer)
        query_texts.append(query_text)
        detail_rows.append(
            {
                "id": row["id"],
                "rewrite_tokens": rewrite_tokens,
                "rewrite_query_tokens": rewrite_query_tokens,
                "target_tokens": target_tokens,
                "original_hyde_tokens": token_count(row["hyde"], tokenizer),
                "matched_hyde_tokens": token_count(matched_hyde, tokenizer),
                "matched_query_tokens": query_tokens,
                "question_preserved": int(query_text.startswith(row["question"].strip())),
                "matched_hyde": matched_hyde,
            }
        )
    return query_texts, detail_rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    return f"{value:.4f}"


def write_markdown(path, length_summary, retrieval_summary):
    lines = [
        "# Query Length and Length-matched HyDE Retrieval Sensitivity",
        "",
        "This no-new-LLM analysis uses frozen HotpotQA query-generation artifacts. Token counts use the MiniLM tokenizer. The hypothetical-only control truncates each hypothetical passage to the corresponding rewritten-query token count. The question-plus-hypothetical control truncates only the hypothetical passage so that the serialized query has no more MiniLM tokens than the corresponding question-plus-rewrite query.",
        "",
        "## Query Token Lengths",
        "",
        "| Query | Mean tokens | Median | P95 | Hit 256-token cap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in length_summary:
        lines.append(
            f"| {row['query']} | {format_float(row['mean_tokens'])} | "
            f"{format_float(row['median_tokens'])} | {row['p95_tokens']} | {row['hit_256_token_cap']} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval Sensitivity",
            "",
            "| Comparator | Target | n | Target any-hit | Target all-hit | Target recall | Delta all-hit [95% CI] | Delta recall [95% CI] |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in retrieval_summary:
        lines.append(
            f"| {row['baseline']} | {row['target']} | {row['n']} | "
            f"{format_float(row['target_any_hit'])} | {format_float(row['target_all_hit'])} | "
            f"{format_float(row['target_support_recall'])} | "
            f"{format_float(row['delta_all_hit'])} [{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}] | "
            f"{format_float(row['delta_support_recall'])} [{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}] |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text):
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write_latex(path, length_summary, retrieval_summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Query-token length audit and length-matched HyDE retrieval sensitivity on HotpotQA. Token counts use the MiniLM tokenizer. The hypothetical-only control matches the rewritten-query length; the question-plus-hypothetical control truncates only the hypothetical passage so that the serialized query is length-matched to question plus rewrite. No new LLM calls are made.}",
        r"\label{tab:query_length_matched_hyde}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",
        r"\toprule",
        r"Query & Mean tokens & Median & P95 & Hits 256-token cap \\",
        r"\midrule",
    ]
    for row in length_summary:
        lines.append(
            f"{latex_escape(row['query'])} & {format_float(row['mean_tokens'])} & "
            f"{format_float(row['median_tokens'])} & {row['p95_tokens']} & {row['hit_256_token_cap']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\vspace{1mm}",
            r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrrrrr}",
            r"\toprule",
            r"Comparator & Target retrieval query & Any-hit@10 & All-hit@10 & Recall@10 & $\Delta$ All-hit@10 & $\Delta$ Recall@10 \\",
            r"\midrule",
        ]
    )
    for row in retrieval_summary:
        lines.append(
            f"{latex_escape(row['baseline'])} & {latex_escape(row['target'])} & "
            f"{format_float(row['target_any_hit'])} & "
            f"{format_float(row['target_all_hit'])} & "
            f"{format_float(row['target_support_recall'])} & "
            f"{format_float(row['delta_all_hit'])} [{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}] & "
            f"{format_float(row['delta_support_recall'])} [{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}] \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Deltas are paired against the comparator shown in the first column: the hypothetical-only control is compared with the single rewritten query, and the question-plus-hypothetical control is compared with the question-plus-rewritten-query control. The question-plus-hypothetical row preserves the original question and matches the serialized query length to the corresponding question-plus-rewrite query, up to tokenizer discreteness. The sensitivity controls show that matched-length rewriting does not recover the full HyDE result, so the safest claim is a richer-query-budget effect rather than document-like form alone.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_debug_examples(path, rows, h_details, qh_details, h_queries, qh_queries, tokenizer, sample_size, seed):
    rng = random.Random(seed)
    count = min(sample_size, len(rows))
    indices = sorted(rng.sample(range(len(rows)), count)) if count else []
    lines = [
        "# Length-matched HyDE Query Debug Examples",
        "",
        "These examples audit the actual strings sent to dense retrieval after length matching.",
        "",
    ]
    for idx in indices:
        row = rows[idx]
        h_detail = h_details[idx]
        qh_detail = qh_details[idx]
        lines.extend(
            [
                f"## {row['id']}",
                "",
                "Original question:",
                "",
                row["question"],
                "",
                f"Rewrite tokens: {h_detail['rewrite_tokens']}",
                "",
                "Rewrite:",
                "",
                row["rewrite"],
                "",
                f"Question + rewrite tokens: {qh_detail['rewrite_query_tokens']}",
                "",
                "Truncated hypothetical:",
                "",
                qh_detail["matched_hyde"],
                "",
                f"Matched hypothetical tokens: {qh_detail['matched_hyde_tokens']}",
                "",
                f"Final q+h length-matched query tokens: {token_count(qh_queries[idx], tokenizer)}",
                "",
                "Final q+h length-matched query:",
                "",
                qh_queries[idx],
                "",
                f"Question preserved: {bool(qh_detail['question_preserved'])}",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_answers", default="results/ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl")
    parser.add_argument("--hyde_answers", default="results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl")
    parser.add_argument("--rewrite_retrieval", default="results/ircot_hotpotqa_test500_single_query_reformulation_top10_retrieval.jsonl")
    parser.add_argument("--question_plus_rewrite_retrieval", default="results/ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_retrieval.jsonl")
    parser.add_argument("--index_dir", default="datasets/ircot_hotpotqa_test500/faiss_index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--debug_examples", type=int, default=20)
    args = parser.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    code_root = Path(__file__).resolve().parents[2]
    project_root = code_root.parent
    rows = load_query_rows(
        code_root / args.rewrite_answers,
        code_root / args.hyde_answers,
    )
    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer
    docs = list(read_jsonl(Path(code_root / args.index_dir) / "docstore.jsonl"))
    index = faiss.read_index(str(Path(code_root / args.index_dir) / "index.faiss"))

    length_rows = [build_query_length_row(row, tokenizer, args.max_seq_length) for row in rows]
    length_summary = summarize_lengths(length_rows)

    h_query_texts, h_details = build_length_matched_queries(rows, tokenizer, mode="hypothetical_only")
    qh_query_texts, qh_details = build_length_matched_queries(rows, tokenizer, mode="question_plus_hypothetical")
    h_retrieval = dense_retrieval_rows(
        rows, h_query_texts, docs, index, model, args.top_k, args.batch_size, "length_matched_hypothetical_only"
    )
    qh_retrieval = dense_retrieval_rows(
        rows, qh_query_texts, docs, index, model, args.top_k, args.batch_size, "length_matched_question_plus_hypothetical"
    )
    for retrieval_row, detail in zip(h_retrieval, h_details):
        retrieval_row["length_matching"] = detail
    for retrieval_row, detail in zip(qh_retrieval, qh_details):
        retrieval_row["length_matching"] = detail

    results_dir = code_root / "results"
    h_path = results_dir / "ircot_hotpotqa_test500_length_matched_hyde_hypothetical_only_top10_retrieval.jsonl"
    qh_path = results_dir / "ircot_hotpotqa_test500_length_matched_hyde_q_plus_h_top10_retrieval.jsonl"
    write_jsonl(h_path, h_retrieval)
    write_jsonl(qh_path, qh_retrieval)

    rewrite_by_id = {row["id"]: row for row in read_jsonl(code_root / args.rewrite_retrieval)}
    question_plus_rewrite_by_id = {
        row["id"]: row for row in read_jsonl(code_root / args.question_plus_rewrite_retrieval)
    }
    h_by_id = {row["id"]: row for row in h_retrieval}
    qh_by_id = {row["id"]: row for row in qh_retrieval}
    retrieval_summary = [
        paired_retrieval_summary(
            "HotpotQA",
            "Single rewritten query",
            "Length-matched hypothetical only",
            rewrite_by_id,
            h_by_id,
            iterations=args.iterations,
            seed=args.seed,
        ),
        paired_retrieval_summary(
            "HotpotQA",
            "Question + rewritten query",
            "Length-matched question + hypothetical",
            question_plus_rewrite_by_id,
            qh_by_id,
            iterations=args.iterations,
            seed=args.seed + 10,
        ),
    ]

    notes_dir = project_root / "local-artifacts"
    write_csv(notes_dir / "query_token_length_stats.csv", length_summary)
    write_csv(notes_dir / "query_token_length_per_example.csv", length_rows)
    write_csv(notes_dir / "length_matched_hyde_retrieval_sensitivity.csv", retrieval_summary)
    write_markdown(notes_dir / "length_matched_hyde_retrieval_sensitivity.md", length_summary, retrieval_summary)
    write_debug_examples(
        notes_dir / "length_matched_hyde_query_debug_examples.md",
        rows,
        h_details,
        qh_details,
        h_query_texts,
        qh_query_texts,
        tokenizer,
        args.debug_examples,
        args.seed,
    )
    write_latex(project_root / "paper/latex" / "table_query_length_matched_hyde.tex", length_summary, retrieval_summary)

    for row in length_summary:
        print(
            f"{row['query']}: mean={format_float(row['mean_tokens'])}, "
            f"median={format_float(row['median_tokens'])}, p95={row['p95_tokens']}, "
            f"cap_hits={row['hit_256_token_cap']}"
        )
    for row in retrieval_summary:
        print(
            f"{row['target']}: all-hit={format_float(row['target_all_hit'])}, "
            f"recall={format_float(row['target_support_recall'])}, "
            f"delta-all={format_float(row['delta_all_hit'])} "
            f"[{format_float(row['delta_all_hit_ci_low'])}, {format_float(row['delta_all_hit_ci_high'])}], "
            f"delta-recall={format_float(row['delta_support_recall'])} "
            f"[{format_float(row['delta_support_recall_ci_low'])}, {format_float(row['delta_support_recall_ci_high'])}]"
        )


if __name__ == "__main__":
    main()
