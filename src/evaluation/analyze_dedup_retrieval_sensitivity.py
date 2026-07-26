import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from retriever.retrieve_bm25_topk import bm25_score, build_bm25_index, tokenize
from retriever.retrieve_hybrid_topk import fuse_retrieved_docs
from utils import read_jsonl, write_jsonl


DATASETS = [
    {
        "dataset": "HotpotQA",
        "prefix": "ircot_hotpotqa_test500",
        "dataset_dir": "datasets/ircot_hotpotqa_test500",
        "dense_original": "results/ircot_hotpotqa_test500_top10_retrieval.jsonl",
        "bm25_original": "results/ircot_hotpotqa_test500_bm25_top10_retrieval.jsonl",
        "hybrid_original": "results/ircot_hotpotqa_test500_hybrid_top10_retrieval.jsonl",
        "hyde_original": "results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl",
        "hyde_answers": "results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
    },
    {
        "dataset": "2WikiMultihopQA",
        "prefix": "ircot_2wiki_test500",
        "dataset_dir": "datasets/ircot_2wikimultihopqa_test500",
        "dense_original": "results/ircot_2wiki_test500_top10_retrieval.jsonl",
        "bm25_original": "results/ircot_2wiki_test500_bm25_top10_retrieval.jsonl",
        "hybrid_original": "results/ircot_2wiki_test500_hybrid_top10_retrieval.jsonl",
        "hyde_original": "results/ircot_2wiki_test500_hyde_top10_retrieval.jsonl",
        "hyde_answers": "results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
    },
]

METHOD_LABELS = {
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hyde": "HyDE",
}


def normalize_dedup_text(value):
    return " ".join(str(value or "").split()).casefold()


def dedup_key(doc):
    return (
        normalize_dedup_text(doc.get("title", "")),
        normalize_dedup_text(doc.get("text", "")),
    )


def truncated_hyde_document(row, max_hyde_chars=900):
    hyde_document = row.get("prediction", "").strip()
    if len(hyde_document) > max_hyde_chars:
        hyde_document = hyde_document[:max_hyde_chars].rstrip() + "..."
    return hyde_document


def build_hyde_query_text(row, max_hyde_chars=900):
    hyde_document = truncated_hyde_document(row, max_hyde_chars=max_hyde_chars)
    return f"{row['question']}\n\nHypothetical supporting passage:\n{hyde_document}".strip()


def deduplicate_documents(docs):
    deduped_by_key = OrderedDict()
    for doc in docs:
        key = dedup_key(doc)
        if key not in deduped_by_key:
            kept = dict(doc)
            kept["duplicate_count"] = 0
            kept["duplicate_doc_ids"] = []
            kept["duplicate_source_question_ids"] = []
            deduped_by_key[key] = kept

        kept = deduped_by_key[key]
        kept["duplicate_count"] += 1
        kept["duplicate_doc_ids"].append(doc.get("doc_id"))
        kept["duplicate_source_question_ids"].append(doc.get("source_question_id"))

    deduped = list(deduped_by_key.values())
    stats = {
        "input_docs": len(docs),
        "dedup_docs": len(deduped),
        "removed_exact_duplicates": len(docs) - len(deduped),
    }
    return deduped, stats


def support_titles(row):
    facts = row.get("supporting_facts", {})
    if isinstance(facts, dict):
        return set(facts.get("title", []))
    return set()


def retrieval_metrics(rows):
    total = 0
    any_hit = 0
    all_hit = 0
    recall_sum = 0.0
    for row in rows:
        gold_titles = support_titles(row)
        retrieved_titles = {doc.get("title") for doc in row.get("retrieved", [])}
        hit_titles = gold_titles & retrieved_titles
        total += 1
        if hit_titles:
            any_hit += 1
        if gold_titles and gold_titles.issubset(retrieved_titles):
            all_hit += 1
        if gold_titles:
            recall_sum += len(hit_titles) / len(gold_titles)
    if total == 0:
        return {"n": 0, "any_hit": 0.0, "all_hit": 0.0, "support_recall": 0.0}
    return {
        "n": total,
        "any_hit": any_hit / total,
        "all_hit": all_hit / total,
        "support_recall": recall_sum / total,
    }


def duplicate_slot_stats(row):
    retrieved = row.get("retrieved", [])
    title_keys = [normalize_dedup_text(doc.get("title", "")) for doc in retrieved]
    exact_keys = [dedup_key(doc) for doc in retrieved]
    return {
        "duplicate_title_slots": len(title_keys) - len(set(title_keys)),
        "duplicate_exact_slots": len(exact_keys) - len(set(exact_keys)),
    }


def aggregate_duplicate_slot_stats(rows):
    rows = list(rows)
    if not rows:
        return {
            "duplicate_title_slots_per_q": 0.0,
            "duplicate_exact_slots_per_q": 0.0,
        }
    title_slots = 0
    exact_slots = 0
    for row in rows:
        stats = duplicate_slot_stats(row)
        title_slots += stats["duplicate_title_slots"]
        exact_slots += stats["duplicate_exact_slots"]
    return {
        "duplicate_title_slots_per_q": title_slots / len(rows),
        "duplicate_exact_slots_per_q": exact_slots / len(rows),
    }


def format_retrieved_doc(doc, score, **extra):
    row = {
        "score": float(score),
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "text": doc["text"],
        "source_question_id": doc.get("source_question_id"),
    }
    row.update(extra)
    return row


def dense_retrieval_rows(questions, docs, query_texts, model, top_k, batch_size):
    import faiss

    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    scores, indices = index.search(query_embeddings, top_k)

    rows = []
    for question, score_row, index_row in zip(questions, scores, indices):
        retrieved = [
            format_retrieved_doc(docs[int(doc_idx)], score)
            for score, doc_idx in zip(score_row, index_row)
        ]
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "supporting_facts": question["supporting_facts"],
                "retrieved": retrieved,
            }
        )
    return rows


def bm25_retrieval_rows(questions, docs, top_k):
    tokenized_docs, document_frequency, average_doc_len = build_bm25_index(docs)
    rows = []
    for question in questions:
        query_tokens = tokenize(question["question"])
        scored = []
        for doc, doc_tokens in zip(docs, tokenized_docs):
            score = bm25_score(
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                document_frequency=document_frequency,
                total_docs=len(docs),
                average_doc_len=average_doc_len,
            )
            scored.append((score, doc))

        retrieved = [
            format_retrieved_doc(doc, score)
            for score, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        ]
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "supporting_facts": question["supporting_facts"],
                "retrieved": retrieved,
            }
        )
    return rows


def hyde_retrieval_rows(questions_by_id, docs, hyde_rows, model, top_k, batch_size):
    query_texts = [build_hyde_query_text(row) for row in hyde_rows]
    questions = [questions_by_id[row["id"]] for row in hyde_rows]
    rows = dense_retrieval_rows(
        questions=questions,
        docs=docs,
        query_texts=query_texts,
        model=model,
        top_k=top_k,
        batch_size=batch_size,
    )
    for row, hyde_row in zip(rows, hyde_rows):
        row["hyde_document"] = hyde_row.get("prediction", "").strip()
        row["retrieval_strategy"] = {
            "name": "hyde_dense",
            "index_policy": "deduplicate_exact_title_paragraph",
            "top_k": top_k,
            "query_mode": "question_plus_hypothetical",
        }
    return rows


def summarize_rows(dataset, method, index_policy, rows, corpus_stats):
    metrics = retrieval_metrics(rows)
    duplicate_stats = aggregate_duplicate_slot_stats(rows)
    return {
        "dataset": dataset,
        "method": method,
        "index_policy": index_policy,
        "n": metrics["n"],
        "any_hit_at10": metrics["any_hit"],
        "all_hit_at10": metrics["all_hit"],
        "recall_at10": metrics["support_recall"],
        "duplicate_title_slots_per_q": duplicate_stats["duplicate_title_slots_per_q"],
        "duplicate_exact_slots_per_q": duplicate_stats["duplicate_exact_slots_per_q"],
        **corpus_stats,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "index_policy",
        "n",
        "any_hit_at10",
        "all_hit_at10",
        "recall_at10",
        "duplicate_title_slots_per_q",
        "duplicate_exact_slots_per_q",
        "input_docs",
        "dedup_docs",
        "removed_exact_duplicates",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value, signed=False):
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def write_latex(path, rows):
    by_key = {(row["dataset"], row["method"], row["index_policy"]): row for row in rows}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval-side sensitivity to exact title--paragraph deduplication in the local joint index.}",
        r"\label{tab:dedup_retrieval_sensitivity}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X l r r r r r r}",
        r"\toprule",
        r"Dataset & Method & Retain all@10 & Dedup all@10 & $\Delta$ all & Retain recall & Dedup recall & $\Delta$ recall \\",
        r"\midrule",
    ]
    for dataset in ["HotpotQA", "2WikiMultihopQA"]:
        for method in ["Dense", "Hybrid", "HyDE"]:
            retained = by_key[(dataset, method, "retain_duplicates")]
            deduped = by_key[(dataset, method, "deduplicate_title_paragraph")]
            delta_all = deduped["all_hit_at10"] - retained["all_hit_at10"]
            delta_recall = deduped["recall_at10"] - retained["recall_at10"]
            lines.append(
                f"{dataset} & {method} & "
                f"{format_float(retained['all_hit_at10'])} & "
                f"{format_float(deduped['all_hit_at10'])} & "
                f"{format_float(delta_all, signed=True)} & "
                f"{format_float(retained['recall_at10'])} & "
                f"{format_float(deduped['recall_at10'])} & "
                f"{format_float(delta_recall, signed=True)} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Retain duplicates uses the canonical joint local index with repeated title--paragraph records preserved. Dedup removes exact normalized title--paragraph duplicates before retrieval, then rebuilds dense, BM25, hybrid, and HyDE retrieval artifacts with the same top-$k=10$ setting. Metrics still collapse duplicate supporting titles during evaluation.",
            r"\item The CSV artifact additionally reports average duplicate title slots and exact duplicate slots in the retrieved top-10 lists.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, rows):
    by_key = {(row["dataset"], row["method"], row["index_policy"]): row for row in rows}
    lines = [
        "# Exact Title-Paragraph Deduplication Sensitivity",
        "",
        "| Dataset | Method | Retain all@10 | Dedup all@10 | Delta all | Retain recall | Dedup recall | Delta recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["HotpotQA", "2WikiMultihopQA"]:
        for method in ["Dense", "Hybrid", "HyDE"]:
            retained = by_key[(dataset, method, "retain_duplicates")]
            deduped = by_key[(dataset, method, "deduplicate_title_paragraph")]
            lines.append(
                f"| {dataset} | {method} | "
                f"{format_float(retained['all_hit_at10'])} | "
                f"{format_float(deduped['all_hit_at10'])} | "
                f"{format_float(deduped['all_hit_at10'] - retained['all_hit_at10'], signed=True)} | "
                f"{format_float(retained['recall_at10'])} | "
                f"{format_float(deduped['recall_at10'])} | "
                f"{format_float(deduped['recall_at10'] - retained['recall_at10'], signed=True)} |"
            )
    lines.extend(
        [
            "",
            "The deduplicated index removes exact normalized `(title, paragraph text)` duplicates before retrieval.",
            "Reader answers are not regenerated; this is a retrieval-side sensitivity check.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dataset(config, code_root, model, top_k, batch_size):
    dataset_dir = code_root / config["dataset_dir"]
    results_dir = code_root / "results"
    questions = list(read_jsonl(dataset_dir / "questions.jsonl"))
    questions_by_id = {row["id"]: row for row in questions}
    docs = list(read_jsonl(dataset_dir / "corpus.jsonl"))
    dedup_docs, dedup_stats = deduplicate_documents(docs)

    dedup_corpus = dataset_dir / "corpus_dedup_title_text.jsonl"
    write_jsonl(dedup_corpus, dedup_docs)
    save_json(
        dataset_dir / "corpus_dedup_title_text_metadata.json",
        {
            "dedup_key": "normalized title + normalized paragraph text",
            **dedup_stats,
        },
    )

    summaries = []
    original_files = {
        "Dense": config["dense_original"],
        "Hybrid": config["hybrid_original"],
        "HyDE": config["hyde_original"],
    }
    retained_stats = {
        "input_docs": len(docs),
        "dedup_docs": len(dedup_docs),
        "removed_exact_duplicates": dedup_stats["removed_exact_duplicates"],
    }
    for method, file_name in original_files.items():
        rows = list(read_jsonl(code_root / file_name))
        summaries.append(
            summarize_rows(
                dataset=config["dataset"],
                method=method,
                index_policy="retain_duplicates",
                rows=rows,
                corpus_stats=retained_stats,
            )
        )

    dense_query_texts = [question["question"] for question in questions]
    dense_rows = dense_retrieval_rows(
        questions=questions,
        docs=dedup_docs,
        query_texts=dense_query_texts,
        model=model,
        top_k=top_k,
        batch_size=batch_size,
    )
    bm25_rows = bm25_retrieval_rows(questions, dedup_docs, top_k=top_k)
    hybrid_rows = []
    bm25_by_id = {row["id"]: row for row in bm25_rows}
    for dense_row in dense_rows:
        lexical_row = bm25_by_id[dense_row["id"]]
        hybrid_rows.append(
            {
                "id": dense_row["id"],
                "question": dense_row["question"],
                "answer": dense_row["answer"],
                "supporting_facts": dense_row["supporting_facts"],
                "retrieved": fuse_retrieved_docs(
                    dense_docs=dense_row["retrieved"],
                    lexical_docs=lexical_row["retrieved"],
                    top_k=top_k,
                ),
            }
        )

    hyde_answers = list(read_jsonl(code_root / config["hyde_answers"]))
    hyde_rows = hyde_retrieval_rows(
        questions_by_id=questions_by_id,
        docs=dedup_docs,
        hyde_rows=hyde_answers,
        model=model,
        top_k=top_k,
        batch_size=batch_size,
    )

    generated = {
        "Dense": dense_rows,
        "BM25": bm25_rows,
        "Hybrid": hybrid_rows,
        "HyDE": hyde_rows,
    }
    for method, rows in generated.items():
        method_slug = method.lower()
        out_path = results_dir / f"{config['prefix']}_dedup_title_text_{method_slug}_top{top_k}_retrieval.jsonl"
        write_jsonl(out_path, rows)

    dedup_summary_stats = {
        "input_docs": len(docs),
        "dedup_docs": len(dedup_docs),
        "removed_exact_duplicates": dedup_stats["removed_exact_duplicates"],
    }
    for method in ["Dense", "Hybrid", "HyDE"]:
        summaries.append(
            summarize_rows(
                dataset=config["dataset"],
                method=method,
                index_policy="deduplicate_title_paragraph",
                rows=generated[method],
                corpus_stats=dedup_summary_stats,
            )
        )

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_root", default=".")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_tex", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    code_root = Path(args.code_root)
    model = SentenceTransformer(args.model)
    all_rows = []
    for config in DATASETS:
        all_rows.extend(
            run_dataset(
                config=config,
                code_root=code_root,
                model=model,
                top_k=args.top_k,
                batch_size=args.batch_size,
            )
        )

    write_csv(args.out_csv, all_rows)
    write_markdown(args.out_md, all_rows)
    write_latex(args.out_tex, all_rows)
    print(f"Saved CSV to {args.out_csv}")
    print(f"Saved Markdown to {args.out_md}")
    print(f"Saved LaTeX table to {args.out_tex}")


if __name__ == "__main__":
    main()
