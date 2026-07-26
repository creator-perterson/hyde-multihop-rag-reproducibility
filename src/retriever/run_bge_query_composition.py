import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


DATASETS = {
    "hotpotqa": {
        "name": "HotpotQA",
        "questions": "datasets/ircot_hotpotqa_test500/questions.jsonl",
        "corpus": "datasets/ircot_hotpotqa_test500/corpus.jsonl",
        "hyde_answers": "results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
        "query_answers": "results/ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_hotpotqa_test500_bge_base",
    },
    "2wiki": {
        "name": "2WikiMultihopQA",
        "questions": "datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "corpus": "datasets/ircot_2wikimultihopqa_test500/corpus.jsonl",
        "hyde_answers": "results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
        "query_answers": "results/ircot_2wiki_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_2wiki_test500_bge_base",
    },
}


QUERY_MODES = [
    ("question_only", "Question only"),
    ("single_rewritten_query", "Single rewritten query"),
    ("question_plus_rewritten_query", "Question + rewritten query"),
    ("hypothetical_only", "Hypothetical passage only"),
    ("question_plus_hypothetical", "Question + hypothetical passage"),
]

REWRITE_QUERY_MODES = {"single_rewritten_query", "question_plus_rewritten_query"}


def truncate_text(text, max_chars):
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def build_query_text(question_row, hyde_row, rewrite_row, mode, max_hyde_chars, max_query_chars):
    question = question_row["question"].strip()
    hyde = truncate_text(hyde_row.get("prediction", ""), max_hyde_chars)
    rewrite = truncate_text(rewrite_row.get("prediction", ""), max_query_chars)
    if not rewrite:
        rewrite = question

    if mode == "question_only":
        return question
    if mode == "single_rewritten_query":
        return rewrite
    if mode == "question_plus_rewritten_query":
        return f"{question}\n\nRewritten retrieval query:\n{rewrite}".strip()
    if mode == "hypothetical_only":
        return hyde
    if mode == "question_plus_hypothetical":
        return f"{question}\n\nHypothetical supporting passage:\n{hyde}".strip()
    raise ValueError(f"Unsupported query mode: {mode}")


def encode_texts(texts, tokenizer, model, batch_size, max_length, device):
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = output.last_hidden_state[:, 0]
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).contiguous()


def format_retrieved_docs(scores, indices, docs):
    retrieved = []
    for score, doc_idx in zip(scores.tolist(), indices.tolist()):
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


def retrieval_metrics(rows):
    total = 0
    any_hit = 0
    all_hit = 0
    recall_sum = 0.0
    for row in rows:
        gold_titles = set(row["supporting_facts"]["title"])
        retrieved_titles = {doc["title"] for doc in row["retrieved"]}
        hit_titles = gold_titles & retrieved_titles
        total += 1
        if hit_titles:
            any_hit += 1
        if gold_titles and gold_titles.issubset(retrieved_titles):
            all_hit += 1
        if gold_titles:
            recall_sum += len(hit_titles) / len(gold_titles)
    return {
        "n": total,
        "any_hit@10": any_hit / total,
        "all_support_hit@10": all_hit / total,
        "supporting_title_recall@10": recall_sum / total,
    }


def run_dataset(dataset_key, cfg, tokenizer, model, args, device, query_modes):
    questions = list(read_jsonl(cfg["questions"]))
    docs = list(read_jsonl(cfg["corpus"]))
    hyde_by_id = {row["id"]: row for row in read_jsonl(cfg["hyde_answers"])}
    needs_rewrite = any(mode in REWRITE_QUERY_MODES for mode, _ in query_modes)
    rewrite_by_id = {}
    if needs_rewrite:
        rewrite_path = Path(cfg["query_answers"])
        if not rewrite_path.exists():
            raise FileNotFoundError(
                f"Missing query reformulation file for {dataset_key}: {rewrite_path}"
            )
        rewrite_by_id = {row["id"]: row for row in read_jsonl(rewrite_path)}

    missing_hyde = [row["id"] for row in questions if row["id"] not in hyde_by_id]
    missing_rewrite = (
        [row["id"] for row in questions if row["id"] not in rewrite_by_id]
        if needs_rewrite
        else []
    )
    if missing_hyde or missing_rewrite:
        raise ValueError(
            f"Missing generated query artifacts for {dataset_key}: "
            f"hyde={len(missing_hyde)}, rewrite={len(missing_rewrite)}"
        )

    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    cache_path = None
    if args.doc_embedding_cache_dir:
        safe_model = args.model.replace("/", "__")
        cache_path = (
            Path(args.doc_embedding_cache_dir)
            / f"{dataset_key}_{safe_model}_docmax{args.doc_max_length}.pt"
        )
    if cache_path and cache_path.exists():
        print(f"[{dataset_key}] Loading cached corpus embeddings from {cache_path}")
        doc_embeddings = torch.load(cache_path, map_location="cpu").contiguous()
    else:
        print(f"[{dataset_key}] Encoding {len(docs)} corpus documents")
        doc_embeddings = encode_texts(
            doc_texts,
            tokenizer,
            model,
            batch_size=args.doc_batch_size,
            max_length=args.doc_max_length,
            device=device,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(doc_embeddings, cache_path)
            print(f"[{dataset_key}] Saved corpus embeddings to {cache_path}")

    summary_rows = []
    for mode, label in query_modes:
        print(f"[{dataset_key}] Retrieving mode: {label}")
        query_texts = [
            build_query_text(
                question,
                hyde_by_id[question["id"]],
                rewrite_by_id.get(question["id"], {}),
                mode=mode,
                max_hyde_chars=args.max_hyde_chars,
                max_query_chars=args.max_query_chars,
            )
            for question in questions
        ]
        query_embeddings = encode_texts(
            query_texts,
            tokenizer,
            model,
            batch_size=args.query_batch_size,
            max_length=args.query_max_length,
            device=device,
        )
        scores = query_embeddings @ doc_embeddings.T
        top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=1)

        rows = []
        for question, query_text, score_row, index_row in zip(
            questions, query_texts, top_scores, top_indices
        ):
            hyde_document = hyde_by_id[question["id"]].get("prediction", "").strip()
            rewrite = rewrite_by_id.get(question["id"], {}).get("prediction", "").strip()
            rows.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "supporting_facts": question.get("supporting_facts", {}),
                    "retrieved": format_retrieved_docs(score_row, index_row, docs),
                    "hyde_document": hyde_document,
                    "reformulated_query": rewrite,
                    "retrieval_query_text": query_text,
                    "retrieval_strategy": {
                        "name": "bge_base_query_composition_dense",
                        "encoder": args.model,
                        "query_mode": mode,
                        "top_k": args.top_k,
                        "max_hyde_chars": args.max_hyde_chars,
                        "max_query_chars": args.max_query_chars,
                    },
                }
            )

        out_path = Path(args.results_dir) / f"{cfg['out_prefix']}_{mode}_top10_retrieval.jsonl"
        write_jsonl(out_path, rows)
        metrics = retrieval_metrics(rows)
        summary_rows.append(
            {
                "dataset": cfg["name"],
                "encoder": args.model,
                "query_mode": mode,
                "query_input": label,
                "retrieval_file": str(out_path),
                **metrics,
            }
        )
        print(
            f"[{dataset_key}] {label}: "
            f"any={metrics['any_hit@10']:.4f}, "
            f"all={metrics['all_support_hit@10']:.4f}, "
            f"recall={metrics['supporting_title_recall@10']:.4f}"
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=["hotpotqa", "2wiki"])
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--summary_csv", default="results/bge_base_query_composition_retrieval_summary.csv")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--max_hyde_chars", type=int, default=900)
    parser.add_argument("--max_query_chars", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--doc_embedding_cache_dir", default=None)
    parser.add_argument(
        "--query_modes",
        nargs="+",
        choices=[mode for mode, _ in QUERY_MODES],
        default=[mode for mode, _ in QUERY_MODES],
    )
    args = parser.parse_args()
    selected_modes = [(mode, label) for mode, label in QUERY_MODES if mode in args.query_modes]

    device = torch.device(args.device)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    all_summary_rows = []
    for dataset_key in args.datasets:
        all_summary_rows.extend(
            run_dataset(
                dataset_key,
                DATASETS[dataset_key],
                tokenizer,
                model,
                args,
                device,
                selected_modes,
            )
        )

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "encoder",
                "query_mode",
                "query_input",
                "n",
                "any_hit@10",
                "all_support_hit@10",
                "supporting_title_recall@10",
                "retrieval_file",
            ],
        )
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"Saved summary to {summary_path}")
    print(json.dumps(all_summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
