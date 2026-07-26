import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


DATASETS = {
    "hotpotqa": {
        "name": "HotpotQA",
        "questions": "datasets/ircot_hotpotqa_test500/questions.jsonl",
        "corpus": "datasets/ircot_hotpotqa_test500/corpus.jsonl",
        "out_prefix": "ircot_hotpotqa_test500_equal_budget_bge_base",
    },
    "2wiki": {
        "name": "2WikiMultihopQA",
        "questions": "datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "corpus": "datasets/ircot_2wikimultihopqa_test500/corpus.jsonl",
        "out_prefix": "ircot_2wiki_test500_equal_budget_bge_base",
    },
}


QUERY_MODES = [
    "keyword_expansion",
    "direct_rewrite",
    "question_decomposition",
    "document_like_passage",
]


def query_word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text or ""))


def build_equal_budget_query_text(row):
    return f"{row['question'].strip()}\n\nGenerated retrieval text:\n{row.get('prediction', '').strip()}".strip()


def summarize_generation_lengths(rows, default_mode=None):
    grouped = {}
    for row in rows:
        mode = row.get("equal_budget_query_mode") or default_mode or "unknown"
        grouped.setdefault(mode, []).append(query_word_count(row.get("prediction", "")))
    summary = {}
    for mode, lengths in grouped.items():
        summary[mode] = {
            "n": len(lengths),
            "mean_words": sum(lengths) / len(lengths),
            "min_words": min(lengths),
            "max_words": max(lengths),
        }
    return summary


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
                "source_question_id": doc.get("source_question_id", ""),
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
        any_hit += 1 if hit_titles else 0
        all_hit += 1 if gold_titles and gold_titles.issubset(retrieved_titles) else 0
        recall_sum += len(hit_titles) / len(gold_titles) if gold_titles else 0.0
    return {
        "n": total,
        "any_hit@10": any_hit / total,
        "all_support_hit@10": all_hit / total,
        "supporting_title_recall@10": recall_sum / total,
    }


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_doc_embeddings(dataset_key, docs, tokenizer, model, args, device):
    safe_model = args.model.replace("/", "__")
    cache_path = (
        Path(args.cache_dir)
        / f"{dataset_key}_{safe_model}_docmax{args.doc_max_length}.pt"
    )
    if cache_path.exists():
        print(f"[{dataset_key}] Loading cached corpus embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu").contiguous()
    print(f"[{dataset_key}] Encoding {len(docs)} corpus documents")
    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    embeddings = encode_texts(
        doc_texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.doc_batch_size,
        max_length=args.doc_max_length,
        device=device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def run_dataset(dataset_key, cfg, tokenizer, model, args, device):
    questions = {row["id"]: row for row in read_jsonl(resolve_path(cfg["questions"]))}
    docs = list(read_jsonl(resolve_path(cfg["corpus"])))
    doc_embeddings = load_doc_embeddings(dataset_key, docs, tokenizer, model, args, device)
    summary_rows = []
    length_rows = []

    for mode in args.query_modes:
        generated_path = Path(args.generated_dir) / f"{cfg['out_prefix']}_{mode}_generations.jsonl"
        generated_rows = list(read_jsonl(generated_path))
        missing = [row["id"] for row in generated_rows if row["id"] not in questions]
        if missing:
            raise ValueError(f"{dataset_key}/{mode} has {len(missing)} unknown question ids")

        length_summary = summarize_generation_lengths(generated_rows, default_mode=mode).get(mode, {})
        length_rows.append({"dataset": cfg["name"], "query_mode": mode, **length_summary})
        query_texts = [build_equal_budget_query_text(row) for row in generated_rows]
        query_embeddings = encode_texts(
            query_texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=args.query_batch_size,
            max_length=args.query_max_length,
            device=device,
        )
        scores = query_embeddings @ doc_embeddings.T
        top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=1)

        rows = []
        for generated, query_text, score_row, index_row in zip(
            generated_rows, query_texts, top_scores, top_indices
        ):
            question = questions[generated["id"]]
            rows.append(
                {
                    "id": generated["id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "supporting_facts": question.get("supporting_facts", {}),
                    "retrieved": format_retrieved_docs(score_row, index_row, docs),
                    "generated_retrieval_text": generated.get("prediction", "").strip(),
                    "retrieval_query_text": query_text,
                    "equal_budget_query_mode": mode,
                    "generation_word_count": query_word_count(generated.get("prediction", "")),
                    "retrieval_strategy": {
                        "name": "equal_budget_bge_base_query_composition",
                        "encoder": args.model,
                        "query_mode": mode,
                        "top_k": args.top_k,
                        "query_serialization": "question + Generated retrieval text label + generated text",
                        "doc_max_length": args.doc_max_length,
                        "query_max_length": args.query_max_length,
                    },
                }
            )

        out_path = Path(args.results_dir) / f"{cfg['out_prefix']}_{mode}_top{args.top_k}_retrieval.jsonl"
        write_jsonl(out_path, rows)
        metrics = retrieval_metrics(rows)
        summary_rows.append(
            {
                "dataset": cfg["name"],
                "encoder": args.model,
                "query_mode": mode,
                "mean_generation_words": length_summary.get("mean_words", 0.0),
                "retrieval_file": str(out_path),
                **metrics,
            }
        )
        print(
            f"[{dataset_key}] {mode}: all={metrics['all_support_hit@10']:.4f}, "
            f"recall={metrics['supporting_title_recall@10']:.4f}, "
            f"mean_words={length_summary.get('mean_words', 0.0):.1f}"
        )

    return summary_rows, length_rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=["hotpotqa", "2wiki"])
    parser.add_argument("--generated_dir", default="results/equal_budget_query_generations")
    parser.add_argument("--results_dir", default="results/equal_budget_query_retrieval")
    parser.add_argument("--summary_csv", default="results/equal_budget_query_retrieval_summary.csv")
    parser.add_argument("--length_csv", default="results/equal_budget_query_length_summary.csv")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--cache_dir", default="results/cache/equal_budget_bge_base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--query_modes", nargs="+", choices=QUERY_MODES, default=QUERY_MODES)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    all_summary_rows = []
    all_length_rows = []
    for dataset_key in args.datasets:
        summary_rows, length_rows = run_dataset(dataset_key, DATASETS[dataset_key], tokenizer, model, args, device)
        all_summary_rows.extend(summary_rows)
        all_length_rows.extend(length_rows)

    write_csv(
        args.summary_csv,
        all_summary_rows,
        [
            "dataset",
            "encoder",
            "query_mode",
            "n",
            "mean_generation_words",
            "any_hit@10",
            "all_support_hit@10",
            "supporting_title_recall@10",
            "retrieval_file",
        ],
    )
    write_csv(
        args.length_csv,
        all_length_rows,
        ["dataset", "query_mode", "n", "mean_words", "min_words", "max_words"],
    )
    print(f"Saved summary to {args.summary_csv}")
    print(json.dumps(all_summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
