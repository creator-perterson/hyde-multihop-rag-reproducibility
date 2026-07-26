import argparse
import csv
import json
import random
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
        "base_corpus": "datasets/ircot_hotpotqa_test500/corpus.jsonl",
        "processed_train": "../_external/paper_repos/ircot/processed_data/hotpotqa/train.jsonl",
        "hyde_answers": "results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
        "rewrite_answers": "results/ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_hotpotqa_test500",
    },
    "2wiki": {
        "name": "2WikiMultihopQA",
        "questions": "datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "base_corpus": "datasets/ircot_2wikimultihopqa_test500/corpus.jsonl",
        "processed_train": "../_external/paper_repos/ircot/processed_data/2wikimultihopqa/train.jsonl",
        "hyde_answers": "results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
        "rewrite_answers": "results/ircot_2wiki_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_2wiki_test500",
    },
}


QUERY_MODES = [
    ("question_only", "Question only"),
    ("single_rewritten_query", "Single rewritten query"),
    ("question_plus_hypothetical", "Question + hypothetical passage"),
]

DEFAULT_CORPUS_SIZES = [50000, 100000]


def normalize_space(text):
    return " ".join((text or "").strip().lower().split())


def document_key(doc):
    return (normalize_space(doc.get("title", "")), normalize_space(doc.get("text", "")))


def context_to_document(row, context, source_split):
    question_id = row.get("question_id") or row.get("id")
    idx = context.get("idx", 0)
    return {
        "doc_id": f"{source_split}:{question_id}:{idx}",
        "title": context.get("title", ""),
        "text": context.get("paragraph_text") or context.get("text", ""),
        "source_question_id": question_id,
        "source_split": source_split,
    }


def iter_processed_context_documents(path, source_split):
    for row in read_jsonl(path):
        for context in row.get("contexts", []):
            yield context_to_document(row, context, source_split=source_split)


def build_expanded_corpus(base_docs, source_rows, target_size, source_split):
    docs = []
    seen = set()
    skipped_duplicates = 0
    for doc in base_docs:
        key = document_key(doc)
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        docs.append(dict(doc))

    for row in source_rows:
        for context in row.get("contexts", []):
            if len(docs) >= target_size:
                break
            doc = context_to_document(row, context, source_split=source_split)
            key = document_key(doc)
            if key in seen:
                skipped_duplicates += 1
                continue
            seen.add(key)
            docs.append(doc)
        if len(docs) >= target_size:
            break

    stats = {
        "target_size": target_size,
        "base_docs": len(base_docs),
        "expanded_docs": len(docs),
        "added_distractors": max(0, len(docs) - len(base_docs)),
        "skipped_duplicates": skipped_duplicates,
    }
    return docs, stats


def load_or_build_expanded_corpus(base_corpus_path, train_path, target_size, out_path, seed):
    if out_path.exists():
        return list(read_jsonl(out_path)), {"loaded_from_cache": str(out_path)}

    base_docs = list(read_jsonl(base_corpus_path))
    train_rows = list(read_jsonl(train_path))
    random.Random(seed).shuffle(train_rows)
    docs, stats = build_expanded_corpus(
        base_docs=base_docs,
        source_rows=train_rows,
        target_size=target_size,
        source_split="train",
    )
    if len(docs) < target_size:
        raise ValueError(
            f"Only built {len(docs)} docs for target_size={target_size}; "
            f"source pool may be too small: {train_path}"
        )
    write_jsonl(out_path, docs)
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return docs, stats


def truncate_text(text, max_chars):
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def build_query_text(question_row, hyde_row, rewrite_row, mode, max_hyde_chars, max_query_chars):
    question = question_row["question"].strip()
    hyde = truncate_text(hyde_row.get("prediction", ""), max_hyde_chars)
    rewrite = truncate_text(rewrite_row.get("prediction", ""), max_query_chars) or question
    if mode == "question_only":
        return question
    if mode == "single_rewritten_query":
        return rewrite
    if mode == "question_plus_hypothetical":
        return f"{question}\n\nHypothetical supporting passage:\n{hyde}".strip()
    raise ValueError(f"Unsupported query mode: {mode}")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, batch_size, max_length, device, pooling):
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
            if pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            elif pooling == "mean":
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).contiguous()


def load_or_encode_docs(docs, tokenizer, model, cache_path, args, device):
    if cache_path.exists():
        print(f"Loading cached document embeddings: {cache_path}")
        return torch.load(cache_path, map_location="cpu").contiguous()

    texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    print(f"Encoding {len(texts)} documents")
    embeddings = encode_texts(
        texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.doc_batch_size,
        max_length=args.doc_max_length,
        device=device,
        pooling=args.pooling,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


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
                "source_split": doc.get("source_split", ""),
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


def per_example_retrieval_metrics(rows):
    metrics = {}
    for row in rows:
        gold_titles = set(row["supporting_facts"]["title"])
        retrieved_titles = {doc["title"] for doc in row["retrieved"]}
        hit_titles = gold_titles & retrieved_titles
        metrics[row["id"]] = {
            "any_hit@10": 1.0 if hit_titles else 0.0,
            "all_support_hit@10": 1.0 if gold_titles and gold_titles.issubset(retrieved_titles) else 0.0,
            "supporting_title_recall@10": (len(hit_titles) / len(gold_titles)) if gold_titles else 0.0,
        }
    return metrics


def bootstrap_delta_ci(deltas, iterations=2000, seed=13, alpha=0.05):
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        sample_sum = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(sample_sum / n)
    means.sort()
    lo_idx = int((alpha / 2) * iterations)
    hi_idx = int((1 - alpha / 2) * iterations) - 1
    return means[lo_idx], means[max(lo_idx, min(hi_idx, iterations - 1))]


def paired_delta_summary(baseline, target, metric_name, iterations=2000, seed=13):
    ids = sorted(set(baseline) & set(target))
    deltas = [target[item_id][metric_name] - baseline[item_id][metric_name] for item_id in ids]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    ci_low, ci_high = bootstrap_delta_ci(deltas, iterations=iterations, seed=seed)
    return {
        f"{metric_name}_delta": mean_delta,
        f"{metric_name}_ci_low": ci_low,
        f"{metric_name}_ci_high": ci_high,
    }


def add_pairwise_deltas(summary_rows, per_example_by_mode, args):
    target_mode = "question_plus_hypothetical"
    comparisons = [
        ("vs_question", "question_only"),
        ("vs_rewrite", "single_rewritten_query"),
    ]
    if target_mode not in per_example_by_mode:
        return summary_rows
    target = per_example_by_mode[target_mode]
    for row in summary_rows:
        if row["query_mode"] != target_mode:
            continue
        for prefix, baseline_mode in comparisons:
            if baseline_mode not in per_example_by_mode:
                continue
            baseline = per_example_by_mode[baseline_mode]
            for offset, metric_name in enumerate(["all_support_hit@10", "supporting_title_recall@10"]):
                stats = paired_delta_summary(
                    baseline,
                    target,
                    metric_name,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + offset + (100 if prefix == "vs_rewrite" else 0),
                )
                row[f"{prefix}_{metric_name}_delta"] = stats[f"{metric_name}_delta"]
                row[f"{prefix}_{metric_name}_ci_low"] = stats[f"{metric_name}_ci_low"]
                row[f"{prefix}_{metric_name}_ci_high"] = stats[f"{metric_name}_ci_high"]
    return summary_rows


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def run_dataset(dataset_key, cfg, args, tokenizer, model, device):
    questions = list(read_jsonl(resolve_path(cfg["questions"])))
    hyde_by_id = {row["id"]: row for row in read_jsonl(resolve_path(cfg["hyde_answers"]))}
    rewrite_by_id = {row["id"]: row for row in read_jsonl(resolve_path(cfg["rewrite_answers"]))}
    missing = [row["id"] for row in questions if row["id"] not in hyde_by_id or row["id"] not in rewrite_by_id]
    if missing:
        raise ValueError(f"Missing generated query artifacts for {dataset_key}: {len(missing)}")

    all_summary_rows = []
    for target_size in args.corpus_sizes:
        corpus_path = Path(args.artifact_dir) / f"{cfg['out_prefix']}_corpus_scale_{target_size}.jsonl"
        docs, corpus_stats = load_or_build_expanded_corpus(
            base_corpus_path=resolve_path(cfg["base_corpus"]),
            train_path=resolve_path(cfg["processed_train"]),
            target_size=target_size,
            out_path=corpus_path,
            seed=args.seed,
        )
        print(f"[{dataset_key} {target_size}] corpus stats: {corpus_stats}")

        safe_model = args.model.replace("/", "__")
        cache_path = (
            Path(args.artifact_dir)
            / "embedding_cache"
            / f"{dataset_key}_{safe_model}_{target_size}_{args.pooling}_docmax{args.doc_max_length}.pt"
        )
        doc_embeddings = load_or_encode_docs(docs, tokenizer, model, cache_path, args, device)

        size_summary_rows = []
        per_example_by_mode = {}
        for mode, label in QUERY_MODES:
            print(f"[{dataset_key} {target_size}] Retrieving: {label}")
            query_texts = [
                build_query_text(
                    question,
                    hyde_by_id[question["id"]],
                    rewrite_by_id[question["id"]],
                    mode=mode,
                    max_hyde_chars=args.max_hyde_chars,
                    max_query_chars=args.max_query_chars,
                )
                for question in questions
            ]
            query_embeddings = encode_texts(
                query_texts,
                tokenizer=tokenizer,
                model=model,
                batch_size=args.query_batch_size,
                max_length=args.query_max_length,
                device=device,
                pooling=args.pooling,
            )
            scores = query_embeddings @ doc_embeddings.T
            top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=1)

            rows = []
            for question, query_text, score_row, index_row in zip(
                questions, query_texts, top_scores, top_indices
            ):
                rows.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "supporting_facts": question.get("supporting_facts", {}),
                        "retrieved": format_retrieved_docs(score_row, index_row, docs),
                        "retrieval_query_text": query_text,
                        "hyde_document": hyde_by_id[question["id"]].get("prediction", "").strip(),
                        "reformulated_query": rewrite_by_id[question["id"]].get("prediction", "").strip(),
                        "retrieval_strategy": {
                            "name": "corpus_scale_dense_stress",
                            "dataset": cfg["name"],
                            "encoder": args.model,
                            "pooling": args.pooling,
                            "corpus_size": target_size,
                            "query_mode": mode,
                            "top_k": args.top_k,
                            "max_hyde_chars": args.max_hyde_chars,
                            "max_query_chars": args.max_query_chars,
                        },
                    }
                )

            out_path = (
                Path(args.results_dir)
                / f"{cfg['out_prefix']}_corpus_scale_{target_size}_{mode}_top{args.top_k}_retrieval.jsonl"
            )
            write_jsonl(out_path, rows)
            metrics = retrieval_metrics(rows)
            per_example_by_mode[mode] = per_example_retrieval_metrics(rows)
            size_summary_rows.append(
                    {
                        "dataset": cfg["name"],
                        "dataset_key": dataset_key,
                        "encoder": args.model,
                        "pooling": args.pooling,
                        "corpus_size": target_size,
                        "query_mode": mode,
                        "query_input": label,
                        "corpus_file": str(corpus_path),
                        "retrieval_file": str(out_path),
                        **metrics,
                    }
            )
            print(
                f"[{dataset_key} {target_size}] {label}: "
                f"any={metrics['any_hit@10']:.4f}, "
                f"all={metrics['all_support_hit@10']:.4f}, "
                f"recall={metrics['supporting_title_recall@10']:.4f}"
            )
        all_summary_rows.extend(add_pairwise_deltas(size_summary_rows, per_example_by_mode, args))
    return all_summary_rows


def write_summary(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_key",
        "encoder",
        "pooling",
        "corpus_size",
        "query_mode",
        "query_input",
        "n",
        "any_hit@10",
        "all_support_hit@10",
        "supporting_title_recall@10",
        "vs_question_all_support_hit@10_delta",
        "vs_question_all_support_hit@10_ci_low",
        "vs_question_all_support_hit@10_ci_high",
        "vs_question_supporting_title_recall@10_delta",
        "vs_question_supporting_title_recall@10_ci_low",
        "vs_question_supporting_title_recall@10_ci_high",
        "vs_rewrite_all_support_hit@10_delta",
        "vs_rewrite_all_support_hit@10_ci_low",
        "vs_rewrite_all_support_hit@10_ci_high",
        "vs_rewrite_supporting_title_recall@10_delta",
        "vs_rewrite_supporting_title_recall@10_ci_low",
        "vs_rewrite_supporting_title_recall@10_ci_high",
        "corpus_file",
        "retrieval_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=["hotpotqa", "2wiki"])
    parser.add_argument("--corpus_sizes", nargs="+", type=int, default=DEFAULT_CORPUS_SIZES)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--artifact_dir", default="results/corpus_scale_stress")
    parser.add_argument("--results_dir", default="results/corpus_scale_stress")
    parser.add_argument("--summary_csv", default="results/corpus_scale_stress/corpus_scale_retrieval_summary.csv")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--max_hyde_chars", type=int, default=900)
    parser.add_argument("--max_query_chars", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Model: {args.model}")
    print(f"Pooling: {args.pooling}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    rows = []
    for dataset_key in args.datasets:
        rows.extend(run_dataset(dataset_key, DATASETS[dataset_key], args, tokenizer, model, device))

    write_summary(rows, args.summary_csv)
    print(f"Saved summary to {args.summary_csv}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
