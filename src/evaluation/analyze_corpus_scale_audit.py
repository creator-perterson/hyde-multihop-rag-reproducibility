import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


DATASETS = {
    "hotpotqa": {
        "name": "HotpotQA",
        "questions": "local-artifacts/datasets/ircot_hotpotqa_test500/questions.jsonl",
        "base_corpus": "local-artifacts/datasets/ircot_hotpotqa_test500/corpus.jsonl",
        "hyde_answers": "local-artifacts/results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
        "rewrite_answers": "local-artifacts/results/ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_hotpotqa_test500",
    },
    "2wiki": {
        "name": "2WikiMultihopQA",
        "questions": "local-artifacts/datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "base_corpus": "local-artifacts/datasets/ircot_2wikimultihopqa_test500/corpus.jsonl",
        "hyde_answers": "local-artifacts/results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
        "rewrite_answers": "local-artifacts/results/ircot_2wiki_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_2wiki_test500",
    },
}


QUERY_MODES = [
    ("question_only", "Question only"),
    ("single_rewritten_query", "Single rewritten query"),
    ("question_plus_hypothetical", "Question + hypothetical passage"),
]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}


def normalize_space(text):
    return " ".join(str(text or "").strip().lower().split())


def normalize_title(text):
    return normalize_space(text)


def normalize_text(text):
    return normalize_space(text)


def simple_tokens(text):
    token = []
    tokens = []
    for char in normalize_text(text):
        if char.isalnum():
            token.append(char)
        elif token:
            word = "".join(token)
            if len(word) > 2 and word not in STOPWORDS:
                tokens.append(word)
            token = []
    if token:
        word = "".join(token)
        if len(word) > 2 and word not in STOPWORDS:
            tokens.append(word)
    return tokens


def support_title_set(questions):
    titles = set()
    for question in questions:
        facts = question.get("supporting_facts", {})
        raw_titles = facts.get("title", []) if isinstance(facts, dict) else []
        if not isinstance(raw_titles, list):
            raw_titles = [raw_titles]
        for title in raw_titles:
            normalized = normalize_title(title)
            if normalized:
                titles.add(normalized)
    return titles


def is_added_doc(doc):
    return doc.get("source_split") == "train"


def gold_support_docs(base_docs, gold_titles):
    docs = []
    for doc in base_docs:
        title = normalize_title(doc.get("title", ""))
        if doc.get("is_supporting") is True or title in gold_titles:
            docs.append(doc)
    return docs


def jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_gold_token_index(gold_docs):
    gold_sets = []
    inverted = {}
    exact_texts = set()
    for idx, doc in enumerate(gold_docs):
        exact_texts.add(normalize_text(doc.get("text", "")))
        tokens = set(simple_tokens(doc.get("text", "")))
        gold_sets.append(tokens)
        for token in tokens:
            inverted.setdefault(token, set()).add(idx)
    return gold_sets, inverted, exact_texts


def max_gold_jaccard(doc, gold_sets, inverted):
    tokens = set(simple_tokens(doc.get("text", "")))
    if not tokens:
        return 0.0
    candidates = set()
    for token in tokens:
        candidates.update(inverted.get(token, ()))
    if not candidates:
        return 0.0
    return max(jaccard(tokens, gold_sets[idx]) for idx in candidates)


def audit_corpus_overlap(
    questions,
    base_docs,
    expanded_docs,
    near_duplicate_threshold=0.85,
):
    gold_titles = support_title_set(questions)
    added_docs = [doc for doc in expanded_docs if is_added_doc(doc)]
    gold_docs = gold_support_docs(base_docs, gold_titles)
    gold_sets, inverted, exact_gold_texts = build_gold_token_index(gold_docs)

    added_gold_title_docs = [
        doc for doc in added_docs if normalize_title(doc.get("title", "")) in gold_titles
    ]
    exact_duplicates = 0
    near_duplicates = 0
    for doc in added_docs:
        if normalize_text(doc.get("text", "")) in exact_gold_texts:
            exact_duplicates += 1
        if max_gold_jaccard(doc, gold_sets, inverted) >= near_duplicate_threshold:
            near_duplicates += 1

    return {
        "total_docs": len(expanded_docs),
        "base_docs": len(base_docs),
        "added_docs": len(added_docs),
        "unique_added_titles": len({normalize_title(doc.get("title", "")) for doc in added_docs}),
        "unique_added_texts": len({normalize_text(doc.get("text", "")) for doc in added_docs}),
        "gold_support_titles": len(gold_titles),
        "gold_support_paragraphs": len(gold_docs),
        "added_gold_title_docs": len(added_gold_title_docs),
        "added_gold_title_unique_titles": len(
            {normalize_title(doc.get("title", "")) for doc in added_gold_title_docs}
        ),
        "exact_gold_paragraph_duplicate_docs": exact_duplicates,
        "near_duplicate_gold_paragraph_docs": near_duplicates,
        "near_duplicate_threshold": near_duplicate_threshold,
    }


def filter_added_gold_title_docs(docs, gold_titles):
    return [
        doc
        for doc in docs
        if not (is_added_doc(doc) and normalize_title(doc.get("title", "")) in gold_titles)
    ]


def select_hard_negative_indices(candidate_indices, max_scores, count):
    ranked = sorted(candidate_indices, key=lambda idx: (-max_scores[idx], idx))
    return ranked[:count]


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def similarity_summary(scores):
    values = sorted(float(score) for score in scores)
    return {
        "added_max_question_similarity_mean": statistics.mean(values) if values else 0.0,
        "added_max_question_similarity_p50": quantile(values, 0.50),
        "added_max_question_similarity_p90": quantile(values, 0.90),
        "added_max_question_similarity_p95": quantile(values, 0.95),
        "added_max_question_similarity_p99": quantile(values, 0.99),
        "added_max_question_similarity_max": values[-1] if values else 0.0,
    }


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
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, batch_size, max_length, device, pooling):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

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


def safe_model_name(model_name):
    return model_name.replace("/", "__")


def load_doc_embeddings(args, dataset_key, docs, tokenizer, model, device):
    import torch

    cache_path = (
        Path(args.artifact_dir)
        / "embedding_cache"
        / f"{dataset_key}_{safe_model_name(args.model)}_{args.source_corpus_size}_{args.pooling}_docmax{args.doc_max_length}.pt"
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu").contiguous()
    texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
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


def load_query_embeddings(args, dataset_key, mode, query_texts, tokenizer, model, device):
    import torch

    cache_path = (
        Path(args.artifact_dir)
        / "embedding_cache"
        / f"{dataset_key}_{safe_model_name(args.model)}_{mode}_{args.pooling}_querymax{args.query_max_length}.pt"
    )
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu").contiguous()
    embeddings = encode_texts(
        query_texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.query_batch_size,
        max_length=args.query_max_length,
        device=device,
        pooling=args.pooling,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def max_question_similarity_by_doc(doc_embeddings, question_embeddings, candidate_indices, chunk_size):
    max_scores = {}
    for start in range(0, len(candidate_indices), chunk_size):
        chunk_indices = candidate_indices[start : start + chunk_size]
        chunk = doc_embeddings[chunk_indices]
        scores = chunk @ question_embeddings.T
        values = scores.max(dim=1).values.tolist()
        for idx, score in zip(chunk_indices, values):
            max_scores[idx] = float(score)
    return max_scores


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
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int((alpha / 2) * iterations)
    hi_idx = int((1 - alpha / 2) * iterations) - 1
    return means[lo_idx], means[max(lo_idx, min(hi_idx, iterations - 1))]


def add_pairwise_deltas(summary_rows, per_example_by_mode, args):
    target_mode = "question_plus_hypothetical"
    if target_mode not in per_example_by_mode:
        return summary_rows
    target = per_example_by_mode[target_mode]
    comparisons = [("vs_question", "question_only"), ("vs_rewrite", "single_rewritten_query")]
    for row in summary_rows:
        if row["query_mode"] != target_mode:
            continue
        for prefix, baseline_mode in comparisons:
            baseline = per_example_by_mode[baseline_mode]
            for offset, metric_name in enumerate(["all_support_hit@10", "supporting_title_recall@10"]):
                ids = sorted(set(baseline) & set(target))
                deltas = [target[item_id][metric_name] - baseline[item_id][metric_name] for item_id in ids]
                mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
                ci_low, ci_high = bootstrap_delta_ci(
                    deltas,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + offset + (100 if prefix == "vs_rewrite" else 0),
                )
                row[f"{prefix}_{metric_name}_delta"] = mean_delta
                row[f"{prefix}_{metric_name}_ci_low"] = ci_low
                row[f"{prefix}_{metric_name}_ci_high"] = ci_high
    return summary_rows


def retrieve_scenario(
    args,
    dataset_key,
    cfg,
    scenario_name,
    docs,
    doc_embeddings,
    questions,
    hyde_by_id,
    rewrite_by_id,
    query_embeddings_by_mode,
):
    import torch

    results_dir = Path(args.out_dir) / "retrieval"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_example_by_mode = {}
    for mode, label in QUERY_MODES:
        query_embeddings = query_embeddings_by_mode[mode]
        scores = query_embeddings @ doc_embeddings.T
        top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=1)
        rows = []
        for question, score_row, index_row in zip(questions, top_scores, top_indices):
            rows.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "answer": question["answer"],
                    "supporting_facts": question.get("supporting_facts", {}),
                    "retrieved": format_retrieved_docs(score_row, index_row, docs),
                    "hyde_document": hyde_by_id[question["id"]].get("prediction", "").strip(),
                    "reformulated_query": rewrite_by_id[question["id"]].get("prediction", "").strip(),
                    "retrieval_strategy": {
                        "name": "corpus_scale_audit_dense",
                        "dataset": cfg["name"],
                        "scenario": scenario_name,
                        "encoder": args.model,
                        "pooling": args.pooling,
                        "query_mode": mode,
                        "top_k": args.top_k,
                    },
                }
            )
        out_path = (
            results_dir
            / f"{cfg['out_prefix']}_{scenario_name}_{mode}_top{args.top_k}_retrieval.jsonl"
        )
        write_jsonl(out_path, rows)
        metrics = retrieval_metrics(rows)
        per_example_by_mode[mode] = per_example_retrieval_metrics(rows)
        summary_rows.append(
            {
                "dataset": cfg["name"],
                "dataset_key": dataset_key,
                "scenario": scenario_name,
                "docs": len(docs),
                "query_mode": mode,
                "query_input": label,
                "retrieval_file": str(out_path),
                **metrics,
            }
        )
    return add_pairwise_deltas(summary_rows, per_example_by_mode, args)


def read_existing_random_rows(args, dataset_key, cfg):
    summary_path = Path(args.artifact_dir) / "corpus_scale_retrieval_summary.csv"
    if not summary_path.exists():
        return []
    rows = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset_key") == dataset_key and int(row.get("corpus_size", 0)) == args.source_corpus_size:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "dataset_key": dataset_key,
                        "scenario": f"random_{args.source_corpus_size}",
                        "docs": int(row["corpus_size"]),
                        "query_mode": row["query_mode"],
                        "query_input": row["query_input"],
                        "n": int(row["n"]),
                        "any_hit@10": float(row["any_hit@10"]),
                        "all_support_hit@10": float(row["all_support_hit@10"]),
                        "supporting_title_recall@10": float(row["supporting_title_recall@10"]),
                        "vs_question_all_support_hit@10_delta": row.get("vs_question_all_support_hit@10_delta", ""),
                        "vs_question_all_support_hit@10_ci_low": row.get("vs_question_all_support_hit@10_ci_low", ""),
                        "vs_question_all_support_hit@10_ci_high": row.get("vs_question_all_support_hit@10_ci_high", ""),
                    }
                )
    return rows


def build_query_embedding_cache(args, dataset_key, questions, hyde_by_id, rewrite_by_id, tokenizer, model, device):
    query_embeddings_by_mode = {}
    for mode, _ in QUERY_MODES:
        query_texts = [
            build_query_text(
                question,
                hyde_by_id[question["id"]],
                rewrite_by_id[question["id"]],
                mode,
                max_hyde_chars=args.max_hyde_chars,
                max_query_chars=args.max_query_chars,
            )
            for question in questions
        ]
        query_embeddings_by_mode[mode] = load_query_embeddings(
            args, dataset_key, mode, query_texts, tokenizer, model, device
        )
    return query_embeddings_by_mode


def fmt(value, digits=4):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_lookup(rows, dataset_key, scenario, mode):
    for row in rows:
        if row["dataset_key"] == dataset_key and row["scenario"] == scenario and row["query_mode"] == mode:
            return row
    return None


def write_markdown(path, audit_rows, retrieval_rows):
    lines = [
        "| Dataset | Stress corpus | Docs | Added gold-title docs | Near-dup gold paragraphs | Added sim p95 | Q All@10 | HyDE All@10 | Delta All |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in audit_rows:
        dataset_key = audit["dataset_key"]
        scenario = audit["scenario"]
        q = row_lookup(retrieval_rows, dataset_key, scenario, "question_only")
        h = row_lookup(retrieval_rows, dataset_key, scenario, "question_plus_hypothetical")
        if not q or not h:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    audit["dataset"],
                    audit["label"],
                    str(audit["total_docs"]),
                    str(audit["added_gold_title_docs"]),
                    str(audit["near_duplicate_gold_paragraph_docs"]),
                    fmt(audit["added_max_question_similarity_p95"]),
                    fmt(q["all_support_hit@10"]),
                    fmt(h["all_support_hit@10"]),
                    fmt(h.get("vs_question_all_support_hit@10_delta", "")),
                ]
            )
            + " |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, audit_rows, retrieval_rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Corpus-scale leakage and BGE hard-negative audit.}",
        r"\label{tab:corpus_scale_audit}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Stress corpus & Docs & Gold-title add. & Near-dup gold & Sim p95 & Q All & HyDE All & $\Delta$All \\",
        r"\midrule",
    ]
    for audit in audit_rows:
        dataset_key = audit["dataset_key"]
        scenario = audit["scenario"]
        q = row_lookup(retrieval_rows, dataset_key, scenario, "question_only")
        h = row_lookup(retrieval_rows, dataset_key, scenario, "question_plus_hypothetical")
        if not q or not h:
            continue
        lines.append(
            " & ".join(
                [
                    audit["dataset"],
                    audit["label"],
                    str(audit["total_docs"]),
                    str(audit["added_gold_title_docs"]),
                    str(audit["near_duplicate_gold_paragraph_docs"]),
                    fmt(audit["added_max_question_similarity_p95"]),
                    fmt(q["all_support_hit@10"]),
                    fmt(h["all_support_hit@10"]),
                    fmt(h.get("vs_question_all_support_hit@10_delta", "")),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item The random 100k rows audit the existing processed-train distractor expansion. The no-gold-title rows remove only added train records whose normalized title matches any test supporting title, while preserving the released evaluation corpus. The BGE-hard rows keep the released evaluation corpus and select the most question-similar added train distractors from the audited 100k pool after the same gold-title exclusion; they are a harder 50k diagnostic, not a larger index. Sim p95 is the 95th percentile of each added distractor's maximum BGE cosine similarity to any test question. Near-dup gold counts added records whose token-Jaccard similarity to any test gold supporting paragraph is at least 0.85.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def process_dataset(args, dataset_key, cfg, tokenizer, model, device):
    import torch

    paper_root = Path(args.paper_root)
    questions = list(read_jsonl(paper_root / cfg["questions"]))
    base_docs = list(read_jsonl(paper_root / cfg["base_corpus"]))
    source_corpus_path = (
        Path(args.artifact_dir)
        / f"{cfg['out_prefix']}_corpus_scale_{args.source_corpus_size}.jsonl"
    )
    source_docs = list(read_jsonl(source_corpus_path))
    metadata_path = source_corpus_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    gold_titles = support_title_set(questions)

    hyde_by_id = {row["id"]: row for row in read_jsonl(paper_root / cfg["hyde_answers"])}
    rewrite_by_id = {row["id"]: row for row in read_jsonl(paper_root / cfg["rewrite_answers"])}
    query_embeddings_by_mode = build_query_embedding_cache(
        args, dataset_key, questions, hyde_by_id, rewrite_by_id, tokenizer, model, device
    )
    doc_embeddings = load_doc_embeddings(args, dataset_key, source_docs, tokenizer, model, device)

    base_indices = [idx for idx, doc in enumerate(source_docs) if not is_added_doc(doc)]
    added_indices = [idx for idx, doc in enumerate(source_docs) if is_added_doc(doc)]
    non_gold_added_indices = [
        idx
        for idx in added_indices
        if normalize_title(source_docs[idx].get("title", "")) not in gold_titles
    ]

    question_embeddings = query_embeddings_by_mode["question_only"]
    max_scores = max_question_similarity_by_doc(
        doc_embeddings,
        question_embeddings,
        non_gold_added_indices,
        chunk_size=args.similarity_chunk_size,
    )

    random_audit = audit_corpus_overlap(
        questions,
        base_docs,
        source_docs,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    random_audit.update(
        {
            "dataset": cfg["name"],
            "dataset_key": dataset_key,
            "scenario": f"random_{args.source_corpus_size}",
            "label": "Random 100k",
            "skipped_duplicates": metadata.get("skipped_duplicates", ""),
            **similarity_summary(max_scores.values()),
        }
    )

    no_gold_docs = filter_added_gold_title_docs(source_docs, gold_titles)
    no_gold_indices = [
        idx
        for idx, doc in enumerate(source_docs)
        if not (is_added_doc(doc) and normalize_title(doc.get("title", "")) in gold_titles)
    ]
    no_gold_embeddings = doc_embeddings[no_gold_indices].contiguous()
    no_gold_path = (
        Path(args.out_dir)
        / f"{cfg['out_prefix']}_corpus_scale_{args.source_corpus_size}_no_test_gold_titles.jsonl"
    )
    write_jsonl(no_gold_path, no_gold_docs)
    no_gold_audit = audit_corpus_overlap(
        questions,
        base_docs,
        no_gold_docs,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    no_gold_audit.update(
        {
            "dataset": cfg["name"],
            "dataset_key": dataset_key,
            "scenario": f"no_gold_title_{args.source_corpus_size}",
            "label": "No-gold-title 100k",
            "skipped_duplicates": metadata.get("skipped_duplicates", ""),
            **similarity_summary(max_scores.values()),
        }
    )

    hard_added_count = max(0, args.hard_target_size - len(base_indices))
    hard_added_indices = select_hard_negative_indices(
        non_gold_added_indices,
        max_scores,
        min(hard_added_count, len(non_gold_added_indices)),
    )
    hard_indices = base_indices + hard_added_indices
    hard_docs = [source_docs[idx] for idx in hard_indices]
    hard_embeddings = doc_embeddings[hard_indices].contiguous()
    hard_path = (
        Path(args.out_dir)
        / f"{cfg['out_prefix']}_corpus_scale_bge_hard_{len(hard_docs)}.jsonl"
    )
    write_jsonl(hard_path, hard_docs)
    hard_score_values = [max_scores[idx] for idx in hard_added_indices]
    hard_audit = audit_corpus_overlap(
        questions,
        base_docs,
        hard_docs,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    hard_audit.update(
        {
            "dataset": cfg["name"],
            "dataset_key": dataset_key,
            "scenario": f"bge_hard_{len(hard_docs)}",
            "label": "BGE-hard 50k",
            "skipped_duplicates": metadata.get("skipped_duplicates", ""),
            **similarity_summary(hard_score_values),
        }
    )

    retrieval_rows = read_existing_random_rows(args, dataset_key, cfg)
    retrieval_rows.extend(
        retrieve_scenario(
            args,
            dataset_key,
            cfg,
            no_gold_audit["scenario"],
            no_gold_docs,
            no_gold_embeddings,
            questions,
            hyde_by_id,
            rewrite_by_id,
            query_embeddings_by_mode,
        )
    )
    retrieval_rows.extend(
        retrieve_scenario(
            args,
            dataset_key,
            cfg,
            hard_audit["scenario"],
            hard_docs,
            hard_embeddings,
            questions,
            hyde_by_id,
            rewrite_by_id,
            query_embeddings_by_mode,
        )
    )
    return [random_audit, no_gold_audit, hard_audit], retrieval_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=list(DATASETS))
    parser.add_argument("--artifact_dir", default="local-artifacts/corpus_scale_stress")
    parser.add_argument("--out_dir", default="local-artifacts/corpus_scale_audit")
    parser.add_argument("--source_corpus_size", type=int, default=100000)
    parser.add_argument("--hard_target_size", type=int, default=50000)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--near_duplicate_threshold", type=float, default=0.85)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--doc_max_length", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=512)
    parser.add_argument("--max_hyde_chars", type=int, default=900)
    parser.add_argument("--max_query_chars", type=int, default=300)
    parser.add_argument("--similarity_chunk_size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    paper_root = Path(args.paper_root)
    args.artifact_dir = str(paper_root / args.artifact_dir)
    args.out_dir = str(paper_root / args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    audit_rows = []
    retrieval_rows = []
    for dataset_key in args.datasets:
        dataset_audit_rows, dataset_retrieval_rows = process_dataset(
            args, dataset_key, DATASETS[dataset_key], tokenizer, model, device
        )
        audit_rows.extend(dataset_audit_rows)
        retrieval_rows.extend(dataset_retrieval_rows)

    audit_fields = [
        "dataset",
        "dataset_key",
        "scenario",
        "label",
        "total_docs",
        "base_docs",
        "added_docs",
        "unique_added_titles",
        "unique_added_texts",
        "gold_support_titles",
        "gold_support_paragraphs",
        "added_gold_title_docs",
        "added_gold_title_unique_titles",
        "exact_gold_paragraph_duplicate_docs",
        "near_duplicate_gold_paragraph_docs",
        "near_duplicate_threshold",
        "skipped_duplicates",
        "added_max_question_similarity_mean",
        "added_max_question_similarity_p50",
        "added_max_question_similarity_p90",
        "added_max_question_similarity_p95",
        "added_max_question_similarity_p99",
        "added_max_question_similarity_max",
    ]
    retrieval_fields = [
        "dataset",
        "dataset_key",
        "scenario",
        "docs",
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
        "retrieval_file",
    ]
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "corpus_scale_overlap_audit.csv", audit_rows, audit_fields)
    write_csv(out_dir / "corpus_scale_audit_retrieval_summary.csv", retrieval_rows, retrieval_fields)
    write_markdown(out_dir / "corpus_scale_audit_summary.md", audit_rows, retrieval_rows)
    write_latex(
        Path(args.paper_root) / "paper/latex/table_corpus_scale_audit.tex",
        audit_rows,
        retrieval_rows,
    )
    print(json.dumps({"audit": audit_rows, "retrieval": retrieval_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
