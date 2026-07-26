import argparse
import re
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def load_retrieval_runtime():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "FAISS and sentence-transformers are required to run dense retrieval. "
            "Install the reproducibility environment before running this script."
        ) from exc
    return faiss, SentenceTransformer


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "before", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "of", "on", "or",
    "that", "the", "their", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "with",
}


def normalize_query(text):
    return " ".join(text.strip().split())


def extract_entity_terms(question):
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", question)
    entity_terms = []
    for token in tokens:
        stripped = token.strip("'.-")
        if not stripped:
            continue
        if stripped[:1].isupper() or any(char.isdigit() for char in stripped):
            if stripped.lower() not in STOPWORDS:
                entity_terms.append(stripped)
    return entity_terms


def extract_keyword_terms(question):
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", question)
    keywords = []
    for token in tokens:
        stripped = token.strip("'.-")
        if len(stripped) <= 2:
            continue
        if stripped.lower() in STOPWORDS:
            continue
        keywords.append(stripped)
    return keywords


def add_variant(variants, text):
    text = normalize_query(text)
    if text and text not in variants:
        variants.append(text)


def build_query_variants(question, max_queries=3):
    variants = []
    add_variant(variants, question)

    entity_terms = extract_entity_terms(question)
    if entity_terms:
        add_variant(variants, " ".join(entity_terms))

    keyword_terms = extract_keyword_terms(question)
    if keyword_terms:
        add_variant(variants, " ".join(keyword_terms))

    if len(variants) < max_queries:
        punctuation_removed = re.sub(r"[^A-Za-z0-9'.-]+", " ", question)
        add_variant(variants, punctuation_removed)

    return variants[:max_queries]


def fuse_multi_query_results(score_rows, index_rows, docs, top_k=10, rrf_k=60, query_decay=0.95):
    candidates = {}
    for query_idx, (scores, indices) in enumerate(zip(score_rows, index_rows)):
        query_weight = query_decay ** query_idx
        for rank, (score, doc_idx) in enumerate(zip(scores, indices), start=1):
            doc = docs[int(doc_idx)]
            doc_id = doc["doc_id"]
            rrf_score = query_weight / (rrf_k + rank)
            adjusted_score = float(score) * query_weight

            if doc_id not in candidates:
                candidates[doc_id] = {
                    "score": adjusted_score,
                    "rrf_score": 0.0,
                    "raw_score": float(score),
                    "matched_query_indices": [],
                    "best_rank": rank,
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "text": doc["text"],
                    "source_question_id": doc["source_question_id"],
                }

            item = candidates[doc_id]
            item["rrf_score"] += rrf_score
            if adjusted_score > item["score"]:
                item["score"] = adjusted_score
                item["raw_score"] = float(score)
                item["best_rank"] = rank
            if query_idx not in item["matched_query_indices"]:
                item["matched_query_indices"].append(query_idx)

    fused = sorted(
        candidates.values(),
        key=lambda item: (item["rrf_score"], item["score"]),
        reverse=True,
    )
    return fused[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--index_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--out", default="results/hotpotqa_multiquery_top10_retrieval.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--per_query_k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--query_decay", type=float, default=0.95)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    docs = list(read_jsonl(Path(args.index_dir) / "docstore.jsonl"))
    faiss, SentenceTransformer = load_retrieval_runtime()
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))
    model = SentenceTransformer(args.model)

    query_slices = []
    query_texts = []
    for question in questions:
        variants = build_query_variants(question["question"], max_queries=args.num_queries)
        start = len(query_texts)
        query_texts.extend(variants)
        query_slices.append((start, len(query_texts), variants))

    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embeddings, args.per_query_k)

    rows = []
    for question, (start, end, variants) in zip(questions, query_slices):
        retrieved = fuse_multi_query_results(
            scores[start:end],
            indices[start:end],
            docs,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            query_decay=args.query_decay,
        )
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "answer": question["answer"],
            "supporting_facts": question["supporting_facts"],
            "retrieved": retrieved,
            "query_variants": variants,
            "retrieval_strategy": {
                "name": "multi_query_dense_rrf",
                "num_queries": len(variants),
                "per_query_k": args.per_query_k,
                "top_k": args.top_k,
                "rrf_k": args.rrf_k,
                "query_decay": args.query_decay,
            },
        })

    write_jsonl(args.out, rows)
    print(f"Saved multi-query retrieval results to {args.out}")
    print(f"Questions: {len(rows)}")
    if rows:
        print("First result:")
        print(rows[0]["question"])
        print("Query variants:")
        for idx, variant in enumerate(rows[0]["query_variants"]):
            print(f"  {idx}. {variant}")
        for rank, doc in enumerate(rows[0]["retrieved"], start=1):
            matched = ",".join(str(idx) for idx in doc["matched_query_indices"])
            print(f"{rank}. {doc['title']} rrf={doc['rrf_score']:.4f} queries={matched}")


if __name__ == "__main__":
    main()
