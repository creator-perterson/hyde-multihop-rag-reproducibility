import argparse
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def build_expanded_queries(question, first_round_docs, expand_docs):
    queries = [question]
    for doc in first_round_docs[:expand_docs]:
        text = doc["text"].strip()
        if len(text) > 700:
            text = text[:700].rstrip()
        queries.append(f"{question}\nRelated evidence title: {doc['title']}\nRelated evidence text: {text}")
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--index_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--first_retrieval", default="results/hotpotqa_top10_retrieval.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_iterative_top10_retrieval.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--per_query_k", type=int, default=10)
    parser.add_argument("--expand_docs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    questions = {row["id"]: row for row in read_jsonl(args.questions)}
    first_rows = list(read_jsonl(args.first_retrieval))
    docs = list(read_jsonl(Path(args.index_dir) / "docstore.jsonl"))
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))
    model = SentenceTransformer(args.model)

    outputs = []
    for row in tqdm(first_rows):
        question = questions[row["id"]]
        expanded_queries = build_expanded_queries(
            row["question"],
            row["retrieved"],
            args.expand_docs,
        )
        query_embeddings = model.encode(
            expanded_queries,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = index.search(query_embeddings, args.per_query_k)

        candidates = {}
        for query_idx, (score_row, index_row) in enumerate(zip(scores, indices)):
            # Slightly prefer original-query matches, but still allow second-hop expansions to surface new evidence.
            query_weight = 1.0 if query_idx == 0 else 0.97
            for score, doc_idx in zip(score_row, index_row):
                doc = docs[int(doc_idx)]
                adjusted_score = float(score) * query_weight
                old = candidates.get(doc["doc_id"])
                if old is None or adjusted_score > old["score"]:
                    candidates[doc["doc_id"]] = {
                        "score": adjusted_score,
                        "raw_score": float(score),
                        "matched_query_index": query_idx,
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "text": doc["text"],
                        "source_question_id": doc["source_question_id"],
                    }

        retrieved = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)[:args.top_k]
        outputs.append({
            "id": row["id"],
            "question": row["question"],
            "answer": question["answer"],
            "supporting_facts": question["supporting_facts"],
            "retrieved": retrieved,
            "retrieval_strategy": {
                "name": "iterative_query_expansion",
                "first_retrieval": args.first_retrieval,
                "expand_docs": args.expand_docs,
                "per_query_k": args.per_query_k,
                "top_k": args.top_k,
            },
        })

    write_jsonl(args.out, outputs)
    print(f"Saved iterative retrieval results to {args.out}")
    print("First result:")
    print(outputs[0]["question"])
    for rank, doc in enumerate(outputs[0]["retrieved"], start=1):
        query_label = "original" if doc["matched_query_index"] == 0 else f"expanded-{doc['matched_query_index']}"
        print(f"{rank}. {doc['title']} score={doc['score']:.4f} via {query_label}")


if __name__ == "__main__":
    main()
