import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def add_rank_scores(accumulator, docs, source_name, weight, rrf_k):
    for rank, doc in enumerate(docs, start=1):
        doc_id = doc["doc_id"]
        if doc_id not in accumulator:
            accumulator[doc_id] = {
                "doc": dict(doc),
                "score": 0.0,
                "dense_rank": None,
                "lexical_rank": None,
            }
        accumulator[doc_id]["score"] += weight / (rrf_k + rank)
        accumulator[doc_id][f"{source_name}_rank"] = rank


def fuse_retrieved_docs(dense_docs, lexical_docs, top_k=10, dense_weight=1.0, lexical_weight=1.0, rrf_k=60):
    accumulator = {}
    add_rank_scores(accumulator, dense_docs, "dense", dense_weight, rrf_k)
    add_rank_scores(accumulator, lexical_docs, "lexical", lexical_weight, rrf_k)

    fused = []
    for item in accumulator.values():
        doc = dict(item["doc"])
        doc["score"] = float(item["score"])
        doc["dense_rank"] = item["dense_rank"]
        doc["lexical_rank"] = item["lexical_rank"]
        fused.append(doc)

    return sorted(fused, key=lambda doc: doc["score"], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--lexical", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dense_weight", type=float, default=1.0)
    parser.add_argument("--lexical_weight", type=float, default=1.0)
    parser.add_argument("--rrf_k", type=int, default=60)
    args = parser.parse_args()

    dense_rows = list(read_jsonl(args.dense))
    lexical_rows = list(read_jsonl(args.lexical))
    lexical_by_id = {row["id"]: row for row in lexical_rows}

    rows = []
    for dense_row in dense_rows:
        question_id = dense_row["id"]
        lexical_row = lexical_by_id[question_id]
        retrieved = fuse_retrieved_docs(
            dense_docs=dense_row["retrieved"],
            lexical_docs=lexical_row["retrieved"],
            top_k=args.top_k,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            rrf_k=args.rrf_k,
        )
        rows.append({
            "id": dense_row["id"],
            "question": dense_row["question"],
            "answer": dense_row["answer"],
            "supporting_facts": dense_row["supporting_facts"],
            "retrieved": retrieved,
        })

    write_jsonl(args.out, rows)
    print(f"Saved hybrid retrieval results to {args.out}")
    print(f"Questions: {len(rows)}")
    if rows:
        print("First result:")
        print(rows[0]["question"])
        for rank, doc in enumerate(rows[0]["retrieved"], start=1):
            print(
                f"{rank}. {doc['title']} score={doc['score']:.4f} "
                f"dense_rank={doc['dense_rank']} lexical_rank={doc['lexical_rank']}"
            )


if __name__ == "__main__":
    main()
