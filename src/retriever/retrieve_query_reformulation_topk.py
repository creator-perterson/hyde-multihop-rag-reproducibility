import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def build_reformulated_query_text(
    row,
    max_query_chars=300,
    query_mode="reformulation_only",
):
    query = row.get("prediction", "").strip()
    if not query:
        query = row["question"].strip()
    if len(query) > max_query_chars:
        query = query[:max_query_chars].rstrip() + "..."
    if query_mode == "reformulation_only":
        return query
    if query_mode == "question_plus_reformulation":
        return f"{row['question'].strip()}\n\nRewritten retrieval query:\n{query}".strip()
    raise ValueError(f"Unsupported query reformulation mode: {query_mode}")


def format_retrieved_docs(score_row, index_row, docs):
    retrieved = []
    for score, doc_idx in zip(score_row, index_row):
        doc = docs[int(doc_idx)]
        retrieved.append({
            "score": float(score),
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": doc["text"],
            "source_question_id": doc["source_question_id"],
        })
    return retrieved


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--query_answers", required=True)
    parser.add_argument("--index_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--out", default="results/hotpotqa_query_reformulation_top10_retrieval.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_query_chars", type=int, default=300)
    parser.add_argument(
        "--query_mode",
        choices=["reformulation_only", "question_plus_reformulation"],
        default="reformulation_only",
    )
    args = parser.parse_args()

    questions = {row["id"]: row for row in read_jsonl(args.questions)}
    query_rows = list(read_jsonl(args.query_answers))
    docs = list(read_jsonl(Path(args.index_dir) / "docstore.jsonl"))
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))

    model = SentenceTransformer(args.model)
    query_texts = [
        build_reformulated_query_text(
            row,
            max_query_chars=args.max_query_chars,
            query_mode=args.query_mode,
        )
        for row in query_rows
    ]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embeddings, args.top_k)

    rows = []
    for query_row, query_text, score_row, index_row in zip(query_rows, query_texts, scores, indices):
        question = questions[query_row["id"]]
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": format_retrieved_docs(score_row, index_row, docs),
            "reformulated_query": query_text,
            "raw_reformulation": query_row.get("prediction", "").strip(),
            "retrieval_strategy": {
                "name": (
                    "question_plus_single_query_reformulation_dense"
                    if args.query_mode == "question_plus_reformulation"
                    else "single_query_reformulation_dense"
                ),
                "query_answers": args.query_answers,
                "top_k": args.top_k,
                "max_query_chars": args.max_query_chars,
                "query_mode": args.query_mode,
            },
        })

    write_jsonl(args.out, rows)
    print(f"Saved single-query reformulation retrieval results to {args.out}")
    print(f"Questions: {len(rows)}")
    if rows:
        print("First result:")
        print(rows[0]["question"])
        print("Dense retrieval query:")
        print(rows[0]["reformulated_query"])
        for rank, doc in enumerate(rows[0]["retrieved"], start=1):
            print(f"{rank}. {doc['title']} score={doc['score']:.4f}")


if __name__ == "__main__":
    main()
