import argparse
from pathlib import Path

from tqdm import tqdm

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


def truncated_hyde_document(row, max_hyde_chars=900):
    hyde_document = row.get("prediction", "").strip()
    if len(hyde_document) > max_hyde_chars:
        hyde_document = hyde_document[:max_hyde_chars].rstrip() + "..."
    return hyde_document


def build_hyde_query_text(row, max_hyde_chars=900, query_mode="question_plus_hypothetical"):
    hyde_document = truncated_hyde_document(row, max_hyde_chars=max_hyde_chars)
    if query_mode == "question_only":
        return row["question"].strip()
    if query_mode == "hypothetical_only":
        return hyde_document
    if query_mode != "question_plus_hypothetical":
        raise ValueError(f"Unsupported HyDE query_mode: {query_mode}")
    return f"{row['question']}\n\nHypothetical supporting passage:\n{hyde_document}".strip()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--hyde_answers", required=True)
    parser.add_argument("--index_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--out", default="results/hotpotqa_hyde_top10_retrieval.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_hyde_chars", type=int, default=900)
    parser.add_argument(
        "--query_mode",
        choices=["question_only", "hypothetical_only", "question_plus_hypothetical"],
        default="question_plus_hypothetical",
    )
    args = parser.parse_args()

    questions = {row["id"]: row for row in read_jsonl(args.questions)}
    hyde_rows = list(read_jsonl(args.hyde_answers))
    docs = list(read_jsonl(Path(args.index_dir) / "docstore.jsonl"))
    faiss, SentenceTransformer = load_retrieval_runtime()
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))

    model = SentenceTransformer(args.model)
    query_texts = [
        build_hyde_query_text(
            row,
            max_hyde_chars=args.max_hyde_chars,
            query_mode=args.query_mode,
        )
        for row in hyde_rows
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
    for hyde_row, score_row, index_row in zip(hyde_rows, scores, indices):
        question = questions[hyde_row["id"]]
        hyde_document = hyde_row.get("prediction", "").strip()
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": format_retrieved_docs(score_row, index_row, docs),
            "hyde_document": hyde_document,
            "retrieval_strategy": {
                "name": "hyde_dense",
                "hyde_answers": args.hyde_answers,
                "top_k": args.top_k,
                "max_hyde_chars": args.max_hyde_chars,
                "query_mode": args.query_mode,
            },
        })

    write_jsonl(args.out, rows)
    print(f"Saved HyDE retrieval results to {args.out}")
    print(f"Questions: {len(rows)}")
    if rows:
        print("First result:")
        print(rows[0]["question"])
        print("HyDE document:")
        print(rows[0]["hyde_document"])
        for rank, doc in enumerate(rows[0]["retrieved"], start=1):
            print(f"{rank}. {doc['title']} score={doc['score']:.4f}")


if __name__ == "__main__":
    main()
