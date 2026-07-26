import argparse
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--index_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--out", default="results/hotpotqa_top5_retrieval.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    docs = list(read_jsonl(Path(args.index_dir) / "docstore.jsonl"))
    index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))

    model = SentenceTransformer(args.model)
    query_texts = [q["question"] for q in questions]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embeddings, args.top_k)

    rows = []
    for question, score_row, index_row in zip(questions, scores, indices):
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
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "answer": question["answer"],
            "supporting_facts": question["supporting_facts"],
            "retrieved": retrieved,
        })

    write_jsonl(args.out, rows)
    print(f"Saved retrieval results to {args.out}")
    print("First result:")
    print(rows[0]["question"])
    for rank, doc in enumerate(rows[0]["retrieved"], start=1):
        print(f"{rank}. {doc['title']} score={doc['score']:.4f}")


if __name__ == "__main__":
    main()
