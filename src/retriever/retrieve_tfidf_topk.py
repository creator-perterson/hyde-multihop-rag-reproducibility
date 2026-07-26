import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--corpus", default="datasets/hotpotqa_sample/corpus.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_top5_tfidf_retrieval.jsonl")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    docs = list(read_jsonl(args.corpus))
    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    query_texts = [q["question"] for q in questions]

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(docs)} corpus documents")
    print("Building TF-IDF vectors. This requires no model download.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100000,
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)

    rows = []
    for question, query_vec in tqdm(zip(questions, query_matrix), total=len(questions)):
        sims = cosine_similarity(query_vec, doc_matrix).ravel()
        top_indices = sims.argsort()[::-1][:args.top_k]

        retrieved = []
        for doc_idx in top_indices:
            doc = docs[int(doc_idx)]
            retrieved.append({
                "score": float(sims[doc_idx]),
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
