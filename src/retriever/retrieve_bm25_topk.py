import argparse
import math
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def build_bm25_index(docs):
    tokenized_docs = []
    document_frequency = Counter()

    for doc in docs:
        text = f"{doc['title']} {doc['title']} {doc['text']}"
        tokens = tokenize(text)
        tokenized_docs.append(tokens)
        document_frequency.update(set(tokens))

    average_doc_len = (
        sum(len(tokens) for tokens in tokenized_docs) / len(tokenized_docs)
        if tokenized_docs
        else 0.0
    )
    return tokenized_docs, document_frequency, average_doc_len


def bm25_score(query_tokens, doc_tokens, document_frequency, total_docs, average_doc_len, k1=1.5, b=0.75):
    if not doc_tokens or average_doc_len == 0:
        return 0.0

    term_frequency = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0

    for term in query_tokens:
        if term not in term_frequency:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        tf = term_frequency[term]
        denominator = tf + k1 * (1 - b + b * doc_len / average_doc_len)
        score += idf * (tf * (k1 + 1)) / denominator

    return score


def rank_documents_bm25(query, docs, top_k=10):
    tokenized_docs, document_frequency, average_doc_len = build_bm25_index(docs)
    query_tokens = tokenize(query)
    total_docs = len(docs)

    scored = []
    for doc, doc_tokens in zip(docs, tokenized_docs):
        score = bm25_score(
            query_tokens=query_tokens,
            doc_tokens=doc_tokens,
            document_frequency=document_frequency,
            total_docs=total_docs,
            average_doc_len=average_doc_len,
        )
        ranked_doc = dict(doc)
        ranked_doc["score"] = float(score)
        scored.append(ranked_doc)

    return sorted(scored, key=lambda doc: doc["score"], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--corpus", default="datasets/hotpotqa_sample/corpus.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_top10_bm25_retrieval.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    docs = list(read_jsonl(args.corpus))

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(docs)} corpus documents")
    print("Building BM25 scores. This requires no model download.")

    tokenized_docs, document_frequency, average_doc_len = build_bm25_index(docs)
    rows = []
    for question in tqdm(questions):
        query_tokens = tokenize(question["question"])
        scored = []
        for doc, doc_tokens in zip(docs, tokenized_docs):
            score = bm25_score(
                query_tokens=query_tokens,
                doc_tokens=doc_tokens,
                document_frequency=document_frequency,
                total_docs=len(docs),
                average_doc_len=average_doc_len,
            )
            scored.append((score, doc))

        retrieved = []
        for score, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:args.top_k]:
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
