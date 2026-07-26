import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


DATASETS = {
    "hotpotqa": {
        "questions": "local-artifacts/datasets/ircot_hotpotqa_test500/questions.jsonl",
        "corpus": "local-artifacts/datasets/ircot_hotpotqa_test500/corpus.jsonl",
        "hyde_answers": "local-artifacts/results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
        "direct_answers": "local-artifacts/results/ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_hotpotqa_test500",
    },
    "2wiki": {
        "questions": "local-artifacts/datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "corpus": "local-artifacts/datasets/ircot_2wikimultihopqa_test500/corpus.jsonl",
        "hyde_answers": "local-artifacts/results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
        "direct_answers": "local-artifacts/results/ircot_2wiki_test500_single_query_reformulation_qwenmax_500.jsonl",
        "out_prefix": "ircot_2wiki_test500",
    },
}


MODE_TO_SUFFIX = {
    "question_only": "dense",
    "direct_rewrite": "direct_rewrite",
    "hyde": "hyde",
}


def truncate_text(text, max_chars):
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def build_query_texts(
    questions,
    answers_by_id,
    query_mode,
    max_hyde_chars=900,
    max_query_chars=300,
):
    query_texts = []
    for question in questions:
        question_text = question["question"].strip()
        if query_mode == "question_only":
            query_texts.append(question_text)
            continue

        answer_row = answers_by_id[question["id"]]
        prediction = answer_row.get("prediction", "")
        if query_mode == "direct_rewrite":
            rewrite = truncate_text(prediction, max_query_chars)
            query_texts.append(rewrite if rewrite else question_text)
        elif query_mode == "hyde":
            hyde_document = truncate_text(prediction, max_hyde_chars)
            query_texts.append(
                f"{question_text}\n\nHypothetical supporting passage:\n{hyde_document}".strip()
            )
        else:
            raise ValueError(f"Unsupported query mode: {query_mode}")
    return query_texts


def to_plain_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def format_retrieved_docs(scores, indices, docs):
    retrieved = []
    for score, doc_idx in zip(to_plain_list(scores), to_plain_list(indices)):
        doc = docs[int(doc_idx)]
        retrieved.append(
            {
                "score": float(score),
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": doc["text"],
                "source_question_id": doc.get("source_question_id", ""),
            }
        )
    return retrieved


def load_answers_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def encode_texts(texts, model, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def retrieve_with_dense_matrix(query_embeddings, doc_embeddings, top_k):
    import torch

    query_tensor = torch.as_tensor(query_embeddings, dtype=torch.float32)
    doc_tensor = torch.as_tensor(doc_embeddings, dtype=torch.float32)
    scores = query_tensor @ doc_tensor.T
    return torch.topk(scores, k=top_k, dim=1)


def rows_for_mode(
    questions,
    docs,
    query_texts,
    top_scores,
    top_indices,
    query_mode,
    answers_by_id,
    args,
):
    rows = []
    for question, query_text, scores, indices in zip(
        questions, query_texts, top_scores, top_indices
    ):
        answer_row = answers_by_id.get(question["id"], {})
        row = {
            "id": question["id"],
            "question": question["question"],
            "answer": question.get("answer", ""),
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": format_retrieved_docs(scores, indices, docs),
            "retrieval_query_text": query_text,
            "retrieval_strategy": {
                "name": f"topk_sensitivity_{query_mode}_dense",
                "encoder": args.model,
                "top_k": args.top_k,
                "batch_size": args.batch_size,
                "query_mode": query_mode,
                "search": "dense_matrix_topk_without_faiss",
            },
        }
        if query_mode == "direct_rewrite":
            row["raw_reformulation"] = answer_row.get("prediction", "").strip()
            row["reformulated_query"] = query_text
        elif query_mode == "hyde":
            row["hyde_document"] = answer_row.get("prediction", "").strip()
        rows.append(row)
    return rows


def run_dataset(dataset_key, cfg, args):
    from sentence_transformers import SentenceTransformer

    paper_root = Path(args.paper_root)
    out_dir = paper_root / args.out_dir
    questions = list(read_jsonl(paper_root / cfg["questions"]))
    docs = list(read_jsonl(paper_root / cfg["corpus"]))
    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]

    print(f"[{dataset_key}] Loading encoder: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"[{dataset_key}] Encoding {len(docs)} corpus documents")
    doc_embeddings = encode_texts(doc_texts, model, args.batch_size)

    for query_mode in args.query_modes:
        answers_by_id = {}
        if query_mode == "hyde":
            answers_by_id = load_answers_by_id(paper_root / cfg["hyde_answers"])
        elif query_mode == "direct_rewrite":
            answers_by_id = load_answers_by_id(paper_root / cfg["direct_answers"])

        missing = [
            question["id"]
            for question in questions
            if query_mode != "question_only" and question["id"] not in answers_by_id
        ]
        if missing:
            raise ValueError(
                f"{dataset_key}/{query_mode} is missing {len(missing)} generated rows"
            )

        print(f"[{dataset_key}] Retrieving {query_mode} top-{args.top_k}")
        query_texts = build_query_texts(
            questions,
            answers_by_id,
            query_mode,
            max_hyde_chars=args.max_hyde_chars,
            max_query_chars=args.max_query_chars,
        )
        query_embeddings = encode_texts(query_texts, model, args.batch_size)
        top_scores, top_indices = retrieve_with_dense_matrix(
            query_embeddings, doc_embeddings, args.top_k
        )
        rows = rows_for_mode(
            questions,
            docs,
            query_texts,
            top_scores,
            top_indices,
            query_mode,
            answers_by_id,
            args,
        )
        suffix = MODE_TO_SUFFIX[query_mode]
        out_path = out_dir / f"{cfg['out_prefix']}_{suffix}_top20_retrieval.jsonl"
        write_jsonl(out_path, rows)
        print(f"[{dataset_key}] Saved {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(), default=list(DATASETS))
    parser.add_argument(
        "--query_modes",
        nargs="+",
        choices=MODE_TO_SUFFIX.keys(),
        default=list(MODE_TO_SUFFIX),
    )
    parser.add_argument(
        "--out_dir",
        default="local-artifacts/topk_sensitivity/retrieval",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_hyde_chars", type=int, default=900)
    parser.add_argument("--max_query_chars", type=int, default=300)
    args = parser.parse_args()

    for dataset_key in args.datasets:
        run_dataset(dataset_key, DATASETS[dataset_key], args)


if __name__ == "__main__":
    main()
