import argparse
import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset


def sentence_list_to_text(sentences):
    return " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out_dir", default="datasets/hotpotqa_sample")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=args.split,
        streaming=args.streaming,
    )
    if args.streaming:
        rows = list(islice(ds, args.limit))
    else:
        rows = list(ds.select(range(min(args.limit, len(ds)))))

    questions_path = out_dir / "questions.jsonl"
    corpus_path = out_dir / "corpus.jsonl"

    seen_docs = set()
    doc_count = 0

    with questions_path.open("w", encoding="utf-8") as qf, corpus_path.open("w", encoding="utf-8") as cf:
        for idx, row in enumerate(rows):
            question_id = row.get("id", str(idx))
            question = row["question"]
            answer = row["answer"]
            context = row["context"]

            qf.write(json.dumps({
                "id": question_id,
                "question": question,
                "answer": answer,
                "level": row.get("level"),
                "type": row.get("type"),
                "supporting_facts": row.get("supporting_facts"),
            }, ensure_ascii=False) + "\n")

            titles = context["title"]
            sentence_groups = context["sentences"]
            for title, sentences in zip(titles, sentence_groups):
                text = sentence_list_to_text(sentences)
                if not text:
                    continue
                doc_id = f"{question_id}::{title}"
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                doc_count += 1
                cf.write(json.dumps({
                    "doc_id": doc_id,
                    "source_question_id": question_id,
                    "title": title,
                    "text": text,
                }, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} questions to {questions_path}")
    print(f"Saved {doc_count} context documents to {corpus_path}")


if __name__ == "__main__":
    main()
