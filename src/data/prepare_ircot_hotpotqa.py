import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def extract_answer(answers_objects):
    if not answers_objects:
        return ""
    answer = answers_objects[0]
    spans = answer.get("spans") or []
    if spans:
        return str(spans[0])
    number = answer.get("number")
    if number:
        return str(number)
    date = answer.get("date") or {}
    date_parts = [date.get("day"), date.get("month"), date.get("year")]
    date_text = " ".join(str(part) for part in date_parts if part)
    return date_text


def convert_ircot_hotpotqa_rows(rows):
    questions = []
    corpus = []

    for row in rows:
        question_id = row["question_id"]
        question_text = row["question_text"].strip()
        contexts = row.get("contexts", [])

        supporting_titles = [
            context["title"]
            for context in contexts
            if context.get("is_supporting")
        ]

        questions.append({
            "id": question_id,
            "question": question_text,
            "answer": extract_answer(row.get("answers_objects", [])),
            "supporting_facts": {"title": supporting_titles},
        })

        for context in contexts:
            title = context["title"]
            idx = context.get("idx", len(corpus))
            corpus.append({
                "doc_id": f"{question_id}::{idx}::{title}",
                "title": title,
                "text": context["paragraph_text"].strip(),
                "source_question_id": question_id,
                "is_supporting": bool(context.get("is_supporting")),
            })

    return questions, corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../_external/paper_repos/ircot/processed_data/hotpotqa/test_subsampled.jsonl",
    )
    parser.add_argument("--out_dir", default="datasets/ircot_hotpotqa_test500")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    questions, corpus = convert_ircot_hotpotqa_rows(rows)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "questions.jsonl", questions)
    write_jsonl(out_dir / "corpus.jsonl", corpus)

    print(f"Loaded IRCoT HotpotQA rows: {len(rows)}")
    print(f"Saved questions: {len(questions)} -> {out_dir / 'questions.jsonl'}")
    print(f"Saved corpus docs: {len(corpus)} -> {out_dir / 'corpus.jsonl'}")
    if questions:
        print("First question:")
        print(questions[0]["question"])
        print(f"answer: {questions[0]['answer']}")
        print(f"supporting_titles: {questions[0]['supporting_facts']['title']}")


if __name__ == "__main__":
    main()
