import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


HYDE_PROMPT_TEMPLATE = """You are generating a hypothetical supporting passage for retrieval.
Given the question below, write a concise factual-sounding supporting passage that would likely contain the answer.
The passage may be imperfect, but it should include key entities, relations, and context useful for retrieving evidence.
Do not explain the task. Do not include bullet points. Write one short paragraph.

Question:
{question}

Hypothetical supporting passage:"""


def build_hyde_prompt(question):
    return HYDE_PROMPT_TEMPLATE.format(question=question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_hyde_prompts.jsonl")
    args = parser.parse_args()

    rows = []
    for question in read_jsonl(args.questions):
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": [],
            "prompt": build_hyde_prompt(question["question"]),
        })

    write_jsonl(args.out, rows)
    print(f"Saved {len(rows)} HyDE prompts to {args.out}")
    if rows:
        print("First prompt preview:")
        print(rows[0]["prompt"])


if __name__ == "__main__":
    main()
