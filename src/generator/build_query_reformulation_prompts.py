import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


QUERY_REFORMULATION_PROMPT_TEMPLATE = """You are rewriting a multi-hop question into a single retrieval query.
Given the question below, produce one concise search query for dense retrieval.
Use only entities, relations, and constraints already present in the question.
Do not answer the question. Do not add facts that are not stated in the question.
Do not write a paragraph or explanation. Output only the single retrieval query.

Question:
{question}

Single retrieval query:"""


def build_query_reformulation_prompt(question):
    return QUERY_REFORMULATION_PROMPT_TEMPLATE.format(question=question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_query_reformulation_prompts.jsonl")
    args = parser.parse_args()

    rows = []
    for question in read_jsonl(args.questions):
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": [],
            "prompt": build_query_reformulation_prompt(question["question"]),
            "query_reformulation_strategy": {
                "name": "single_query_reformulation",
                "constraint": "question_terms_only_no_answer",
            },
        })

    write_jsonl(args.out, rows)
    print(f"Saved {len(rows)} query reformulation prompts to {args.out}")
    if rows:
        print("First prompt preview:")
        print(rows[0]["prompt"])


if __name__ == "__main__":
    main()
