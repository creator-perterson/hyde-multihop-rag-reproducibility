import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


LLM_ONLY_TEMPLATE = """You are a careful factual question answering system.
Answer the question using your parametric knowledge only.
Most questions have a short factual answer such as a name, place, number, yes, or no.
Return only the final short answer. Do not explain.

Question:
{question}

Short answer:"""


def build_prompt(question):
    return LLM_ONLY_TEMPLATE.format(question=question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="datasets/hotpotqa_sample/questions.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_llm_only_prompts.jsonl")
    args = parser.parse_args()

    rows = []
    for question in read_jsonl(args.questions):
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": [],
            "prompt": build_prompt(question["question"]),
        })

    write_jsonl(args.out, rows)
    print(f"Saved {len(rows)} LLM-only prompts to {args.out}")
    if rows:
        print("\nFirst prompt preview:\n")
        print(rows[0]["prompt"])


if __name__ == "__main__":
    main()
