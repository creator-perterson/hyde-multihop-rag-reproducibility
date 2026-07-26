import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


EQUAL_BUDGET_QUERY_MODES = [
    "keyword_expansion",
    "direct_rewrite",
    "question_decomposition",
    "document_like_passage",
]


MODE_INSTRUCTIONS = {
    "keyword_expansion": (
        "Write a compact keyword/entity list for dense retrieval. Include entities, "
        "relations, constraints, and likely bridge concepts. Use comma-separated phrases, "
        "not sentences."
    ),
    "direct_rewrite": (
        "Write one single dense-retrieval rewrite of the question. Preserve the original "
        "intent, make implicit relations explicit, and keep it as one sentence."
    ),
    "question_decomposition": (
        "Write numbered subquestions that decompose the multi-hop question into retrieval "
        "steps. Keep the total output within the same length budget."
    ),
    "document_like_passage": (
        "Write one short encyclopedia-style passage that could plausibly support answering "
        "the question. Use prose rather than bullets."
    ),
}


def length_window(target_words):
    return max(1, target_words - 5), target_words + 5


def build_equal_budget_prompt(question, mode, target_words=40):
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"Unsupported equal-budget query mode: {mode}")
    low, high = length_window(target_words)
    return f"""You are generating retrieval text for a controlled multi-hop QA query-composition diagnostic.
{MODE_INSTRUCTIONS[mode]}

Use only one generation call. Do not answer the question. Do not explain the task.
Aim for {low}-{high} English words. Prefer concrete entities and relations over filler.
Output only the retrieval text.

Question:
{question}

Retrieval text:"""


def build_prompt_row(question_row, mode, target_words=40, max_tokens=64):
    return {
        "id": question_row["id"],
        "question": question_row["question"],
        "gold_answer": question_row["answer"],
        "supporting_facts": question_row.get("supporting_facts", {}),
        "retrieved": [],
        "prompt": build_equal_budget_prompt(
            question_row["question"],
            mode=mode,
            target_words=target_words,
        ),
        "equal_budget_query_mode": mode,
        "budget": {
            "target_words": target_words,
            "target_window": list(length_window(target_words)),
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out_dir", default="results/equal_budget_query_prompts")
    parser.add_argument("--target_words", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--modes", nargs="+", choices=EQUAL_BUDGET_QUERY_MODES, default=EQUAL_BUDGET_QUERY_MODES)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        rows = [
            build_prompt_row(
                question,
                mode=mode,
                target_words=args.target_words,
                max_tokens=args.max_tokens,
            )
            for question in questions
        ]
        out_path = out_dir / f"{Path(args.questions).parent.name}_{mode}_prompts.jsonl"
        write_jsonl(out_path, rows)
        print(f"Saved {len(rows)} {mode} prompts to {out_path}")


if __name__ == "__main__":
    main()
