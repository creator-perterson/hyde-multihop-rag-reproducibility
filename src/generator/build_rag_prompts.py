import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


STRICT_PROMPT_TEMPLATE = """You are a question answering system.
Answer the question using only the evidence below.
If the evidence is clearly insufficient, answer "I don't know".
Keep the answer short. Do not explain.

Question:
{question}

Evidence:
{evidence}

Short answer:"""


INFER_PROMPT_TEMPLATE = """You are a careful multi-hop question answering system.
Use only the evidence below, but you may combine facts across evidence items.
Most questions have a short factual answer such as a name, place, number, yes, or no.
Return only the final short answer. Do not explain.
Only answer "I don't know" when the needed facts are not present in the evidence.

Question:
{question}

Evidence:
{evidence}

Short answer:"""


EXTRACTIVE_PROMPT_TEMPLATE = """You are an extractive multi-hop question answering system.
Use only the evidence below. Combine facts across evidence items when needed.
Return the exact short answer phrase requested by the question.

Important rules:
- Return only the answer, with no explanation.
- Prefer the most specific answer phrase in the evidence.
- For yes/no questions, answer only "yes" or "no".
- For location questions, include the full location if the evidence gives it.
- For date/age comparison questions, the earlier birth date means the person is older.
- For role/position questions, return the exact position title, not a broader category.
- Do not answer "I don't know" if the evidence contains enough information to infer the answer.

Question:
{question}

Evidence:
{evidence}

Exact short answer:"""


def format_evidence(retrieved_docs, top_k, max_chars_per_doc):
    chunks = []
    for rank, doc in enumerate(retrieved_docs[:top_k], start=1):
        text = doc["text"].strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        chunks.append(f"[{rank}] Title: {doc['title']}\n{text}")
    return "\n\n".join(chunks)


def safe_console_text(text):
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="results/hotpotqa_top5_retrieval.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_top5_prompts.jsonl")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_chars_per_doc", type=int, default=900)
    parser.add_argument("--style", choices=["strict", "infer", "extractive"], default="infer")
    args = parser.parse_args()

    if args.style == "extractive":
        template = EXTRACTIVE_PROMPT_TEMPLATE
    elif args.style == "infer":
        template = INFER_PROMPT_TEMPLATE
    else:
        template = STRICT_PROMPT_TEMPLATE
    rows = []
    for row in read_jsonl(args.retrieval):
        evidence = format_evidence(row["retrieved"], args.top_k, args.max_chars_per_doc)
        prompt = template.format(question=row["question"], evidence=evidence)
        rows.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["answer"],
            "supporting_facts": row["supporting_facts"],
            "retrieved": row["retrieved"][:args.top_k],
            "prompt": prompt,
        })

    write_jsonl(args.out, rows)
    print(f"Saved {len(rows)} prompts to {args.out}")
    print("\nFirst prompt preview:\n")
    print(safe_console_text(rows[0]["prompt"]))


if __name__ == "__main__":
    main()
