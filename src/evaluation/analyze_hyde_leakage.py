import argparse
import re
import string
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def is_short_or_ambiguous_answer(gold_answer):
    normalized = normalize_text(gold_answer)
    if normalized in {"yes", "no"}:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized or ""):
        return True
    return len(normalized.split()) <= 1 and len(normalized) <= 3


def contains_normalized_answer(gold_answer, document):
    normalized_gold = normalize_text(gold_answer)
    normalized_doc = normalize_text(document)
    if not normalized_gold or not normalized_doc:
        return False
    return normalized_gold in normalized_doc


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hits = gold_titles & retrieved_titles
    recall = len(hits) / len(gold_titles) if gold_titles else 0.0
    return int(bool(hits)), int(gold_titles.issubset(retrieved_titles) if gold_titles else False), recall


def load_by_id(path):
    if not path:
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows):
    return {
        "n": len(rows),
        "any_hit": mean([row["any_hit"] for row in rows]),
        "all_hit": mean([row["all_hit"] for row in rows]),
        "mean_recall": mean([row["support_recall"] for row in rows]),
    }


def format_float(value):
    return f"{value:.4f}"


def write_markdown(path, rows):
    total = len(rows)
    leaked = [row for row in rows if row["answer_in_hyde"]]
    nontrivial = [row for row in leaked if not row["short_or_ambiguous_answer"]]
    short = [row for row in leaked if row["short_or_ambiguous_answer"]]
    no_leak = [row for row in rows if not row["answer_in_hyde"]]

    lines = [
        "# HyDE Hypothetical Document Leakage Audit",
        "",
        "This audit checks whether the generated hypothetical document contains the gold answer string. It is a conservative string-match diagnostic, not proof of benchmark leakage.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total examples | {total} |",
        f"| Gold answer appears in hypothetical document | {len(leaked)} ({format_float(len(leaked) / total if total else 0.0)}) |",
        f"| Non-trivial answer appearances | {len(nontrivial)} ({format_float(len(nontrivial) / total if total else 0.0)}) |",
        f"| Short/ambiguous answer appearances | {len(short)} ({format_float(len(short) / total if total else 0.0)}) |",
        "",
    ]
    if rows and "any_hit" in rows[0]:
        groups = [
            ("answer_in_hyde", leaked),
            ("answer_not_in_hyde", no_leak),
            ("nontrivial_answer_in_hyde", nontrivial),
        ]
        lines.extend([
            "## Retrieval Metrics by Leakage Group",
            "",
            "| Group | n | Any hit@10 | All hit@10 | Recall@10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, group_rows in groups:
            summary = summarize_group(group_rows)
            lines.append(
                f"| {name} | {summary['n']} | {format_float(summary['any_hit'])} | "
                f"{format_float(summary['all_hit'])} | {format_float(summary['mean_recall'])} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation Boundary",
        "",
        "A match means the hypothetical passage includes the normalized gold answer string. This can happen because the model inferred the answer from parametric memory, because the question makes the answer highly predictable, or because the generated passage paraphrases facts that include the answer. The final reader does not receive the hypothetical document as evidence in the current pipeline; it only receives retrieved documents. Therefore, the key risk is retrieval-query leakage, not direct reader leakage.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyde_answers", required=True)
    parser.add_argument("--retrieval", default="")
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    retrieval_by_id = load_by_id(args.retrieval)
    rows = []
    for row in read_jsonl(args.hyde_answers):
        gold = row.get("gold_answer", row.get("answer", ""))
        document = row.get("prediction", "")
        answer_in_hyde = contains_normalized_answer(gold, document)
        audit_row = {
            "id": row["id"],
            "question": row.get("question", ""),
            "gold_answer": gold,
            "hyde_document": document,
            "answer_in_hyde": int(answer_in_hyde),
            "short_or_ambiguous_answer": int(is_short_or_ambiguous_answer(gold)),
            "normalized_gold_answer": normalize_text(gold),
            "hyde_document_chars": len(document),
        }
        retrieval_row = retrieval_by_id.get(row["id"])
        if retrieval_row:
            any_hit, all_hit, recall = support_metrics(retrieval_row)
            audit_row.update({
                "any_hit": any_hit,
                "all_hit": all_hit,
                "support_recall": recall,
            })
        rows.append(audit_row)

    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows)
    total = len(rows)
    leaked = sum(row["answer_in_hyde"] for row in rows)
    nontrivial = sum(
        row["answer_in_hyde"] and not row["short_or_ambiguous_answer"]
        for row in rows
    )
    print(f"Questions: {total}")
    print(f"Answer string in HyDE document: {leaked} ({leaked / total:.4f})")
    print(f"Non-trivial answer string in HyDE document: {nontrivial} ({nontrivial / total:.4f})")
    print(f"Saved audit rows to {args.out_jsonl}")
    print(f"Saved audit report to {args.out_md}")


if __name__ == "__main__":
    main()
