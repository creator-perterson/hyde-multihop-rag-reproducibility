import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.analyze_hyde_leakage import (
    contains_normalized_answer,
    is_short_or_ambiguous_answer,
    normalize_text,
    support_metrics,
)
from utils import read_jsonl, write_jsonl


def load_by_id(path):
    if not path:
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def build_audit_rows(query_rows, retrieval_by_id):
    audit_rows = []
    for row in query_rows:
        gold = row.get("gold_answer", row.get("answer", ""))
        query_text = row.get("prediction", "").strip()
        answer_in_query = contains_normalized_answer(gold, query_text)
        audit_row = {
            "id": row["id"],
            "question": row.get("question", ""),
            "gold_answer": gold,
            "query_text": query_text,
            "answer_in_query": int(answer_in_query),
            "short_or_ambiguous_answer": int(is_short_or_ambiguous_answer(gold)),
            "normalized_gold_answer": normalize_text(gold),
            "query_chars": len(query_text),
        }
        retrieval_row = retrieval_by_id.get(row["id"])
        if retrieval_row:
            any_hit, all_hit, recall = support_metrics(retrieval_row)
            audit_row.update({
                "any_hit": any_hit,
                "all_hit": all_hit,
                "support_recall": recall,
            })
        audit_rows.append(audit_row)
    return audit_rows


def write_markdown(path, rows, label):
    total = len(rows)
    leaked = [row for row in rows if row["answer_in_query"]]
    nontrivial = [row for row in leaked if not row["short_or_ambiguous_answer"]]
    short = [row for row in leaked if row["short_or_ambiguous_answer"]]
    no_leak = [row for row in rows if not row["answer_in_query"]]

    lines = [
        f"# {label} Answer-overlap Audit",
        "",
        "This audit checks whether the generated retrieval query contains the normalized gold answer string. It is a string-match diagnostic for query-side answer leakage risk.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total examples | {total} |",
        f"| Gold answer appears in query | {len(leaked)} ({len(leaked) / total if total else 0.0:.4f}) |",
        f"| Non-trivial answer appearances | {len(nontrivial)} ({len(nontrivial) / total if total else 0.0:.4f}) |",
        f"| Short/ambiguous answer appearances | {len(short)} ({len(short) / total if total else 0.0:.4f}) |",
        "",
    ]
    if rows and "any_hit" in rows[0]:
        groups = [
            ("answer_in_query", leaked),
            ("answer_not_in_query", no_leak),
            ("nontrivial_answer_in_query", nontrivial),
        ]
        lines.extend([
            "| Group | n | Any hit@10 | All hit@10 | Recall@10 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for name, group_rows in groups:
            lines.append(
                f"| {name} | {len(group_rows)} | {mean(group_rows, 'any_hit'):.4f} | "
                f"{mean(group_rows, 'all_hit'):.4f} | {mean(group_rows, 'support_recall'):.4f} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_answers", required=True)
    parser.add_argument("--retrieval", default="")
    parser.add_argument("--label", default="Generated Query")
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    query_rows = list(read_jsonl(args.query_answers))
    retrieval_by_id = load_by_id(args.retrieval)
    rows = build_audit_rows(query_rows, retrieval_by_id)
    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows, args.label)

    total = len(rows)
    leaked = sum(row["answer_in_query"] for row in rows)
    nontrivial = sum(
        row["answer_in_query"] and not row["short_or_ambiguous_answer"]
        for row in rows
    )
    print(f"Questions: {total}")
    print(f"Answer string in query: {leaked} ({leaked / total if total else 0.0:.4f})")
    print(f"Non-trivial answer string in query: {nontrivial} ({nontrivial / total if total else 0.0:.4f})")
    print(f"Saved audit rows to {args.out_jsonl}")
    print(f"Saved audit report to {args.out_md}")


if __name__ == "__main__":
    main()
