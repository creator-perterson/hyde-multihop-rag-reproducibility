import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score
from evaluation.analyze_qwen_errors import is_alias_or_format, normalize_answer
from utils import read_jsonl


CATEGORY_ORDER = {
    "correct": 0,
    "retrieval miss": 1,
    "partial evidence / missing hop": 2,
    "reader reasoning error": 3,
    "answer format / alias error": 4,
}

MANUAL_ALIAS_OVERRIDES = {
    # Evidence states that Martin Ingerman was known professionally as Marty Ingels.
    # EM rejects the professional name, so this is an evaluation/canonical-name
    # mismatch rather than a reader reasoning failure.
    "5a728f015542991f9a20c4e4",
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "first": "1",
    "two": "2",
    "second": "2",
    "three": "3",
    "third": "3",
    "four": "4",
    "fourth": "4",
    "five": "5",
    "fifth": "5",
    "six": "6",
    "sixth": "6",
    "seven": "7",
    "seventh": "7",
    "eight": "8",
    "eighth": "8",
    "nine": "9",
    "ninth": "9",
    "ten": "10",
    "tenth": "10",
    "eleven": "11",
    "eleventh": "11",
    "twelve": "12",
    "twelfth": "12",
    "thirteen": "13",
    "thirteenth": "13",
    "fourteen": "14",
    "fourteenth": "14",
    "fifteen": "15",
    "fifteenth": "15",
    "sixteen": "16",
    "sixteenth": "16",
    "seventeen": "17",
    "seventeenth": "17",
    "eighteen": "18",
    "eighteenth": "18",
    "nineteen": "19",
    "nineteenth": "19",
    "twenty": "20",
    "twentieth": "20",
}


def is_number_word_format(prediction, gold):
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return NUMBER_WORDS.get(pred_norm) == gold_norm or NUMBER_WORDS.get(gold_norm) == pred_norm


def is_format_or_alias_error(prediction, gold):
    return is_alias_or_format(prediction, gold) or is_number_word_format(prediction, gold)


def supporting_title_stats(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hit_titles = gold_titles & retrieved_titles
    return gold_titles, retrieved_titles, hit_titles


def support_recall(gold_titles, hit_titles):
    return len(hit_titles) / len(gold_titles) if gold_titles else 0.0


def classify_four_way_error(row):
    prediction = row.get("prediction", row.get("final_answer", ""))
    gold = row["gold_answer"]
    gold_titles, _, hit_titles = supporting_title_stats(row)
    em = float(exact_match(prediction, gold))
    f1 = f1_score(prediction, gold)

    if em:
        category = "correct"
    elif gold_titles and len(hit_titles) == 0:
        category = "retrieval miss"
    elif gold_titles and len(hit_titles) < len(gold_titles):
        category = "partial evidence / missing hop"
    elif row.get("id", "") in MANUAL_ALIAS_OVERRIDES or is_format_or_alias_error(prediction, gold):
        category = "answer format / alias error"
    else:
        category = "reader reasoning error"

    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "gold_answer": gold,
        "prediction": prediction,
        "error_category": category,
        "em": em,
        "f1": f1,
        "gold_support_count": len(gold_titles),
        "hit_support_count": len(hit_titles),
        "support_recall": support_recall(gold_titles, hit_titles),
        "gold_titles": "; ".join(sorted(gold_titles)),
        "hit_titles": "; ".join(sorted(hit_titles)),
        "retrieved_titles": "; ".join(doc.get("title", "") for doc in row.get("retrieved", [])),
    }


def load_rows_with_optional_base(answers_path, base_answers_path=None):
    answer_rows = list(read_jsonl(answers_path))
    if not base_answers_path:
        return answer_rows

    base_by_id = {row["id"]: row for row in read_jsonl(base_answers_path)}
    merged = []
    for row in answer_rows:
        base = base_by_id.get(row["id"], {})
        prediction = row.get("final_answer", row.get("prediction", ""))
        merged_row = {
            **base,
            **row,
            "prediction": prediction,
            "supporting_facts": base.get("supporting_facts", row.get("supporting_facts", {})),
            "retrieved": base.get("retrieved", row.get("retrieved", [])),
        }
        merged.append(merged_row)
    return merged


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    total = len(records)
    wrong_total = sum(1 for row in records if row["error_category"] != "correct")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["error_category"]].append(row)

    summary = []
    for category, rows in grouped.items():
        count = len(rows)
        summary.append({
            "error_category": category,
            "count": count,
            "percent_all": count / total if total else 0.0,
            "percent_errors": (
                count / wrong_total
                if wrong_total and category != "correct"
                else 0.0
            ),
            "mean_f1": mean([row["f1"] for row in rows]),
            "mean_support_recall": mean([row["support_recall"] for row in rows]),
        })
    return sorted(summary, key=lambda row: CATEGORY_ORDER.get(row["error_category"], 99))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def representative_examples(records, per_category=3):
    examples = []
    counts = Counter()
    for row in records:
        category = row["error_category"]
        if category == "correct":
            continue
        if counts[category] >= per_category:
            continue
        examples.append(row)
        counts[category] += 1
    return examples


def write_markdown(path, method_name, summary, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    errors = sum(1 for row in records if row["error_category"] != "correct")
    lines = [
        "# Error Analysis Table",
        "",
        f"Method: {method_name}",
        "",
        f"Total examples: {total}",
        "",
        f"Wrong by exact match: {errors}",
        "",
        "## Summary",
        "",
        "| Error category | Count | % all | % errors | Mean F1 | Mean support recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['error_category']} | {row['count']} | "
            f"{fmt(row['percent_all'])} | {fmt(row['percent_errors'])} | "
            f"{fmt(row['mean_f1'])} | {fmt(row['mean_support_recall'])} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The table separates retrieval failures from reader-side failures. `retrieval miss` means no gold supporting title is retrieved; `partial evidence / missing hop` means at least one but not all supporting titles are retrieved; `reader reasoning error` means all supporting titles are retrieved but the reader still gives a wrong answer; `answer format / alias error` captures semantically close or canonicalization-related mismatches.",
        "",
        "## Representative Errors",
        "",
    ])
    for row in representative_examples(records):
        lines.extend([
            f"### {row['error_category']} - {row['id']}",
            "",
            f"Q: {row['question']}",
            "",
            f"Gold: `{row['gold_answer']}`",
            "",
            f"Pred: `{row['prediction']}`",
            "",
            f"Support hit: {row['hit_support_count']}/{row['gold_support_count']}; F1: {row['f1']:.4f}",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--base_answers", default=None)
    parser.add_argument("--method_name", default="method")
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_examples_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = load_rows_with_optional_base(args.answers, args.base_answers)
    records = [classify_four_way_error(row) for row in rows]
    summary = summarize(records)

    write_csv(
        args.out_summary_csv,
        summary,
        ["error_category", "count", "percent_all", "percent_errors", "mean_f1", "mean_support_recall"],
    )
    write_csv(
        args.out_examples_csv,
        records,
        [
            "id",
            "question",
            "gold_answer",
            "prediction",
            "error_category",
            "em",
            "f1",
            "gold_support_count",
            "hit_support_count",
            "support_recall",
            "gold_titles",
            "hit_titles",
            "retrieved_titles",
        ],
    )
    write_markdown(args.out_md, args.method_name, summary, records)

    print(f"Analyzed examples: {len(records)}")
    for row in summary:
        print(f"{row['error_category']}: {row['count']} ({row['percent_all']:.4f})")


if __name__ == "__main__":
    main()
