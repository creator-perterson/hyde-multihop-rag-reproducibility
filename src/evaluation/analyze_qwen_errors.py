import argparse
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match, f1_score, normalize_answer
from utils import read_jsonl, write_jsonl


ALIAS_PAIRS = {
    ("american", "united states"),
    ("american", "united states citizen"),
    ("united states", "american"),
    ("united states citizen", "american"),
    ("british", "united kingdom"),
    ("english", "england"),
}


def evidence_text(row):
    return " ".join(doc.get("text", "") for doc in row.get("retrieved", []))


def supporting_title_stats(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hit_titles = gold_titles & retrieved_titles
    return gold_titles, retrieved_titles, hit_titles


def is_alias_or_format(prediction, gold):
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if (pred_norm, gold_norm) in ALIAS_PAIRS:
        return True
    if f1_score(prediction, gold) > 0 and not exact_match(prediction, gold):
        return True
    return False


def classify_error(row):
    prediction = row["prediction"]
    gold = row["gold_answer"]
    gold_titles, _, hit_titles = supporting_title_stats(row)
    all_support_hit = bool(gold_titles) and gold_titles <= hit_titles
    partial_support_hit = bool(hit_titles)
    pred_norm = normalize_answer(prediction)
    ev_norm = normalize_answer(evidence_text(row))

    if exact_match(prediction, gold):
        error_type = "correct"
    elif is_alias_or_format(prediction, gold):
        error_type = "answer_format_or_alias"
    elif gold_titles and not all_support_hit:
        error_type = "retrieval_miss"
    elif pred_norm in {"i dont know", "unknown", "not enough information"}:
        error_type = "over_abstention"
    elif pred_norm and pred_norm not in ev_norm:
        error_type = "hallucination_or_unsupported"
    else:
        error_type = "reader_reasoning_error"

    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "gold_answer": gold,
        "prediction": prediction,
        "em": int(exact_match(prediction, gold)),
        "f1": f1_score(prediction, gold),
        "error_type": error_type,
        "gold_support_count": len(gold_titles),
        "hit_support_count": len(hit_titles),
        "all_support_hit": int(all_support_hit),
        "retrieved_titles": [doc.get("title", "") for doc in row.get("retrieved", [])],
        "gold_titles": sorted(gold_titles),
    }


def write_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["error_type"] for row in rows)
    wrong = [row for row in rows if row["error_type"] != "correct"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Qwen Error Analysis Summary\n\n")
        f.write(f"Total examples: {len(rows)}\n\n")
        f.write(f"Wrong by EM: {len(wrong)}\n\n")
        f.write("## Error Type Counts\n\n")
        f.write("| Error type | Count |\n|---|---:|\n")
        for key, value in counts.most_common():
            f.write(f"| {key} | {value} |\n")
        f.write("\n## Representative Wrong Cases\n\n")
        for row in wrong[:20]:
            f.write(f"### {row['error_type']} - {row['id']}\n\n")
            f.write(f"Q: {row['question']}\n\n")
            f.write(f"Gold: `{row['gold_answer']}`\n\n")
            f.write(f"Pred: `{row['prediction']}`\n\n")
            f.write(f"F1: {row['f1']:.4f}; support hit: {row['hit_support_count']}/{row['gold_support_count']}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    rows = [classify_error(row) for row in read_jsonl(args.answers)]
    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows)

    counts = Counter(row["error_type"] for row in rows)
    print(f"Analyzed examples: {len(rows)}")
    for key, value in counts.most_common():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
