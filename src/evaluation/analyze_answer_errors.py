import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluate_answers import exact_match, f1_score
from utils import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="results/hotpotqa_top5_answers.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    wrong = []
    for row in rows:
        em = exact_match(row["prediction"], row["gold_answer"])
        f1 = f1_score(row["prediction"], row["gold_answer"])
        gold_titles = set(row.get("supporting_facts", {}).get("title", []))
        retrieved_titles = {doc["title"] for doc in row.get("retrieved", [])}
        hit_titles = gold_titles & retrieved_titles
        wrong.append({
            "row": row,
            "em": em,
            "f1": f1,
            "gold_titles": gold_titles,
            "retrieved_titles": retrieved_titles,
            "hit_titles": hit_titles,
        })

    wrong = [item for item in wrong if not item["em"]]
    print(f"Wrong examples: {len(wrong)} / {len(rows)}")
    print()
    for item in wrong[:20]:
        row = item["row"]
        print("-" * 100)
        print(f"Q: {row['question']}")
        print(f"Gold answer: {row['gold_answer']}")
        print(f"Prediction: {row['prediction']}")
        print(f"F1: {item['f1']:.4f}")
        print(f"Gold supporting titles: {sorted(item['gold_titles'])}")
        print(f"Retrieved supporting hits: {sorted(item['hit_titles'])}")
        print("Retrieved titles:")
        for rank, doc in enumerate(row.get("retrieved", []), start=1):
            marker = "*" if doc["title"] in item["gold_titles"] else " "
            print(f"  {rank}. {marker} {doc['title']}")


if __name__ == "__main__":
    main()
