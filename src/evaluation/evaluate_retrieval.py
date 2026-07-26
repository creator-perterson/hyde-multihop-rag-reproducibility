import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="results/hotpotqa_top5_retrieval.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.retrieval))
    total = 0
    any_hit = 0
    all_hit = 0
    recall_sum = 0.0

    for row in rows:
        gold_titles = set(row["supporting_facts"]["title"])
        retrieved_titles = {doc["title"] for doc in row["retrieved"]}
        hit_titles = gold_titles & retrieved_titles

        total += 1
        if hit_titles:
            any_hit += 1
        if gold_titles and gold_titles.issubset(retrieved_titles):
            all_hit += 1
        if gold_titles:
            recall_sum += len(hit_titles) / len(gold_titles)

    print(f"Questions: {total}")
    print(f"Any supporting-title hit@k: {any_hit / total:.4f}")
    print(f"All supporting-title hit@k: {all_hit / total:.4f}")
    print(f"Mean supporting-title recall@k: {recall_sum / total:.4f}")


if __name__ == "__main__":
    main()
