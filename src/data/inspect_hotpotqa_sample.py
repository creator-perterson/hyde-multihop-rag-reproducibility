import json
from pathlib import Path


def read_first_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.loads(next(f))


def main():
    base = Path("datasets/hotpotqa_sample")
    question = read_first_jsonl(base / "questions.jsonl")
    document = read_first_jsonl(base / "corpus.jsonl")

    print("First question:")
    print(json.dumps(question, ensure_ascii=False, indent=2))
    print("\nFirst corpus document:")
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
