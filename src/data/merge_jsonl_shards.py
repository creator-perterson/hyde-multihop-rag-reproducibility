import argparse
import json
from pathlib import Path


def load_rows_by_id(paths):
    rows_by_id = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_by_id[row["id"]] = row
    return rows_by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    args = parser.parse_args()

    rows_by_id = load_rows_by_id(args.shards)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    written = 0
    with Path(args.prompts).open("r", encoding="utf-8") as prompts, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in prompts:
            if not line.strip():
                continue
            prompt = json.loads(line)
            row = rows_by_id.get(prompt["id"])
            if row is None:
                missing.append(prompt["id"])
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    if missing:
        raise RuntimeError(f"Missing {len(missing)} ids; first missing id: {missing[0]}")

    print(f"Merged {written} rows to {args.out}")


if __name__ == "__main__":
    main()
