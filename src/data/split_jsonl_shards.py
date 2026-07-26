import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    if args.shards < 1:
        raise ValueError("--shards must be >= 1")

    handles = []
    try:
        for shard_idx in range(args.shards):
            out_path = Path(f"{args.out_prefix}.shard{shard_idx}.jsonl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(out_path.open("w", encoding="utf-8"))

        counts = [0] * args.shards
        with Path(args.input).open("r", encoding="utf-8") as src:
            for line_idx, line in enumerate(src):
                shard_idx = line_idx % args.shards
                handles[shard_idx].write(line)
                counts[shard_idx] += 1
    finally:
        for handle in handles:
            handle.close()

    print(f"Split {sum(counts)} rows into {args.shards} shards")
    for shard_idx, count in enumerate(counts):
        print(f"shard{shard_idx}: {count}")


if __name__ == "__main__":
    main()
