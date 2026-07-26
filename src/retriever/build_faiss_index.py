import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="datasets/hotpotqa_sample/corpus.jsonl")
    parser.add_argument("--out_dir", default="datasets/hotpotqa_sample/faiss_index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = list(read_jsonl(args.corpus))
    texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    print(f"Loaded {len(docs)} corpus documents")
    print(f"Embedding model: {args.model}")

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    write_jsonl(out_dir / "docstore.jsonl", docs)

    metadata = {
        "model": args.model,
        "num_docs": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP with normalized embeddings",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved FAISS index to {out_dir / 'index.faiss'}")
    print(f"Saved docstore to {out_dir / 'docstore.jsonl'}")


if __name__ == "__main__":
    main()
