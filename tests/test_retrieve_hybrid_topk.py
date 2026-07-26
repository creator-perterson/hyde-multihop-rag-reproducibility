import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.retrieve_hybrid_topk import fuse_retrieved_docs


class RetrieveHybridTopkTests(unittest.TestCase):
    def test_fuses_ranked_lists_and_deduplicates_documents(self):
        dense_docs = [
            {"doc_id": "dense-1", "title": "Dense Only", "text": "A", "score": 0.9},
            {"doc_id": "shared", "title": "Shared", "text": "B", "score": 0.8},
        ]
        lexical_docs = [
            {"doc_id": "shared", "title": "Shared", "text": "B", "score": 12.0},
            {"doc_id": "bm25-1", "title": "BM25 Only", "text": "C", "score": 9.0},
        ]

        fused = fuse_retrieved_docs(dense_docs, lexical_docs, top_k=3)

        self.assertEqual(fused[0]["doc_id"], "shared")
        self.assertEqual(len(fused), 3)
        self.assertEqual(len({doc["doc_id"] for doc in fused}), 3)
        self.assertIn("dense_rank", fused[0])
        self.assertIn("lexical_rank", fused[0])


if __name__ == "__main__":
    unittest.main()
