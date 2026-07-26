import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.retrieve_bm25_topk import rank_documents_bm25


class RetrieveBm25TopkTests(unittest.TestCase):
    def test_ranks_document_matching_query_terms_first(self):
        docs = [
            {
                "doc_id": "d1",
                "title": "Distractor",
                "text": "A paragraph about football and stadiums.",
            },
            {
                "doc_id": "d2",
                "title": "Queen Hyojeong",
                "text": "Queen Hyojeong was the consort of King Heonjong.",
            },
            {
                "doc_id": "d3",
                "title": "Heonjong of Joseon",
                "text": "Heonjong's father was Crown Prince Hyomyeong.",
            },
        ]

        ranked = rank_documents_bm25("Who was Heonjong's father?", docs, top_k=2)

        self.assertEqual(ranked[0]["title"], "Heonjong of Joseon")
        self.assertEqual(len(ranked), 2)
        self.assertIn("score", ranked[0])


if __name__ == "__main__":
    unittest.main()
