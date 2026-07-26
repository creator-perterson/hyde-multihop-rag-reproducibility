import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from retriever.retrieve_hyde_topk import build_hyde_query_text, format_retrieved_docs


class RetrieveHydeTopkTests(unittest.TestCase):
    def test_builds_query_from_question_and_hypothetical_document(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "The novel was written by a British author.",
        }

        query = build_hyde_query_text(row)

        self.assertIn("Who wrote the novel?", query)
        self.assertIn("British author", query)

    def test_builds_hypothetical_only_query(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "The novel was written by a British author.",
        }

        query = build_hyde_query_text(row, query_mode="hypothetical_only")

        self.assertEqual(query, "The novel was written by a British author.")

    def test_builds_question_only_query(self):
        row = {
            "question": "Who wrote the novel?",
            "prediction": "The novel was written by a British author.",
        }

        query = build_hyde_query_text(row, query_mode="question_only")

        self.assertEqual(query, "Who wrote the novel?")

    def test_formats_retrieved_docs_with_hyde_metadata(self):
        docs = [
            {"doc_id": "d1", "title": "Doc One", "text": "A", "source_question_id": "q1"},
            {"doc_id": "d2", "title": "Doc Two", "text": "B", "source_question_id": "q1"},
        ]

        retrieved = format_retrieved_docs([0.9, 0.7], [1, 0], docs)

        self.assertEqual(retrieved[0]["doc_id"], "d2")
        self.assertEqual(retrieved[1]["doc_id"], "d1")
        self.assertIn("score", retrieved[0])


if __name__ == "__main__":
    unittest.main()
