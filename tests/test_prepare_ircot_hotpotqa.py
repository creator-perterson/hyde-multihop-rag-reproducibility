import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.prepare_ircot_hotpotqa import convert_ircot_hotpotqa_rows


class PrepareIrcotHotpotqaTest(unittest.TestCase):
    def test_converts_ircot_rows_to_existing_rag_format(self):
        rows = [
            {
                "question_id": "q1",
                "question_text": "Who wrote the novel?",
                "answers_objects": [
                    {
                        "spans": ["Alice Smith"],
                        "number": "",
                        "date": {"day": "", "month": "", "year": ""},
                    }
                ],
                "contexts": [
                    {
                        "idx": 0,
                        "title": "Novel",
                        "paragraph_text": "Novel evidence.",
                        "is_supporting": True,
                    },
                    {
                        "idx": 1,
                        "title": "Distractor",
                        "paragraph_text": "Distractor evidence.",
                        "is_supporting": False,
                    },
                ],
            }
        ]

        questions, corpus = convert_ircot_hotpotqa_rows(rows)

        self.assertEqual(
            questions,
            [
                {
                    "id": "q1",
                    "question": "Who wrote the novel?",
                    "answer": "Alice Smith",
                    "supporting_facts": {"title": ["Novel"]},
                }
            ],
        )
        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0]["doc_id"], "q1::0::Novel")
        self.assertEqual(corpus[0]["title"], "Novel")
        self.assertEqual(corpus[0]["text"], "Novel evidence.")
        self.assertEqual(corpus[0]["source_question_id"], "q1")


if __name__ == "__main__":
    unittest.main()
