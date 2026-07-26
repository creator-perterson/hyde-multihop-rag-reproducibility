import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.generate_answers_openai_compatible import build_messages


class GenerateAnswersOpenAICompatibleTests(unittest.TestCase):
    def test_build_messages_uses_default_reader_system_prompt(self):
        row = {"prompt": "Question:\nQ?\n\nEvidence:\nE"}

        messages = build_messages(row)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("short factual answers", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": row["prompt"]})

    def test_build_messages_accepts_custom_system_prompt(self):
        row = {"prompt": "Rewrite this question."}

        messages = build_messages(row, system_prompt="Return only one retrieval query.")

        self.assertEqual(messages[0]["content"], "Return only one retrieval query.")
        self.assertEqual(messages[1], {"role": "user", "content": row["prompt"]})


if __name__ == "__main__":
    unittest.main()
