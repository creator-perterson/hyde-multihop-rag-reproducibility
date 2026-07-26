import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.build_query_reformulation_prompts import build_query_reformulation_prompt


class BuildQueryReformulationPromptsTests(unittest.TestCase):
    def test_prompt_requests_single_retrieval_query_without_answering(self):
        prompt = build_query_reformulation_prompt("Who wrote the novel?")

        self.assertIn("Who wrote the novel?", prompt)
        self.assertIn("single retrieval query", prompt.lower())
        self.assertIn("do not answer", prompt.lower())
        self.assertIn("output only", prompt.lower())
        self.assertNotIn("Hypothetical supporting passage:", prompt)
        self.assertNotIn("Evidence:", prompt)


if __name__ == "__main__":
    unittest.main()
