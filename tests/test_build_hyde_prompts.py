import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.build_hyde_prompts import build_hyde_prompt


class BuildHydePromptsTests(unittest.TestCase):
    def test_prompt_asks_for_hypothetical_evidence_without_gold_answer(self):
        prompt = build_hyde_prompt("Who wrote the novel?")

        self.assertIn("Who wrote the novel?", prompt)
        self.assertIn("hypothetical", prompt.lower())
        self.assertIn("supporting passage", prompt.lower())
        self.assertNotIn("Evidence:", prompt)
        self.assertNotIn("gold", prompt.lower())


if __name__ == "__main__":
    unittest.main()
