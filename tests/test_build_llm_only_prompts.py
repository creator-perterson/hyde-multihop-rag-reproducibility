import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.build_llm_only_prompts import build_prompt


class BuildLlmOnlyPromptsTests(unittest.TestCase):
    def test_prompt_contains_question_without_evidence_block(self):
        prompt = build_prompt("Who wrote the novel?")

        self.assertIn("Who wrote the novel?", prompt)
        self.assertNotIn("Evidence:", prompt)
        self.assertIn("Short answer:", prompt)


if __name__ == "__main__":
    unittest.main()
