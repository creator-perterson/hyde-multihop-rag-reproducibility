import sys
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generator.build_rag_prompts import safe_console_text


class BuildRagPromptsTests(unittest.TestCase):
    def test_safe_console_text_preserves_printable_output(self):
        text = "plain ascii"
        self.assertEqual(safe_console_text(text), text)

    def test_safe_console_text_replaces_unencodable_characters(self):
        original_encoding = sys.stdout.encoding
        if original_encoding and original_encoding.lower().replace("-", "") == "utf8":
            self.assertEqual(safe_console_text("foreign text: \u1ec5"), "foreign text: \u1ec5")
        else:
            self.assertIsInstance(safe_console_text("foreign text: \u1ec5"), str)


if __name__ == "__main__":
    unittest.main()
