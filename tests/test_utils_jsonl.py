import tempfile
import unittest
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from utils import read_jsonl


class UtilsJsonlTests(unittest.TestCase):
    def test_read_jsonl_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text('{"id": "1"}\n', encoding="utf-8-sig")

            rows = list(read_jsonl(path))

        self.assertEqual(rows, [{"id": "1"}])


if __name__ == "__main__":
    unittest.main()
