import unittest
from pathlib import Path


class CrossReaderRunnerScriptTests(unittest.TestCase):
    def test_runner_has_resume_and_skip_guards_for_two_readers(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "run_equal_budget_500_cross_reader.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("qwen3.7-max", text)
        self.assertIn("qwen-turbo", text)
        self.assertIn("--resume", text)
        self.assertRegex(text, r"Count-JsonlRows\s+\$answers")
        self.assertRegex(text, r"rowsBefore\s+-ge\s+500")
        self.assertIn("summarize_equal_budget_reader.py", text)

    def test_runner_supports_a_minimal_direct_vs_document_reader_check(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "run_equal_budget_500_cross_reader.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("[string[]]$ModeFilter", text)
        self.assertIn("[string[]]$ReaderFilter", text)
        self.assertIn("--mode", text)
        self.assertIn("$ModeFilter", text)
        self.assertIn("$ReaderFilter", text)

    def test_runner_normalizes_duplicate_windows_path_environment_before_spawning(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "run_equal_budget_500_cross_reader.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn("function Normalize-ProcessPath", text)
        self.assertIn('SetEnvironmentVariable("PATH", $null, "Process")', text)
        self.assertIn("Normalize-ProcessPath", text)

    def test_runner_resolves_summary_script_from_code_root(self):
        script = Path(__file__).resolve().parents[1] / "experiments" / "run_equal_budget_500_cross_reader.ps1"
        text = script.read_text(encoding="utf-8")
        self.assertIn('Join-Path $CodeRoot "src\\evaluation\\summarize_equal_budget_reader.py"', text)


if __name__ == "__main__":
    unittest.main()
