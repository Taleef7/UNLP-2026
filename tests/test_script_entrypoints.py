import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_benchmark_candidate_help_runs(self):
        proc = subprocess.run(
            [sys.executable, "scripts/benchmark_candidate.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("UNLP 2026 local benchmark runner", proc.stdout)

    def test_build_paper_ablation_audit_runs(self):
        proc = subprocess.run(
            [sys.executable, "scripts/build_paper_ablation_audit.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("master_table", proc.stdout)

    def test_run_paper_followup_suite_help_runs(self):
        proc = subprocess.run(
            [sys.executable, "scripts/run_paper_followup_suite.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("paper-facing follow-up suites", proc.stdout)

    def test_backfill_paper_notes_runs(self):
        proc = subprocess.run(
            [sys.executable, "scripts/backfill_paper_notes.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("written", proc.stdout)


if __name__ == "__main__":
    unittest.main()
