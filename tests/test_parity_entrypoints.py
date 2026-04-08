import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import scripts.benchmark_candidate as benchmark_candidate
import scripts.check_kaggle_parity as check_kaggle_parity


class BenchmarkEntrypointTests(unittest.TestCase):
    def test_parse_benchmark_args_includes_lane_flags(self):
        args = benchmark_candidate.parse_args(
            [
                "--preset",
                "v7_baseline",
                "--questions",
                "data/dev_questions.csv",
                "--split",
                "grouped_cv",
                "--fold",
                "3",
                "--repeat-count",
                "2",
                "--seed",
                "17",
                "--hardware-tag",
                "t4x2",
                "--parity-mode",
                "compare",
            ]
        )
        self.assertEqual(args.preset, "v7_baseline")
        self.assertEqual(args.split, "grouped_cv")
        self.assertEqual(args.fold, "3")
        self.assertEqual(args.repeat_count, 2)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.hardware_tag, "t4x2")
        self.assertEqual(args.parity_mode, "compare")

    def test_resolve_questions_path_prefers_split_and_fold_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir) / "data" / "splits"
            splits_dir.mkdir(parents=True)
            fallback = splits_dir / "grouped_cv.csv"
            fallback.write_text("Question_ID\n1\n", encoding="utf-8")
            fold_path = splits_dir / "grouped_cv_fold_3.csv"
            fold_path.write_text("Question_ID\n2\n", encoding="utf-8")

            resolved = benchmark_candidate.resolve_questions_path(
                questions=None,
                split="grouped_cv",
                fold="3",
                splits_dir=splits_dir,
            )
            self.assertEqual(resolved, fold_path)

            resolved_fallback = benchmark_candidate.resolve_questions_path(
                questions=None,
                split="grouped_cv",
                fold=None,
                splits_dir=splits_dir,
            )
            self.assertEqual(resolved_fallback, fallback)

    def test_resolve_questions_path_supports_grouped_cv_fold_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir) / "data" / "splits"
            splits_dir.mkdir(parents=True)
            fold_val = splits_dir / "fold_3_val.csv"
            fold_val.write_text("Question_ID\n3\n", encoding="utf-8")

            resolved = benchmark_candidate.resolve_questions_path(
                questions=None,
                split="grouped_cv",
                fold="3",
                splits_dir=splits_dir,
            )

        self.assertEqual(resolved, fold_val)

    def test_build_run_manifest_records_requested_metadata(self):
        manifest = benchmark_candidate.build_run_manifest(
            preset_name="v7_baseline",
            questions_path=Path("data/dev_questions.csv"),
            output_dir=Path("outputs/benchmarks/v7_baseline"),
            env="local",
            seed=17,
            hardware_tag="t4x2",
            split="grouped_cv",
            fold="3",
            repeat_index=2,
            repeat_count=4,
            parity_mode="compare",
            runtime_budget_hours=7.5,
            overrides={"runtime": {"fast_mode": True}},
            result_summary={
                "reranker_name": "qwen3_0_6b",
                "llm_n_ctx": 8192,
                "n_questions": 2,
            },
        )
        self.assertEqual(manifest["preset"], "v7_baseline")
        self.assertEqual(manifest["hardware_tag"], "t4x2")
        self.assertEqual(manifest["seed"], 17)
        self.assertEqual(manifest["split"], "grouped_cv")
        self.assertEqual(manifest["fold"], "3")
        self.assertEqual(manifest["repeat_index"], 2)
        self.assertEqual(manifest["repeat_count"], 4)
        self.assertEqual(manifest["parity_mode"], "compare")
        self.assertEqual(manifest["summary"]["reranker_name"], "qwen3_0_6b")
        self.assertIn("config_hash", manifest)

    def test_diff_prediction_rows_detects_changes(self):
        base_rows = [
            {
                "Question_ID": "1",
                "Correct_Answer": "A",
                "Doc_ID": "doc-1",
                "Page_Num": "1",
            }
        ]
        candidate_rows = [
            {
                "Question_ID": "1",
                "Correct_Answer": "B",
                "Doc_ID": "doc-2",
                "Page_Num": "3",
            }
        ]
        diff = benchmark_candidate.diff_prediction_rows(base_rows, candidate_rows)
        self.assertEqual(diff["n_questions"], 1)
        self.assertEqual(diff["answer_changed"], 1)
        self.assertEqual(diff["doc_changed"], 1)
        self.assertEqual(diff["page_changed"], 1)

    def test_parity_script_help_runs(self):
        proc = subprocess.run(
            [sys.executable, "scripts/check_kaggle_parity.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Compare the shared pipeline against the Kaggle bundle", proc.stdout)

    def test_prepare_bundle_workspace_preserves_local_runtime_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root, bundle_path = check_kaggle_parity.prepare_bundle_workspace(
                repo_root=Path(__file__).resolve().parents[1],
                temp_root=Path(tmpdir),
                preset_name="v7_baseline",
            )

            self.assertTrue(bundle_path.exists())
            self.assertEqual(bundle_path.parent.name, "notebooks")

            bundle_module = check_kaggle_parity._load_module_from_path("standalone_kaggle_test", bundle_path)
            runtime_paths = bundle_module.build_runtime_paths("local")

            self.assertEqual(workspace_root, bundle_path.parents[1])
            self.assertTrue(runtime_paths["bge_m3_dir"].exists())
            self.assertTrue(runtime_paths["mamaylm_dir"].exists())
            self.assertTrue(runtime_paths["pdf_dir"].exists())


if __name__ == "__main__":
    unittest.main()
