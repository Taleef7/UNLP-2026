import tempfile
import unittest
from pathlib import Path

from scripts.run_paper_followup_suite import (
    FEASIBILITY_PRESET,
    PAPER_RUNS_ROOT,
    RunSpec,
    assess_gate_stability,
    benchmark_command,
    build_run_spec,
    diff_command,
    render_run_notes,
)


class PaperFollowupSuiteTests(unittest.TestCase):
    def test_build_run_spec_uses_expected_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = build_run_spec(
                preset="candidate_dense_margin_lock_v1",
                split_key="fold0",
                output_root=Path(tmpdir),
                stage="mandatory_dense_margin",
                purpose="test",
            )

        self.assertEqual(spec.split_key, "fold0")
        self.assertTrue(str(spec.questions_path).endswith("data/splits/fold_0_val.csv"))
        self.assertTrue(str(spec.baseline_dir).endswith("outputs/benchmarks/v7_baseline_fold0"))
        self.assertTrue(str(spec.output_dir).endswith("candidate_dense_margin_lock_v1_fold0"))

    def test_benchmark_and_diff_commands_include_expected_arguments(self):
        spec = RunSpec(
            preset=FEASIBILITY_PRESET,
            split_key="fold0",
            output_dir=PAPER_RUNS_ROOT / "candidate_stronger_reranker_v3_8b_feasibility_fold0",
            baseline_dir=Path("outputs/benchmarks/v7_baseline_fold0"),
            questions_path=Path("data/splits/fold_0_val.csv"),
            n_questions=1,
            override_json={"rerank": {"batch_size": 2}},
            stage="8b_feasibility",
            purpose="test",
        )

        benchmark = benchmark_command(spec, "a30-paper-suite")
        diff = diff_command(spec)

        self.assertIn("--preset", benchmark)
        self.assertIn(FEASIBILITY_PRESET, benchmark)
        self.assertIn("--override-json", benchmark)
        self.assertIn("--n-questions", benchmark)
        self.assertIn("--candidate-dir", diff)
        self.assertIn(str(spec.output_dir), diff)

    def test_assess_gate_stability_requires_reasonable_deltas(self):
        stable = assess_gate_stability(
            {
                "fold0": {"avg_score_delta": 0.004},
                "fold1": {"avg_score_delta": 0.002},
                "lockbox": {"avg_score_delta": -0.003},
            },
            mean_threshold=-0.001,
            min_threshold=-0.01,
        )
        unstable = assess_gate_stability(
            {
                "fold0": {"avg_score_delta": 0.004},
                "fold1": {"avg_score_delta": -0.012},
                "lockbox": {"avg_score_delta": -0.004},
            },
            mean_threshold=-0.001,
            min_threshold=-0.01,
        )

        self.assertTrue(stable["stable"])
        self.assertFalse(unstable["stable"])
        self.assertEqual(unstable["reason"], "gate_threshold_failed")

    def test_render_run_notes_includes_diff_fields(self):
        spec = RunSpec(
            preset="candidate_dense_margin_lock_v1",
            split_key="fold0",
            output_dir=Path("outputs/paper_ablation_runs/candidate_dense_margin_lock_v1_fold0"),
            baseline_dir=Path("outputs/benchmarks/v7_baseline_fold0"),
            questions_path=Path("data/splits/fold_0_val.csv"),
            stage="mandatory_dense_margin",
            purpose="test",
        )
        text = render_run_notes(
            spec,
            {
                "composite_score": 0.88,
                "answer_accuracy": 0.85,
                "doc_accuracy": 0.9,
                "page_proximity": 0.77,
                "reranker_name": "qwen3_0_6b",
            },
            {
                "avg_score_delta": 0.01,
                "top_1_doc_flip_rate": 0.0,
                "page_change_rate": 0.02,
                "answer_change_rate": 0.01,
                "counts": {"improved": 3, "regressed": 1},
            },
        )

        self.assertIn("Diff vs v7 Baseline", text)
        self.assertIn("avg_score_delta", text.replace("Average score delta", "avg_score_delta"))
        self.assertIn("candidate_dense_margin_lock_v1", text)


if __name__ == "__main__":
    unittest.main()
