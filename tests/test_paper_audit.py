import tempfile
import unittest
from pathlib import Path

from notebooks.paper_audit import classify_run_artifact, render_methodology_audit
from notebooks.pipeline_shared import write_json


class PaperAuditTests(unittest.TestCase):
    def test_classify_run_artifact_marks_override_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_json(
                run_dir / "benchmark_manifest.json",
                {
                    "preset": "v7_baseline",
                    "override_json": {"llm": {"prompt_variant": "v5_refocus"}},
                    "summary": {"n_questions": 78},
                },
            )
            write_json(run_dir / "summary.json", {"composite_score": 0.875426})

            record = classify_run_artifact(run_dir)

        self.assertEqual(record["evidence_type"], "offline_override")
        self.assertEqual(record["trust_tier"], "traceable_override")
        self.assertTrue(any("override_json" in flag for flag in record["methodology_flags"]))

    def test_classify_run_artifact_marks_unfair_reranker_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_json(
                run_dir / "benchmark_manifest.json",
                {
                    "preset": "candidate_stronger_reranker_v1",
                    "override_json": None,
                    "summary": {"n_questions": 50},
                },
            )
            write_json(run_dir / "summary.json", {"composite_score": 0.655264})

            record = classify_run_artifact(run_dir)

        self.assertEqual(record["evidence_type"], "historical_confounded")
        self.assertEqual(record["fairness_status"], "not_fair")
        self.assertTrue(any("1024" in flag and "2048" in flag for flag in record["methodology_flags"]))

    def test_classify_run_artifact_places_refocus_override_in_prompt_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "v5refocus_fold1"
            run_dir.mkdir()
            write_json(
                run_dir / "benchmark_manifest.json",
                {
                    "preset": "v7_baseline",
                    "override_json": {"llm": {"prompt_variant": "v5_refocus"}},
                    "questions_path": "data/splits/fold_1_val.csv",
                    "summary": {"n_questions": 78},
                },
            )
            write_json(run_dir / "summary.json", {"composite_score": 0.875426})

            record = classify_run_artifact(run_dir)

        self.assertEqual(record["block"], "prompt_answering")
        self.assertEqual(record["split"], "fold1")

    def test_classify_run_artifact_marks_fair_4b_reranker_as_trusted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_json(
                run_dir / "benchmark_manifest.json",
                {
                    "preset": "candidate_stronger_reranker_v2_fair",
                    "override_json": None,
                    "summary": {"n_questions": 461},
                },
            )
            write_json(run_dir / "summary.json", {"composite_score": 0.886644})

            record = classify_run_artifact(run_dir)

        self.assertEqual(record["evidence_type"], "offline_named_preset")
        self.assertEqual(record["fairness_status"], "fair")
        self.assertEqual(record["trust_tier"], "trusted_offline")

    def test_classify_run_artifact_marks_8b_lane_as_feasibility_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            write_json(
                run_dir / "benchmark_manifest.json",
                {
                    "preset": "candidate_stronger_reranker_v3_8b_feasibility",
                    "override_json": None,
                    "summary": {"n_questions": 1},
                },
            )
            write_json(run_dir / "summary.json", {"composite_score": 1.0})

            record = classify_run_artifact(run_dir)

        self.assertEqual(record["evidence_type"], "feasibility_only")
        self.assertEqual(record["fairness_status"], "single_gpu_feasibility")
        self.assertEqual(record["trust_tier"], "feasibility_only")

    def test_render_methodology_audit_includes_required_flags(self):
        text = render_methodology_audit()

        self.assertIn("candidate_stronger_reranker_v1", text)
        self.assertIn("1024", text)
        self.assertIn("2048", text)
        self.assertIn("v5refocus_*", text)
        self.assertIn("candidate_dense_margin_lock_v1", text)


if __name__ == "__main__":
    unittest.main()
