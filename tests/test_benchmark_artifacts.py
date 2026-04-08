import csv
import json
import tempfile
import unittest
from pathlib import Path

from notebooks.pipeline_shared import write_benchmark_artifacts


class BenchmarkArtifactTests(unittest.TestCase):
    def test_write_benchmark_artifacts_emits_expected_files(self):
        predictions = [
            {
                "Question_ID": "1",
                "Correct_Answer": "A",
                "Doc_ID": "doc-1.pdf",
                "Page_Num": "2",
            }
        ]
        per_question = [
            {
                "Question_ID": "1",
                "score": 1.0,
                "answer_correct": 1.0,
                "doc_correct": 1.0,
                "page_proximity": 1.0,
            }
        ]
        summary = {
            "preset": "v7_baseline",
            "n_questions": 1,
            "composite_score": 1.0,
            "diagnostics": {"doc_rerank_skipped": 0},
        }
        timings = {"total_seconds": 1.5, "stages": {"retrieval_seconds": 0.5}}
        run_manifest = {
            "preset": "v7_baseline",
            "config_hash": "abc123",
            "hardware_tag": "cpu",
            "components": {"reranker_name": "qwen3_0_6b"},
        }
        ranking_rows = [
            {
                "Question_ID": "1",
                "Domain": "domain_1",
                "true_doc_id": "doc-1.pdf",
                "true_page_num": 2,
                "n_pages": 5,
                "pred_doc_id": "doc-1.pdf",
                "pred_page_num": 2,
                "dense_docs": [{"doc_id": "doc-1.pdf", "score": 0.9}],
                "dense_pages": [{"doc_id": "doc-1.pdf", "page_num": 2, "score": 0.9}],
                "final_docs": [{"doc_id": "doc-1.pdf", "score": 0.95}],
                "final_pages": [{"doc_id": "doc-1.pdf", "page_num": 2, "score": 0.95}],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_benchmark_artifacts(
                output_dir=output_dir,
                predictions=predictions,
                per_question_rows=per_question,
                summary=summary,
                timings=timings,
                ranking_rows=ranking_rows,
                run_manifest=run_manifest,
            )

            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "per_question.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "timings.json").exists())
            self.assertTrue((output_dir / "ranking_details.jsonl").exists())
            self.assertTrue((output_dir / "run_manifest.json").exists())

            with (output_dir / "summary.json").open() as f:
                saved_summary = json.load(f)
            self.assertEqual(saved_summary["preset"], "v7_baseline")

            with (output_dir / "predictions.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["Doc_ID"], "doc-1.pdf")

            with (output_dir / "run_manifest.json").open() as f:
                saved_manifest = json.load(f)
            self.assertEqual(saved_manifest["config_hash"], "abc123")


if __name__ == "__main__":
    unittest.main()
