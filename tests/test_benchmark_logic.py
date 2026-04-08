import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from notebooks.pipeline_shared import (
    PipelineRunner,
    should_lock_to_dense_top_doc,
    ensure_fitz_available,
    ensure_llama_cpp_available,
    first_existing_path,
    compute_margin,
    reciprocal_rank_fusion,
    choose_n_passes,
    compute_ir_metrics,
    compare_benchmark_dirs,
    evaluate_predictions,
    maybe_enable_fast_mode,
    resolve_preset,
    should_skip_doc_rerank,
    write_csv,
)


class BenchmarkLogicTests(unittest.TestCase):
    def test_evaluate_predictions_matches_competition_metric(self):
        predictions = [
            {
                "Question_ID": "1",
                "Correct_Answer": "A",
                "Doc_ID": "doc-1.pdf",
                "Page_Num": "3",
            },
            {
                "Question_ID": "2",
                "Correct_Answer": "C",
                "Doc_ID": "doc-2.pdf",
                "Page_Num": "8",
            },
        ]
        ground_truth = [
            {
                "Question_ID": "1",
                "Domain": "domain_1",
                "N_Pages": "10",
                "Correct_Answer": "A",
                "Doc_ID": "doc-1.pdf",
                "Page_Num": "5",
            },
            {
                "Question_ID": "2",
                "Domain": "domain_2",
                "N_Pages": "20",
                "Correct_Answer": "B",
                "Doc_ID": "doc-9.pdf",
                "Page_Num": "12",
            },
        ]

        results = evaluate_predictions(predictions, ground_truth)

        self.assertEqual(results["n_questions"], 2)
        self.assertAlmostEqual(results["answer_accuracy"], 0.5)
        self.assertAlmostEqual(results["doc_accuracy"], 0.5)
        self.assertAlmostEqual(results["page_proximity"], 0.4)
        self.assertAlmostEqual(results["composite_score"], 0.475)

    def test_should_skip_doc_rerank_for_easy_dense_retrieval_case(self):
        candidates = [
            ("doc-1.pdf", 2, 0.91),
            ("doc-1.pdf", 4, 0.89),
            ("doc-1.pdf", 5, 0.88),
            ("doc-2.pdf", 1, 0.80),
        ]
        rerank_cfg = {"doc_skip_topn": 3, "doc_skip_margin": 0.035}

        self.assertTrue(should_skip_doc_rerank(candidates, rerank_cfg))

    def test_choose_n_passes_escalates_for_ambiguous_doc_ranking(self):
        runtime_cfg = {"fast_mode": False, "passes": 1}
        llm_cfg = {"hard_passes": 3, "vote_margin": 0.08}
        page_counts = {"doc-1.pdf": 15}
        doc_ranking = [("doc-1.pdf", 0.61), ("doc-2.pdf", 0.57)]

        n_passes = choose_n_passes(doc_ranking, "doc-1.pdf", runtime_cfg, llm_cfg, page_counts)

        self.assertEqual(n_passes, 3)

    def test_choose_n_passes_uses_confidence_gate_for_easy_case(self):
        runtime_cfg = {"fast_mode": False, "passes": 3}
        llm_cfg = {
            "base_passes": 3,
            "hard_passes": 3,
            "easy_passes": 1,
            "confidence_gate_enabled": True,
            "easy_doc_margin": 0.12,
            "easy_page_margin": 0.05,
            "long_doc_hard_pages": 80,
            "vote_margin": 0.08,
        }
        page_counts = {"doc-1.pdf": 12}
        doc_ranking = [("doc-1.pdf", 0.73), ("doc-2.pdf", 0.55)]

        n_passes = choose_n_passes(
            doc_ranking,
            "doc-1.pdf",
            runtime_cfg,
            llm_cfg,
            page_counts,
            page_margin=0.11,
        )

        self.assertEqual(n_passes, 1)

    def test_maybe_enable_fast_mode_reduces_runtime_budget(self):
        runtime_cfg = {
            "fast_mode": False,
            "passes": 3,
            "top_k_context": 4,
            "max_chars_per_page": 3200,
            "enable_page_stage2": True,
        }
        runtime_limits = {
            "time_budget_hours": 8.5,
            "fast_mode_warmup_q": 25,
            "fast_context_pages_cap": 3,
            "fast_max_chars_per_page_cap": 2500,
        }

        changed = maybe_enable_fast_mode(
            runtime_cfg=runtime_cfg,
            runtime_limits=runtime_limits,
            done=25,
            total=461,
            elapsed_total_seconds=7 * 3600,
            elapsed_mcq_seconds=6 * 3600,
        )

        self.assertTrue(changed)
        self.assertTrue(runtime_cfg["fast_mode"])
        self.assertEqual(runtime_cfg["passes"], 1)
        self.assertEqual(runtime_cfg["top_k_context"], 3)
        self.assertEqual(runtime_cfg["max_chars_per_page"], 2500)
        self.assertFalse(runtime_cfg["enable_page_stage2"])

    def test_compare_benchmark_dirs_reports_rates_and_domain_deltas(self):
        import tempfile
        from pathlib import Path

        base_rows = [
            {
                "Question_ID": "0",
                "Domain": "domain_1",
                "pred_answer": "A",
                "pred_doc_id": "doc-1.pdf",
                "pred_page_num": "1",
                "score": "1.0",
            },
            {
                "Question_ID": "1",
                "Domain": "domain_2",
                "pred_answer": "B",
                "pred_doc_id": "doc-2.pdf",
                "pred_page_num": "2",
                "score": "0.5",
            },
        ]
        candidate_rows = [
            {
                "Question_ID": "0",
                "Domain": "domain_1",
                "pred_answer": "A",
                "pred_doc_id": "doc-3.pdf",
                "pred_page_num": "4",
                "score": "0.0",
            },
            {
                "Question_ID": "1",
                "Domain": "domain_2",
                "pred_answer": "B",
                "pred_doc_id": "doc-2.pdf",
                "pred_page_num": "2",
                "score": "0.75",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_dir = tmp / "base"
            candidate_dir = tmp / "candidate"
            base_dir.mkdir()
            candidate_dir.mkdir()
            write_csv(base_dir / "per_question.csv", base_rows)
            write_csv(candidate_dir / "per_question.csv", candidate_rows)

            summary, diff_rows = compare_benchmark_dirs(base_dir, candidate_dir)

        self.assertEqual(summary["counts"]["doc_changed"], 1)
        self.assertAlmostEqual(summary["top_1_doc_flip_rate"], 0.5)
        self.assertAlmostEqual(summary["page_change_rate"], 0.5)
        self.assertAlmostEqual(summary["answer_change_rate"], 0.0)
        self.assertAlmostEqual(summary["per_domain_score_delta"]["domain_1"], -1.0)
        self.assertAlmostEqual(summary["per_domain_score_delta"]["domain_2"], 0.25)
        self.assertEqual(diff_rows[0]["doc_changed"], "True")

    def test_structure_chunk_builder_preserves_page_marker(self):
        preset = resolve_preset("candidate_structure_chunks_v1")
        runner = PipelineRunner(preset=preset, n_questions=0)
        chunks = runner._build_structure_chunks_for_page(
            3,
            "РОЗДІЛ 1\nПерший абзац тексту.\nДругий абзац тексту.\nВИСНОВОК:\nЩе трохи тексту.",
        )

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.startswith("[Сторінка 3]") for chunk in chunks))

    def test_pipeline_runner_initializes_easy_pass_diagnostic(self):
        preset = resolve_preset("candidate_confidence_gated_v1")
        runner = PipelineRunner(preset=preset, n_questions=0)

        self.assertIn("easy_pass_count", runner.runtime_diagnostics)
        self.assertEqual(runner.runtime_diagnostics["easy_pass_count"], 0)

    def test_structure_doc_guard_prefers_multi_page_support_over_single_spike(self):
        preset = resolve_preset("candidate_structure_chunks_v2_doc_guard")
        runner = PipelineRunner(preset=preset, n_questions=0)
        ranked_pages = [
            ("doc-a.pdf", 1, 0.90),
            ("doc-b.pdf", 1, 0.80),
            ("doc-b.pdf", 2, 0.70),
        ]

        guarded = runner.build_doc_ranking_from_pages(ranked_pages, use_structure_doc_guard=True)
        unguarded = runner.build_doc_ranking_from_pages(ranked_pages, use_structure_doc_guard=False)

        self.assertEqual(unguarded[0][0], "doc-a.pdf")
        self.assertEqual(guarded[0][0], "doc-b.pdf")

    def test_first_existing_path_returns_first_existing_candidate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing"
            existing = root / "existing"
            existing.mkdir()

            resolved = first_existing_path([missing, existing])

        self.assertEqual(resolved, existing)

    def test_ensure_fitz_available_installs_from_wheel_when_import_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            wheels_dir = Path(tmpdir)
            wheel_path = wheels_dir / "pymupdf-1.26.0-cp39-abi3-manylinux2014_x86_64.whl"
            wheel_path.write_text("placeholder", encoding="utf-8")
            fitz_module = object()

            with mock.patch("notebooks.pipeline_shared.importlib.import_module") as import_module:
                with mock.patch("notebooks.pipeline_shared.subprocess.check_call") as check_call:
                    import_module.side_effect = [ModuleNotFoundError("No module named 'fitz'"), fitz_module]

                    resolved = ensure_fitz_available(wheels_dir)

            self.assertIs(resolved, fitz_module)
            check_call.assert_called_once()
            call_args = check_call.call_args.args[0]
            self.assertIn(str(wheel_path), call_args)
            self.assertIn("--no-index", call_args)

    def test_ensure_llama_cpp_available_installs_from_wheel_when_import_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            wheels_dir = Path(tmpdir)
            wheel_path = wheels_dir / "llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl"
            wheel_path.write_text("placeholder", encoding="utf-8")
            llama_cpp_module = object()

            with mock.patch("notebooks.pipeline_shared.importlib.import_module") as import_module:
                with mock.patch("notebooks.pipeline_shared.subprocess.check_call") as check_call:
                    import_module.side_effect = [ModuleNotFoundError("No module named 'llama_cpp'"), llama_cpp_module]

                    resolved = ensure_llama_cpp_available(wheels_dir)

            self.assertIs(resolved, llama_cpp_module)
            check_call.assert_called_once()
            call_args = check_call.call_args.args[0]
            self.assertIn(str(wheel_path), call_args)
            self.assertIn("--no-index", call_args)
            self.assertIn(f"--find-links={wheels_dir}", call_args)
            self.assertNotIn("--no-deps", call_args)

    def test_ensure_llama_cpp_available_prefers_matching_python_wheel(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            wheels_dir = Path(tmpdir)
            cp39_wheel = wheels_dir / "llama_cpp_python-0.3.16-cp39-cp39-linux_x86_64.whl"
            cp312_wheel = wheels_dir / "llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl"
            cp39_wheel.write_text("placeholder", encoding="utf-8")
            cp312_wheel.write_text("placeholder", encoding="utf-8")
            llama_cpp_module = object()

            with mock.patch("notebooks.pipeline_shared.sys.version_info", SimpleNamespace(major=3, minor=12)):
                with mock.patch("notebooks.pipeline_shared.importlib.import_module") as import_module:
                    with mock.patch("notebooks.pipeline_shared.subprocess.check_call") as check_call:
                        import_module.side_effect = [ModuleNotFoundError("No module named 'llama_cpp'"), llama_cpp_module]

                        resolved = ensure_llama_cpp_available(wheels_dir)

            self.assertIs(resolved, llama_cpp_module)
            check_call.assert_called_once()
            call_args = check_call.call_args.args[0]
            self.assertIn(str(cp312_wheel), call_args)
            self.assertNotIn(str(cp39_wheel), call_args)

    def test_ensure_llama_cpp_available_retries_next_wheel_after_install_failure(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            wheels_dir = Path(tmpdir)
            first_wheel = wheels_dir / "llama_cpp_python-0.3.17-cp312-cp312-linux_x86_64.whl"
            second_wheel = wheels_dir / "llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl"
            first_wheel.write_text("placeholder", encoding="utf-8")
            second_wheel.write_text("placeholder", encoding="utf-8")
            llama_cpp_module = object()

            with mock.patch("notebooks.pipeline_shared.sys.version_info", SimpleNamespace(major=3, minor=12)):
                with mock.patch("notebooks.pipeline_shared.importlib.import_module") as import_module:
                    with mock.patch("notebooks.pipeline_shared.subprocess.check_call") as check_call:
                        import_module.side_effect = [
                            ModuleNotFoundError("No module named 'llama_cpp'"),
                            llama_cpp_module,
                        ]
                        check_call.side_effect = [
                            subprocess.CalledProcessError(1, ["pip", "install", str(first_wheel)]),
                            None,
                        ]

                        resolved = ensure_llama_cpp_available(wheels_dir)

            self.assertIs(resolved, llama_cpp_module)
            self.assertEqual(check_call.call_count, 2)
            first_call = check_call.call_args_list[0].args[0]
            second_call = check_call.call_args_list[1].args[0]
            self.assertIn(str(first_wheel), first_call)
            self.assertIn(str(second_wheel), second_call)

    def test_compute_ir_metrics_reports_recall_mrr_ndcg_and_page_proximity(self):
        ranking_rows = [
            {
                "Question_ID": "1",
                "Domain": "domain_1",
                "true_doc_id": "doc-1.pdf",
                "true_page_num": 3,
                "n_pages": 10,
                "dense_docs": [{"doc_id": "doc-1.pdf", "score": 0.9}],
                "dense_pages": [{"doc_id": "doc-1.pdf", "page_num": 3, "score": 0.9}],
                "final_docs": [{"doc_id": "doc-1.pdf", "score": 0.95}],
                "final_pages": [{"doc_id": "doc-1.pdf", "page_num": 4, "score": 0.95}],
            },
            {
                "Question_ID": "2",
                "Domain": "domain_2",
                "true_doc_id": "doc-2.pdf",
                "true_page_num": 5,
                "n_pages": 20,
                "dense_docs": [{"doc_id": "doc-x.pdf", "score": 0.8}, {"doc_id": "doc-2.pdf", "score": 0.7}],
                "dense_pages": [{"doc_id": "doc-x.pdf", "page_num": 1, "score": 0.8}, {"doc_id": "doc-2.pdf", "page_num": 7, "score": 0.7}],
                "final_docs": [{"doc_id": "doc-2.pdf", "score": 0.85}],
                "final_pages": [{"doc_id": "doc-2.pdf", "page_num": 5, "score": 0.85}],
            },
        ]

        metrics = compute_ir_metrics(ranking_rows, ks=(1, 2))

        self.assertAlmostEqual(metrics["stages"]["dense_doc"]["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["stages"]["dense_doc"]["recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["stages"]["dense_doc"]["mrr"], 0.75)
        self.assertAlmostEqual(metrics["stages"]["final_doc"]["recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["stages"]["dense_page"]["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["stages"]["dense_page"]["recall_at_2"], 0.5)
        self.assertAlmostEqual(metrics["stages"]["dense_page"]["page_proximity_at_2"], 0.95)
        self.assertAlmostEqual(metrics["stages"]["final_page"]["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["stages"]["final_page"]["page_proximity_at_1"], 0.95)

    def test_compute_margin_returns_gap_between_top_two_scores(self):
        self.assertAlmostEqual(compute_margin([("doc-1.pdf", 0.9), ("doc-2.pdf", 0.7)]), 0.2)
        self.assertIsNone(compute_margin([("doc-1.pdf", 0.9)]))

    def test_should_lock_to_dense_top_doc_uses_margin_threshold(self):
        page_sel_cfg = {
            "lock_to_dense_top_doc": True,
            "lock_dense_by_margin_threshold": 0.25,
        }
        dense_candidates = [("doc-1.pdf", 2, 0.90), ("doc-2.pdf", 1, 0.60)]

        self.assertTrue(
            should_lock_to_dense_top_doc(
                page_sel_cfg=page_sel_cfg,
                dense_candidates=dense_candidates,
                doc_meta={},
            )
        )

        dense_candidates = [("doc-1.pdf", 2, 0.90), ("doc-2.pdf", 1, 0.72)]
        self.assertFalse(
            should_lock_to_dense_top_doc(
                page_sel_cfg=page_sel_cfg,
                dense_candidates=dense_candidates,
                doc_meta={},
            )
        )

    def test_should_lock_to_dense_top_doc_requires_all_active_guards_to_pass(self):
        page_sel_cfg = {
            "lock_to_dense_top_doc": True,
            "lock_to_dense_top_doc_min_pages": 27,
            "lock_dense_by_margin_threshold": 0.25,
        }
        dense_candidates = [("doc-1.pdf", 2, 0.90), ("doc-2.pdf", 1, 0.60)]

        self.assertFalse(
            should_lock_to_dense_top_doc(
                page_sel_cfg=page_sel_cfg,
                dense_candidates=dense_candidates,
                doc_meta={"doc-1.pdf": {"n_pages": 26}},
            )
        )
        self.assertTrue(
            should_lock_to_dense_top_doc(
                page_sel_cfg=page_sel_cfg,
                dense_candidates=dense_candidates,
                doc_meta={"doc-1.pdf": {"n_pages": 30}},
            )
        )

    def test_reciprocal_rank_fusion_combines_dense_and_sparse_rankings(self):
        fused = reciprocal_rank_fusion(
            [
                [("doc-a.pdf", 1, 0.9), ("doc-b.pdf", 2, 0.8)],
                [("doc-b.pdf", 2, 11.0), ("doc-a.pdf", 1, 10.0)],
            ],
            k=10,
        )

        self.assertEqual({(row[0], row[1]) for row in fused[:2]}, {("doc-a.pdf", 1), ("doc-b.pdf", 2)})
        self.assertAlmostEqual(fused[0][2], fused[1][2])

    def test_select_page_within_doc_can_disable_answer_conditioning(self):
        preset = resolve_preset("candidate_evidence_first_v1")
        runner = PipelineRunner(preset=preset, n_questions=0)
        runner.page_embs_dict = {
            ("doc-1.pdf", 1): np.asarray([0.1], dtype="float32"),
            ("doc-1.pdf", 2): np.asarray([0.9], dtype="float32"),
        }
        runner.doc_pages_dict["doc-1.pdf"] = [1, 2]
        runner.runtime_cfg = runner.initial_runtime_cfg()
        runner.encode_query = lambda text, max_length=256: np.asarray([1.0], dtype="float32")
        runner.encode_augmented_query = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("answer-conditioned embedding should not be used")
        )

        best_candidate, page_diag = runner.select_page_within_doc(
            question_row={"Question": "Що трапилось?", "A": "Перша", "B": "Друга"},
            pred_answer="A",
            ordered_pages=[("doc-1.pdf", 1, 0.7), ("doc-1.pdf", 2, 0.6)],
            doc_ranking=[("doc-1.pdf", 0.8)],
        )

        self.assertEqual(best_candidate[:2], ("doc-1.pdf", 2))
        self.assertEqual(page_diag["answer_conditioning"], "disabled")

    def test_extract_all_pages_can_disable_neighbor_bleed(self):
        import tempfile

        class FakePage:
            def __init__(self, text):
                self.text = text

            def get_text(self, mode="text"):
                return self.text

            def find_tables(self):
                return SimpleNamespace(tables=[])

        class FakeDoc(list):
            def close(self):
                return None

        class FakeFitz:
            def open(self, _path):
                return FakeDoc([FakePage("ALPHA"), FakePage("BRAVO")])

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            (pdf_dir / "domain_1").mkdir()
            (pdf_dir / "domain_1" / "doc-1.pdf").write_text("stub", encoding="utf-8")
            preset = resolve_preset("candidate_no_bleed_v1")
            runner = PipelineRunner(preset=preset, n_questions=0)
            runner.paths["pdf_dir"] = pdf_dir

            with mock.patch("notebooks.pipeline_shared.ensure_fitz_available", return_value=FakeFitz()):
                runner.extract_all_pages()

        self.assertEqual(runner.page_text_index[("doc-1.pdf", 1)], "ALPHA")
        self.assertEqual(runner.page_text_index[("doc-1.pdf", 2)], "BRAVO")


if __name__ == "__main__":
    unittest.main()
