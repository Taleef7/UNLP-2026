import unittest

from notebooks.pipeline_shared import load_pipeline_presets, resolve_preset


class BenchmarkConfigTests(unittest.TestCase):
    def test_load_pipeline_presets_contains_required_presets(self):
        presets = load_pipeline_presets()
        self.assertIn("v7_baseline", presets)
        self.assertIn("v9_regression", presets)
        self.assertIn("candidate_v5_refocus_v1", presets)
        self.assertIn("candidate_dense_doc_lock_v3", presets)
        self.assertIn("candidate_dense_margin_lock_v1", presets)
        self.assertIn("candidate_v5refocus_margin_lock_v1", presets)
        self.assertIn("candidate_v5refocus_reranker4b_fair", presets)
        self.assertIn("candidate_dense_doc_lock_v3_reranker4b_fair", presets)
        self.assertIn("candidate_v5refocus_ddl_v3_reranker4b_fair", presets)
        self.assertIn("candidate_structure_chunks_v1", presets)
        self.assertIn("candidate_structure_chunks_v2_doc_guard", presets)
        self.assertIn("candidate_page_summary_selector_v1", presets)
        self.assertIn("candidate_avir_threshold_v1", presets)
        self.assertIn("candidate_stronger_reranker_v2_fair", presets)
        self.assertIn("candidate_stronger_reranker_v3_8b_fair", presets)
        self.assertIn("candidate_stronger_reranker_v3_8b_feasibility", presets)
        self.assertIn("candidate_confidence_gated_v1", presets)
        self.assertIn("candidate_hybrid_rrf_v1", presets)
        self.assertIn("candidate_extraction_blocks_v1", presets)
        self.assertIn("candidate_no_bleed_v1", presets)
        self.assertIn("candidate_evidence_first_v1", presets)

    def test_resolve_preset_exposes_expected_sections(self):
        preset = resolve_preset("v7_baseline")
        self.assertEqual(preset["name"], "v7_baseline")
        for key in [
            "extraction",
            "retrieval",
            "rerank",
            "llm",
            "page_selection",
            "confidence_routing",
            "contracts",
            "runtime",
        ]:
            self.assertIn(key, preset)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_0_6b")

    def test_resolve_preset_includes_margin_lock_configuration(self):
        preset = resolve_preset("candidate_dense_margin_lock_v1")

        self.assertEqual(preset["name"], "candidate_dense_margin_lock_v1")
        self.assertTrue(preset["page_selection"]["lock_to_dense_top_doc"])
        self.assertEqual(preset["page_selection"]["lock_dense_by_margin_threshold"], 0.25)

    def test_resolve_preset_combines_margin_lock_with_refocus_prompt(self):
        preset = resolve_preset("candidate_v5refocus_margin_lock_v1")

        self.assertEqual(preset["name"], "candidate_v5refocus_margin_lock_v1")
        self.assertEqual(preset["llm"]["prompt_variant"], "v5_refocus")
        self.assertEqual(preset["page_selection"]["lock_dense_by_margin_threshold"], 0.25)

    def test_resolve_preset_keeps_baseline_rerank_budget_for_fair_4b_test(self):
        preset = resolve_preset("candidate_stronger_reranker_v2_fair")

        self.assertEqual(preset["name"], "candidate_stronger_reranker_v2_fair")
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["rerank"]["batch_size"], 4)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_4b")
        self.assertEqual(preset["llm"]["top_k_context"], 4)
        self.assertEqual(preset["page_selection"]["strategy"], "same_doc_answer_aware")

    def test_resolve_preset_keeps_baseline_rerank_budget_for_fair_8b_test(self):
        preset = resolve_preset("candidate_stronger_reranker_v3_8b_fair")

        self.assertEqual(preset["name"], "candidate_stronger_reranker_v3_8b_fair")
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_8b", "qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["rerank"]["batch_size"], 4)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_8b")
        self.assertEqual(preset["llm"]["top_k_context"], 4)
        self.assertEqual(preset["page_selection"]["strategy"], "same_doc_answer_aware")

    def test_resolve_preset_exposes_feasibility_alias_for_8b_lane(self):
        preset = resolve_preset("candidate_stronger_reranker_v3_8b_feasibility")

        self.assertEqual(preset["name"], "candidate_stronger_reranker_v3_8b_feasibility")
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_8b", "qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_8b")

    def test_resolve_preset_combines_refocus_prompt_with_fair_4b_reranker(self):
        preset = resolve_preset("candidate_v5refocus_reranker4b_fair")

        self.assertEqual(preset["name"], "candidate_v5refocus_reranker4b_fair")
        self.assertEqual(preset["llm"]["prompt_variant"], "v5_refocus")
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_4b")

    def test_resolve_preset_combines_dense_doc_lock_v3_with_fair_4b_reranker(self):
        preset = resolve_preset("candidate_dense_doc_lock_v3_reranker4b_fair")

        self.assertEqual(preset["name"], "candidate_dense_doc_lock_v3_reranker4b_fair")
        self.assertTrue(preset["page_selection"]["lock_to_dense_top_doc"])
        self.assertEqual(preset["page_selection"]["lock_to_dense_top_doc_min_pages"], 27)
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_4b")

    def test_resolve_preset_combines_refocus_dense_doc_lock_and_fair_4b(self):
        preset = resolve_preset("candidate_v5refocus_ddl_v3_reranker4b_fair")

        self.assertEqual(preset["name"], "candidate_v5refocus_ddl_v3_reranker4b_fair")
        self.assertEqual(preset["llm"]["prompt_variant"], "v5_refocus")
        self.assertTrue(preset["page_selection"]["lock_to_dense_top_doc"])
        self.assertEqual(preset["page_selection"]["lock_to_dense_top_doc_min_pages"], 27)
        self.assertEqual(
            preset["rerank"]["model_preference"],
            ["qwen3_4b", "qwen3_0_6b", "bge"],
        )
        self.assertEqual(preset["rerank"]["max_length"], 2048)
        self.assertEqual(preset["contracts"]["require_reranker"], "qwen3_4b")


if __name__ == "__main__":
    unittest.main()
