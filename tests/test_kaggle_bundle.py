import importlib.util
import tempfile
import unittest
from pathlib import Path


class KaggleBundleTests(unittest.TestCase):
    def test_rendered_standalone_script_imports_without_pipeline_shared_file(self):
        from notebooks.kaggle_bundle import render_standalone_kaggle_script

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "standalone_submission.py"
            render_standalone_kaggle_script(
                preset_name="candidate_structure_chunks_v2_doc_guard",
                output_path=output_path,
            )

            spec = importlib.util.spec_from_file_location("standalone_submission", output_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            preset = module.resolve_preset("candidate_structure_chunks_v2_doc_guard")
            self.assertEqual(preset["name"], "candidate_structure_chunks_v2_doc_guard")
            self.assertTrue(hasattr(module, "run_pipeline_from_preset"))

    def test_rendered_standalone_script_executes_without___file__(self):
        from notebooks.kaggle_bundle import render_standalone_kaggle_script

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "standalone_submission.py"
            render_standalone_kaggle_script(
                preset_name="candidate_structure_chunks_v2_doc_guard",
                output_path=output_path,
            )

            source = output_path.read_text(encoding="utf-8")
            module_globals = {"__name__": "standalone_submission"}
            exec(compile(source, str(output_path), "exec"), module_globals)

            preset = module_globals["resolve_preset"]("candidate_structure_chunks_v2_doc_guard")
            self.assertEqual(preset["name"], "candidate_structure_chunks_v2_doc_guard")
            self.assertIn("NOTEBOOK_DIR", module_globals)
            self.assertIn("REPO_ROOT", module_globals)


if __name__ == "__main__":
    unittest.main()
