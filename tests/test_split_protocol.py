import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "00_create_splits.py"
QUESTIONS_PATH = ROOT / "data" / "dev_questions.csv"


def load_split_module():
    spec = importlib.util.spec_from_file_location("create_splits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SplitProtocolTest(unittest.TestCase):
    def test_build_split_protocol_groups_docs_and_is_stratified(self):
        module = load_split_module()
        questions = load_questions()

        protocol = module.build_split_protocol(questions, seed=42, n_folds=5, lockbox_ratio=0.2)

        self.assertEqual(len(protocol["folds"]), 5)
        all_val_docs = []
        fold_sizes = []
        for fold in protocol["folds"]:
            train_doc_ids = {row["Doc_ID"] for row in fold["train"]}
            val_doc_ids = {row["Doc_ID"] for row in fold["val"]}
            self.assertGreater(len(val_doc_ids), 0)
            self.assertTrue(train_doc_ids.isdisjoint(val_doc_ids))
            self.assertTrue(all(len({row["Doc_ID"] for row in rows}) == 1 for rows in fold["train_by_doc"].values()))
            self.assertTrue(all(len({row["Doc_ID"] for row in rows}) == 1 for rows in fold["val_by_doc"].values()))
            all_val_docs.extend(sorted(val_doc_ids))
            fold_sizes.append(len(fold["val_by_doc"]))

        self.assertEqual(len(all_val_docs), len(set(all_val_docs)))
        self.assertEqual(
            sum(fold_sizes),
            len(protocol["docs"]) - len(protocol["lockbox"]["docs"]),
        )

        lockbox_doc_ids = {doc["doc_id"] for doc in protocol["lockbox"]["docs"]}
        fold_doc_ids = {doc["doc_id"] for fold in protocol["folds"] for doc in fold["docs"]}
        self.assertTrue(lockbox_doc_ids.isdisjoint(fold_doc_ids))
        self.assertEqual(len(lockbox_doc_ids), len(protocol["lockbox"]["docs"]))

    def test_write_split_protocol_emits_manifests(self):
        module = load_split_module()
        questions = load_questions()
        protocol = module.build_split_protocol(questions, seed=42, n_folds=5, lockbox_ratio=0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            module.write_split_protocol(protocol, out_dir)

            self.assertTrue((out_dir / "split_info.json").exists())
            self.assertTrue((out_dir / "folds.json").exists())
            self.assertTrue((out_dir / "lockbox.json").exists())
            self.assertTrue((out_dir / "train.csv").exists())
            self.assertTrue((out_dir / "val.csv").exists())
            self.assertTrue((out_dir / "lockbox.csv").exists())
            self.assertTrue((out_dir / "fold_0_train.csv").exists())
            self.assertTrue((out_dir / "fold_4_val.csv").exists())

            with open(out_dir / "split_info.json", "r", encoding="utf-8") as handle:
                split_info = json.load(handle)
            with open(out_dir / "folds.json", "r", encoding="utf-8") as handle:
                folds_manifest = json.load(handle)
            with open(out_dir / "lockbox.json", "r", encoding="utf-8") as handle:
                lockbox_manifest = json.load(handle)

            self.assertEqual(split_info["protocol"]["n_folds"], 5)
            self.assertEqual(len(folds_manifest["folds"]), 5)
            self.assertGreater(len(lockbox_manifest["doc_ids"]), 0)
            self.assertIn("standard", split_info)
            self.assertIn("lodo", split_info)


if __name__ == "__main__":
    unittest.main()
