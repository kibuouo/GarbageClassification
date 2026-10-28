import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_utils import manifest_fingerprint, verify_data_identity
from evaluate import export_comparison


class ManifestIdentityTests(unittest.TestCase):
    def test_lf_and_crlf_have_same_identity_and_legacy_hashes_work(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            lf = b"\xef\xbb\xbfsplit,class,filename\ntrain,paper,paper1.jpg\n"
            path.write_bytes(lf)
            audit = {**manifest_fingerprint(path), "dataset_sha256": "images-and-splits"}
            self.assertEqual(audit["manifest_sha256"], hashlib.sha256(lf).hexdigest())
            for contents in (lf, lf.replace(b"\n", b"\r\n")):
                legacy = {"manifest_sha256": hashlib.sha256(contents).hexdigest(),
                          "dataset_sha256": audit["dataset_sha256"]}
                for checkout in (lf, lf.replace(b"\n", b"\r\n")):
                    path.write_bytes(checkout)
                    current = {**manifest_fingerprint(path), "dataset_sha256": audit["dataset_sha256"]}
                    self.assertEqual(current["manifest_sha256"], audit["manifest_sha256"])
                    verify_data_identity(legacy, current, path)
                    verify_data_identity(audit, current, path)

    def test_changes_and_unknown_algorithms_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            path.write_bytes(b"split,class,filename\ntrain,paper,paper1.jpg\n")
            audit = {**manifest_fingerprint(path), "dataset_sha256": "original-images"}
            variants = [dict(audit, dataset_sha256="changed-images"),
                        dict(audit, manifest_sha256="unrelated-manifest"),
                        dict(audit, manifest_hash_algorithm="unknown"),
                        {"manifest_sha256": "unrelated-manifest", "dataset_sha256": "original-images"}]
            for record in variants:
                with self.subTest(record=record), self.assertRaises(ValueError):
                    verify_data_identity(record, audit, path)
            legacy = {"manifest_sha256": hashlib.sha256(path.read_bytes().replace(b"\n", b"\r\n")).hexdigest(),
                      "dataset_sha256": "original-images"}
            path.write_bytes(path.read_bytes().replace(b"train", b"test"))
            with self.assertRaises(ValueError):
                verify_data_identity(legacy, {**manifest_fingerprint(path), "dataset_sha256": "original-images"}, path)

    def test_merging_legacy_crlf_report_preserves_original_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            manifest = root / "data/split.csv"
            manifest.write_bytes(b"split,class,filename\ntrain,paper,paper1.jpg\n")
            legacy_hash = hashlib.sha256(manifest.read_bytes().replace(b"\n", b"\r\n")).hexdigest()
            metrics = dict(manifest_fingerprint(manifest), dataset_sha256="same", smoke_test=False,
                           training_metadata={"run_id": "test", "epoch": 1, "validation_accuracy": 0.5,
                               "validation_macro_f1": 0.5, "config": {"pretrained": True, "seed": 42,
                                   "batch_size": 4, "accumulation_steps": 4}},
                           epochs_trained=1, accuracy=0.5, macro_precision=0.5, macro_recall=0.5,
                           macro_f1=0.5, samples=6, checkpoint_sha256="checkpoint")
            old = dict(metrics, manifest_sha256=legacy_hash)
            old.pop("manifest_hash_algorithm")
            (root / "test_metrics.json").write_text(json.dumps({"experiments": {"A": old}}))
            with patch("experiment_utils.ROOT", root):
                export_comparison(root, "B", metrics)
            merged = json.loads((root / "test_metrics.json").read_text())
            self.assertEqual(merged["experiments"]["A"]["legacy_manifest_sha256"], legacy_hash)
            self.assertEqual(merged["experiments"]["A"]["manifest_sha256"], metrics["manifest_sha256"])
            with patch("experiment_utils.ROOT", root), self.assertRaises(ValueError):
                export_comparison(root, "C", dict(metrics, dataset_sha256="different"))


if __name__ == "__main__":
    unittest.main()
