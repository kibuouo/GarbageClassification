import json
import tempfile
import unittest
from pathlib import Path

from experiment_utils import sha256
from model_evidence import model_evidence


class ModelEvidenceTests(unittest.TestCase):
    def test_only_matching_formal_test_checkpoint_displays_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, report = Path(folder) / "model.pth", Path(folder) / "metrics.json"
            checkpoint.write_bytes(b"current checkpoint")
            metrics = {"checkpoint_sha256": sha256(checkpoint), "split": "test", "smoke_test": False,
                       "accuracy": 0.9, "samples": 378, "dataset_sha256": "data", "manifest_sha256": "split",
                       "training_metadata": {"run_id": "run", "training_split": "train", "selection_split": "val",
                           "dataset_sha256": "data", "manifest_sha256": "split", "code_sha256": {"model.py": "source"}}}
            def write(value):
                report.write_text(json.dumps({"split": "test", "experiments": {"C": value}}))
            write(metrics)
            self.assertEqual(model_evidence(checkpoint, report)["experiment"], "C")
            for changes in ({"checkpoint_sha256": "other"}, {"smoke_test": True}, {"split": "val"},
                            {"training_metadata": {}}):
                write(dict(metrics, **changes))
                self.assertIsNone(model_evidence(checkpoint, report))
            write(metrics)
            checkpoint.write_bytes(b"replacement checkpoint")
            self.assertIsNone(model_evidence(checkpoint, report))
            report.write_text("corrupted report")
            self.assertIsNone(model_evidence(checkpoint, report))
            for malformed in ([], {"split": "test", "experiments": None},
                              {"split": "test", "experiments": {"C": None}}):
                report.write_text(json.dumps(malformed))
                self.assertIsNone(model_evidence(checkpoint, report))


if __name__ == "__main__":
    unittest.main()
