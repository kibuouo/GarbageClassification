"""Integration checks on disposable images; no real dataset or weights needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from garbage_classifier.common import CLASSES
from garbage_classifier.comparison import compare
from garbage_classifier.config import ExperimentConfig, load_config
from garbage_classifier.data import validate_split
from garbage_classifier.engine import train
from garbage_classifier.evaluation import evaluate
from garbage_classifier.inference import Predictor
from models.resnet18_model import build_resnet18, set_trainable, training_mode
from prepare_data import prepare


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_raw(self, root):
        rng = np.random.default_rng(123)
        raw = root / "raw"
        for label in CLASSES:
            (raw / label).mkdir(parents=True)
            for index in range(4):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(raw / label / f"{index}.png")
        return raw

    def test_fixed_split_and_changed_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.make_raw(root)
            _, identity = prepare(raw, root / "data", root / "reports")
            _, repeated = prepare(raw, root / "repeat", root / "reports2")
            self.assertEqual(identity, repeated)
            self.assertEqual(identity, validate_split(root / "data"))
            with self.assertRaises(FileExistsError):
                prepare(raw, root / "data", root / "reports")
            image = next((root / "data" / "train").rglob("*.png"))
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "数据缺失或被改动"):
                validate_split(root / "data")

    def test_duplicates_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.make_raw(root)
            (raw / "glass" / "0.png").write_bytes((raw / "cardboard" / "0.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "重复"):
                prepare(raw, root / "data", root / "reports")
            self.assertFalse((root / "data").exists())

    def test_freezing_gradients_and_batchnorm_statistics(self):
        model = build_resnet18(6)
        set_trainable(model, "frozen")
        training_mode(model, "frozen")
        original = model.bn1.running_mean.clone()
        model(torch.randn(2, 3, 32, 32)).sum().backward()
        self.assertTrue(torch.equal(original, model.bn1.running_mean))
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertIsNone(model.layer4[0].conv1.weight.grad)
        set_trainable(model, "finetune")
        training_mode(model, "finetune")
        self.assertFalse(model.bn1.training)
        self.assertTrue(model.layer4[0].bn1.training)
        self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
        self.assertFalse(model.layer3[0].conv1.weight.requires_grad)

    def test_train_evaluate_predict_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self.make_raw(root)
            prepare(raw, root / "data", root / "reports")
            config = ExperimentConfig(name="smoke", strategy="scratch", data_dir=str(root / "data"),
                                      epochs=1, batch_size=6, image_size=32, device="cpu")
            checkpoint = train(config, output_root=root)
            output = root / "results" / "smoke"
            metrics = evaluate(checkpoint, root / "data", output, batch_size=6, device="cpu")
            self.assertEqual(metrics["test_samples"], 6)
            self.assertTrue((output / "accuracy_curve.png").is_file())
            self.assertTrue((output / "misclassified.csv").is_file())
            predictor = Predictor(checkpoint, "cpu")
            with Image.open(raw / "paper" / "0.png") as image:
                result = predictor.predict(image)
            self.assertIn(result["label"], CLASSES)
            self.assertEqual(len(result["top3"]), 3)
            self.assertTrue(0 <= result["confidence"] <= 1)
            rows = compare([output], root / "comparison")
            self.assertEqual(rows[0]["macro_f1"], metrics["macro_f1"])

    def test_configs_and_missing_model(self):
        for path in ("configs/A_scratch.json", "configs/B_frozen.json", "configs/C_finetune.json"):
            load_config(path)
        with self.assertRaises(ValueError):
            ExperimentConfig(strategy="finetune", epochs=2, warmup_epochs=5).validate()
        with self.assertRaises(FileNotFoundError):
            Predictor(Path("/nonexistent/classifier.pth"), "cpu")


if __name__ == "__main__":
    unittest.main()
