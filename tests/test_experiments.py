import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_utils import ROOT, load_config, validate_config
from model import build_model
from train import run_epoch, save_checkpoint, set_stage, train_experiment


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Linear(2, 2)
        self.fc = nn.Linear(2, 6)

    def freeze_features(self, freeze=True):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not freeze or name.startswith("fc."))


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_accumulation_matches_full_batch_including_short_tail(self):
        torch.manual_seed(7)
        full = nn.Linear(2, 6)
        micro = copy.deepcopy(full)
        data = TensorDataset(torch.randn(5, 2), torch.tensor([0, 1, 2, 3, 4]))
        run_epoch(full, DataLoader(data, batch_size=5), "cpu", torch.optim.SGD(full.parameters(), lr=0.1))
        run_epoch(micro, DataLoader(data, batch_size=2), "cpu", torch.optim.SGD(micro.parameters(), lr=0.1), accumulation_steps=3)
        for left, right in zip(full.parameters(), micro.parameters()):
            self.assertTrue(torch.allclose(left, right, atol=1e-7))

    def test_c_unfreezes_only_layer4_after_warmup(self):
        config = load_config(ROOT / "experiments/C.json")
        model = build_model()
        self.assertEqual(set_stage(model, config, 3), "head")
        self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("fc.")))
        self.assertEqual(set_stage(model, config, 4), "layer4")
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith(("fc.", "layer4.")))

    def test_failed_checkpoint_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pth"
            save_checkpoint(path, {"epoch": 1})
            original = path.read_bytes()
            def interrupted(value, file):
                file.write(b"incomplete")
                raise OSError("simulated interrupted write")
            with patch("train.torch.save", side_effect=interrupted):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"epoch": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 1)

    def test_invalid_warmup_is_rejected(self):
        config = load_config(ROOT / "experiments/C.json")
        config["warmup_epochs"] = config["epochs"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_final_checkpoint_repairs_interrupted_best_write(self):
        config = load_config(ROOT / "experiments/A.json")
        config.update(epochs=1, pretrained=False, max_train_samples=6, max_val_samples=6)
        audit = {"manifest_sha256": "fixture", "dataset_sha256": "fixture", "counts": {}}
        data = TensorDataset(torch.zeros(6, 2), torch.arange(6))
        metrics = (0.5, {"accuracy": 0.5, "macro_f1": 0.5})
        actual_save = save_checkpoint
        def interrupt_best(path, value):
            if path.name == "best.pth":
                raise OSError("simulated interruption after last checkpoint")
            return actual_save(path, value)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            with patch("train.build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch("train.ManifestDataset", return_value=data), patch("train.run_epoch", return_value=metrics), \
                 patch("train.save_checkpoint", side_effect=interrupt_best):
                with self.assertRaises(OSError):
                    train_experiment(config, run, audit, "cpu")
            self.assertTrue((run / "last.pth").exists())
            self.assertEqual(json.loads((run / "status.json").read_text())["last_epoch"], 1)
            with patch("train.build_model", side_effect=lambda *a, **k: TinyModel()), patch("train.run_epoch") as epoch:
                train_experiment(config, run, audit, "cpu", resume=True)
                epoch.assert_not_called()
            self.assertTrue((run / "best.pth").exists())
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "completed")
            self.assertEqual(len(json.loads((run / "history.json").read_text())), 1)

    def test_early_stopping_uses_validation_and_keeps_best_epoch(self):
        config = load_config(ROOT / "experiments/B.json")
        config.update(epochs=5, patience=2, pretrained=False, max_train_samples=6, max_val_samples=6)
        data = TensorDataset(torch.zeros(6, 2), torch.arange(6))
        audit = {"manifest_sha256": "fixture", "dataset_sha256": "fixture", "counts": {}}
        results = []
        for accuracy in [0.8, 0.7, 0.6]:
            results.extend([(0.5, {"accuracy": 0.9, "macro_f1": 0.9}),
                            (0.5, {"accuracy": accuracy, "macro_f1": accuracy})])
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            with patch("train.build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch("train.ManifestDataset", return_value=data), patch("train.run_epoch", side_effect=results):
                train_experiment(config, run, audit, "cpu")
            status = json.loads((run / "status.json").read_text())
            self.assertEqual((status["status"], status["last_epoch"], status["best_epoch"]), ("early_stopped", 3, 1))


if __name__ == "__main__":
    unittest.main()
