import unittest
import torch
from torch import nn
from model import build_model
from train import set_stage


class EfficientNetTests(unittest.TestCase):
    def test_head_cache_and_unfreezing(self):
        torch.set_num_threads(2)
        model = build_model(6, architecture='efficientnet_b0', freeze_backbone=True)
        self.assertTrue(all(not p.requires_grad for p in model.network.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            direct = model(x)
            features = nn.Sequential(model.network.features, model.network.avgpool, nn.Flatten(1))(x)
            cached = model.fc(features)
        torch.testing.assert_close(direct, cached)
        set_stage(model, {'mode': 'layer4', 'warmup_epochs': 3}, 4)
        self.assertTrue(all(p.requires_grad for p in model.network.features[7:].parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.network.features[:7].parameters()))

    def test_unknown_architecture_rejected(self):
        with self.assertRaises(ValueError):
            build_model(6, architecture='not-a-model')

if __name__ == '__main__':
    unittest.main()
