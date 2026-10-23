"""Experiment configuration; relative paths use the project root."""
import json
from dataclasses import asdict, dataclass

from .common import project_path


@dataclass
class ExperimentConfig:
    name: str = "B_frozen"
    strategy: str = "frozen"
    data_dir: str = "data"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    backbone_learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    warmup_epochs: int = 5
    patience: int = 7
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    image_size: int = 224

    def validate(self):
        if self.strategy not in ("scratch", "frozen", "finetune"):
            raise ValueError("strategy 必须为 scratch、frozen 或 finetune")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device 必须为 auto、cpu、cuda 或 mps")
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.name):
            raise ValueError("实验名只能包含英文字母、数字、下划线和短横线")
        if min(self.epochs, self.batch_size, self.patience) < 1 or self.image_size < 32:
            raise ValueError("epochs、batch_size、patience 必须为正，image_size 至少为 32")
        if min(self.num_workers, self.seed, self.weight_decay) < 0:
            raise ValueError("num_workers、seed、weight_decay 不可为负")
        if min(self.learning_rate, self.backbone_learning_rate) <= 0:
            raise ValueError("学习率必须大于零")
        if self.strategy == "finetune" and not 1 <= self.warmup_epochs < self.epochs:
            raise ValueError("微调实验需要 1 <= warmup_epochs < epochs")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with project_path(path).open(encoding="utf-8") as file:
        return ExperimentConfig(**json.load(file)).validate()
