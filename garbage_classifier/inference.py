"""One checkpoint loader and prediction interface for CLI, evaluation and UI."""
import time

import torch
from PIL import ImageOps

from models.resnet18_model import build_resnet18
from .common import choose_device, synchronize
from .data import MEAN, STD, image_transform


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"模型不存在：{checkpoint}。请先准备数据并训练。")
        self.device = choose_device(device)
        self.checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.classes = self.checkpoint["classes"]
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("模型类别映射无效")
        if tuple(self.checkpoint.get("mean", MEAN)) != MEAN or tuple(self.checkpoint.get("std", STD)) != STD:
            raise ValueError("模型的标准化参数与当前预处理不一致")
        self.model = build_resnet18(len(self.classes)).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()
        self.image_size = self.checkpoint.get("image_size", 224)
        self.transform = image_transform(image_size=self.image_size)

    def predict(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        synchronize(self.device)
        started = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(1)[0].cpu().tolist()
        synchronize(self.device)
        ranking = sorted(zip(self.classes, probabilities), key=lambda row: row[1], reverse=True)
        return {"label": ranking[0][0], "confidence": ranking[0][1],
                "top3": [{"label": label, "probability": probability} for label, probability in ranking[:3]],
                "elapsed_ms": (time.perf_counter() - started) * 1000}
