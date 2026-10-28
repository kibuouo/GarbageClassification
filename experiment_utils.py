"""实验配置、原始数据审计和持久化工具。"""

import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if config["name"] not in ("A", "B", "C") or config["mode"] not in ("head", "layer4"):
        raise ValueError("实验名称必须为 A/B/C，mode 必须为 head/layer4。")
    for key in ("epochs", "batch_size", "accumulation_steps", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} 必须为正整数。")
    for key in ("warmup_epochs", "patience", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} 必须为非负整数。")
    if config["mode"] == "layer4" and config["warmup_epochs"] >= config["epochs"]:
        raise ValueError("layer4 模式的 warmup_epochs 必须小于 epochs。")
    for key in ("head_lr", "layer4_lr"):
        if not 0 < config[key] < 1:
            raise ValueError(f"{key} 必须在 0 与 1 之间。")
    if config["weight_decay"] < 0 or config["min_delta"] < 0:
        raise ValueError("weight_decay 和 min_delta 不能为负数。")
    for key in ("pretrained", "freeze_bn_stats"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} 必须为布尔值。")
    if config["augmentation"] not in ("horizontal_flip", "none"):
        raise ValueError("augmentation 必须为 horizontal_flip/none。")


def audit_manifest():
    manifest = ROOT / "data/split.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    identities, content_splits, counts = set(), {}, {s: {c: 0 for c in CLASSES} for s in ("train", "val", "test")}
    dataset_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (r["split"], r["class"], r["filename"])):
        split, label, filename = row["split"], row["class"], row["filename"]
        if split not in counts or label not in CLASSES or Path(filename).name != filename or "\\" in filename:
            raise ValueError(f"无效划分记录：{row}")
        identity = (label, filename)
        if identity in identities:
            raise ValueError(f"图片被重复划分：{identity}")
        identities.add(identity)
        digest = sha256(ROOT / "data/raw/dataset-resized" / label / filename)
        if digest in content_splits and content_splits[digest] != split:
            raise ValueError(f"跨集合发现内容完全相同的图片：{identity}")
        content_splits[digest] = split
        counts[split][label] += 1
        dataset_digest.update(f"{split}/{label}/{filename}:{digest}\n".encode())
    if any(n == 0 for group in counts.values() for n in group.values()):
        raise ValueError("每个划分都必须包含全部六个类别。")
    return {"manifest_sha256": sha256(manifest), "dataset_sha256": dataset_digest.hexdigest(), "counts": counts}


def limit_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < len(CLASSES):
        raise ValueError("小样本验证至少需要 6 张图片。")
    from torch.utils.data import Subset
    groups = [[i for i, label in enumerate(dataset.targets) if label == c] for c in range(len(CLASSES))]
    indices = []
    for items in zip(*groups):
        indices.extend(items)
        if len(indices) >= maximum:
            break
    # zip 受最小类数量限制；不足时按原顺序补齐。
    selected = set(indices)
    indices.extend(i for i in range(len(dataset)) if i not in selected)
    return Subset(dataset, indices[:maximum])


def latest_run(results_root, experiment, allow_smoke=False):
    candidates = []
    for status_path in (Path(results_root) / "runs" / experiment).glob("*/status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] not in ("completed", "early_stopped"):
            continue
        metadata = json.loads((status_path.parent / "metadata.json").read_text(encoding="utf-8"))
        if allow_smoke or not metadata["smoke_test"]:
            candidates.append(status_path.parent)
    if not candidates:
        raise FileNotFoundError(f"没有 {experiment} 已完成的实验。请先运行 train.py --experiment {experiment}。")
    return max(candidates, key=lambda path: json.loads((path / "metadata.json").read_text(encoding="utf-8"))["started_at"])
