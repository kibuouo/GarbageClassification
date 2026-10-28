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


def manifest_fingerprint(path):
    """清单身份使用 LF 字节；保留 BOM，兼容已有 LF 原始字节哈希。"""
    raw = Path(path).read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "manifest_hash_algorithm": "sha256-lf-v1",
        "manifest_raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def verify_data_identity(record, audit, manifest=None):
    """旧记录仅允许当前清单重建的 LF/CRLF 哈希，始终核对图片与划分综合哈希。"""
    if record.get("dataset_sha256") != audit["dataset_sha256"]:
        raise ValueError("dataset_sha256 与记录不同，拒绝混用数据。")
    algorithm = record.get("manifest_hash_algorithm")
    if algorithm not in (None, "sha256-lf-v1"):
        raise ValueError(f"不支持的清单哈希算法：{algorithm}")
    if record.get("manifest_sha256") == audit["manifest_sha256"]:
        return
    if algorithm is not None:
        raise ValueError("manifest_sha256 与记录不同，拒绝混用划分。")
    path = Path(manifest) if manifest is not None else ROOT / "data/split.csv"
    canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if hashlib.sha256(canonical).hexdigest() != audit["manifest_sha256"]:
        raise ValueError("当前清单与数据审计不一致，请重新审计。")
    legacy_hashes = {hashlib.sha256(data).hexdigest() for data in
                     (canonical, canonical.replace(b"\n", b"\r\n"))}
    if record.get("manifest_sha256") not in legacy_hashes:
        raise ValueError("manifest_sha256 与记录不同，拒绝混用划分。")


SOURCE_FILES = ("train.py", "model.py", "evaluate.py", "experiment_utils.py", "feature_cache.py")


def snapshot_source(run_dir):
    """保存实际执行源码与原始/LF 哈希，避免仅有哈希却无法取回源码。"""
    source = Path(run_dir) / "source"
    source.mkdir()
    hashes, normalized = {}, {}
    for name in SOURCE_FILES:
        data = (ROOT / name).read_bytes()
        (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        normalized[name] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    return {"code_sha256": hashes, "code_lf_sha256": normalized}


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
    return {**manifest_fingerprint(manifest), "dataset_sha256": dataset_digest.hexdigest(), "counts": counts}


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
