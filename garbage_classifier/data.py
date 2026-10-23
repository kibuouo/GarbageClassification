"""Shared preprocessing and manifest-checked datasets."""
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .common import CLASSES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_transform(training=False, image_size=224):
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_id(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_split(data_dir):
    path = data_dir / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少固定划分清单 {path}，请先运行 prepare_data.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest["records"]
    if manifest["dataset_id"] != manifest_id(records):
        raise ValueError("数据划分清单已被改动，请重新准备数据")
    hashes = set()
    expected = {"train": set(), "val": set(), "test": set()}
    for row in records:
        relative = row["destination"]
        parts = PurePosixPath(relative).parts
        if len(parts) != 3 or parts[0] not in expected or parts[1] not in CLASSES or ".." in parts:
            raise ValueError(f"划分清单中的路径无效：{relative}")
        if relative in expected[parts[0]]:
            raise ValueError(f"清单存在重复路径：{relative}")
        expected[parts[0]].add(relative)
        if row["sha256"] in hashes:
            raise ValueError("划分中存在内容重复图片，可能造成数据泄漏")
        hashes.add(row["sha256"])
        file = data_dir / relative
        if not file.is_file() or file_hash(file) != row["sha256"]:
            raise ValueError(f"数据缺失或被改动：{file}")
    for split in expected:
        dataset = datasets.ImageFolder(data_dir / split)
        if dataset.classes != list(CLASSES):
            raise ValueError(f"{split} 必须包含 TrashNet 的六个类别")
        actual = {Path(p).relative_to(data_dir).as_posix() for p, _ in dataset.samples}
        if actual != expected[split]:
            raise ValueError(f"{split} 中的图片与固定划分清单不一致")
    return manifest["dataset_id"]


def make_loader(data_dir, split, batch_size, num_workers=0, seed=42, image_size=224):
    dataset = datasets.ImageFolder(data_dir / split, transform=image_transform(split == "train", image_size))
    if dataset.classes != list(CLASSES):
        raise ValueError(f"{split} 类别不正确，预期：{', '.join(CLASSES)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed))
