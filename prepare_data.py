"""Validate local TrashNet images and create a reproducible stratified split."""
import argparse
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from garbage_classifier.common import CLASSES, project_path, write_csv, write_json
from garbage_classifier.data import file_hash, manifest_id

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare(source, output, report_dir, seed=42, ratios=(0.70, 0.15, 0.15)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("三个划分比例须为正数且总和为 1")
    if seed < 0:
        raise ValueError("seed 不可为负")
    if not source.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在：{source}")
    if any(source == output / s or (output / s) in source.parents for s in ("train", "val", "test")):
        raise ValueError("原始数据不可放在划分输出目录内")
    for split in ("train", "val", "test"):
        target = output / split
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"{target} 非空，为避免覆盖固定划分，请使用新的 --output 目录")
    if (output / "split_manifest.json").exists():
        raise FileExistsError("划分清单已存在，请使用新的输出目录")
    actual_classes = {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if actual_classes != set(CLASSES):
        raise ValueError(f"原始目录应直接包含六个类别文件夹：{', '.join(CLASSES)}；当前：{sorted(actual_classes)}")
    rng, seen, records, counts = random.Random(seed), {}, [], []
    # Validate all files before copying anything. Duplicates are rejected, never silently deleted.
    for label in CLASSES:
        paths = sorted(p for p in (source / label).iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
        if len(paths) < 3:
            raise ValueError(f"{label} 至少需要三张有效图片")
        hashes = {}
        for path in paths:
            with Image.open(path) as image:
                image.convert("RGB").load()
            digest = file_hash(path)
            if digest in seen:
                raise ValueError(f"发现内容重复图片：{path} 和 {seen[digest]}，请检查后再划分")
            hashes[path] = digest
            seen[digest] = path
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * ratios[1]))
        n_test = max(1, round(len(paths) * ratios[2]))
        n_train = len(paths) - n_val - n_test
        if n_train < 1:
            raise ValueError(f"{label} 样本不足以满足指定比例")
        counts.append({"class": label, "total": len(paths), "train": n_train, "val": n_val, "test": n_test})
        groups = (paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:])
        for split, group in zip(("train", "val", "test"), groups):
            for path in group:
                records.append({"source": path.relative_to(source).as_posix(), "class": label,
                                "destination": f"{split}/{label}/{path.name}", "sha256": hashes[path]})
    for row in records:
        target = output / row["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row["source"], target)
    dataset_id = manifest_id(records)
    write_json(output / "split_manifest.json", {"seed": seed, "ratios": ratios, "dataset_id": dataset_id, "records": records})
    write_csv(report_dir / "class_distribution.csv", counts)
    plt.figure(figsize=(8, 4))
    plt.bar([r["class"] for r in counts], [r["total"] for r in counts])
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close()
    return counts, dataset_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/dataset-resized", help="直接包含六个类别文件夹的目录")
    parser.add_argument("--output", default="data")
    parser.add_argument("--report-dir", default="results/dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    args = parser.parse_args()
    counts, dataset_id = prepare(project_path(args.source), project_path(args.output), project_path(args.report_dir), args.seed, args.ratios)
    print(*counts, sep="\n")
    print(f"固定划分完成，dataset_id={dataset_id}")


if __name__ == "__main__":
    main()
