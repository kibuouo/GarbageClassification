"""评估指定权重或已完成的 A/B/C 实验，汇总测试指标。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

# 必须在导入数值库前限制线程，避免 Windows 小内存环境中创建过多线程。
for variable, value in (("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
    os.environ.setdefault(variable, value)

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model
from experiment_utils import atomic_json, audit_manifest, latest_run, limit_dataset, sha256, verify_data_identity

ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ManifestDataset(Dataset):
    """直接读取 raw 图片，避免复制数据或重新随机划分。"""

    def __init__(self, split, classes, transform=None):
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform or evaluation_transform()
        with (ROOT / "data/split.csv").open(encoding="utf-8-sig", newline="") as file:
            self.rows = [row for row in csv.DictReader(file) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"划分 {split} 没有图片。")
        self.targets = [self.class_to_idx[row["class"]] for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / "data/raw/dataset-resized" / row["class"] / row["filename"]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, self.targets[index]


def classification_metrics(confusion, classes):
    per_class = {}
    f1_scores = []
    for index, name in enumerate(classes):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        per_class[name] = dict(support=support, precision=precision, recall=recall, f1=f1)
    total = int(confusion.sum())
    if total == 0:
        raise ValueError("不能评估空数据集。")
    return {
        "samples": total,
        "accuracy": float(confusion.diag().sum()) / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(classes),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(classes),
        "classes": list(classes),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted",
    }


@torch.inference_mode()
def evaluate_model(model, loader, classes, device="cpu"):
    model.eval()
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    predictions = []
    for images, labels in loader:
        probabilities = model(images.to(device)).softmax(1).cpu()
        scores, predicted = probabilities.max(1)
        flat = labels * len(classes) + predicted
        confusion += torch.bincount(flat, minlength=len(classes) ** 2).reshape(confusion.shape)
        predictions.extend(
            {"true": classes[label], "predicted": classes[prediction], "probability": score}
            for label, prediction, score in zip(labels.tolist(), predicted.tolist(), scores.tolist())
        )
    return classification_metrics(confusion, classes), predictions


def evaluate_checkpoint(path, split, batch_size, audit, max_samples=None, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    metadata = checkpoint.get("training_metadata") or {}
    smoke = bool(metadata.get("smoke_test") or max_samples)
    if smoke and not allow_smoke:
        raise ValueError("这是小样本流程验证，不是正式实验；需要显式 --allow-smoke。")
    if metadata:
        verify_data_identity(metadata, audit)
    dataset = limit_dataset(ManifestDataset(split, classes), max_samples)
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    print(f"Evaluating {len(dataset)} {split} images: {path}", flush=True)
    metrics, predictions = evaluate_model(model, DataLoader(dataset, batch_size=batch_size), classes)
    original = dataset.dataset if hasattr(dataset, "dataset") else dataset
    rows = [original.rows[i] for i in dataset.indices] if hasattr(dataset, "indices") else original.rows
    known_source = metadata.get("initialization") in ("ImageNet ResNet18_Weights.DEFAULT", "random")
    metrics.update({
        "split": split, "smoke_test": smoke, "checkpoint_sha256": sha256(path),
        "manifest_sha256": audit["manifest_sha256"], "dataset_sha256": audit["dataset_sha256"],
        "manifest_hash_algorithm": audit.get("manifest_hash_algorithm"),
        "training_metadata": metadata,
        "provenance_note": "从已记录初始化出发，仅 train 参与训练、val 参与选模。" if known_source else
            "旧权重缺少完整训练来源，无法证明其训练数据与当前评估集互斥。",
        "predictions": [dict(row, **prediction) for row, prediction in zip(rows, predictions)],
    })
    print(f"{split}: accuracy={metrics['accuracy']:.2%}, macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return metrics


def export_comparison(final_dir, name, metrics):
    path = final_dir / "test_metrics.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema_version": 1, "split": "test", "selection_rule": "best epoch by validation accuracy; latest completed run per experiment", "experiments": {}}
    for existing in report["experiments"].values():
        verify_data_identity(existing, metrics)
        if existing["smoke_test"] != metrics["smoke_test"]:
            raise ValueError("汇总目录包含不同数据或正式/小样本混合结果，请另选 results-root。")
        # 保留原始记录的哈希，并将汇总身份统一成当前规范哈希。
        if existing["manifest_sha256"] != metrics["manifest_sha256"]:
            existing.setdefault("legacy_manifest_sha256", existing["manifest_sha256"])
        existing["manifest_sha256"] = metrics["manifest_sha256"]
        existing["manifest_hash_algorithm"] = metrics.get("manifest_hash_algorithm")
    report["experiments"][name] = metrics
    atomic_json(path, report)
    columns = ["experiment", "run_id", "best_epoch", "epochs_trained", "pretrained", "seed", "effective_batch",
               "val_accuracy", "val_macro_f1", "test_accuracy", "macro_precision", "macro_recall", "macro_f1",
               "test_samples", "smoke_test", "checkpoint_sha256", "manifest_sha256"]
    temporary = final_dir / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for experiment, result in sorted(report["experiments"].items()):
            meta = result["training_metadata"]
            config = meta["config"]
            writer.writerow(dict(experiment=experiment, run_id=meta["run_id"], best_epoch=meta["epoch"],
                epochs_trained=result["epochs_trained"], pretrained=config["pretrained"], seed=config["seed"],
                effective_batch=config["batch_size"] * config["accumulation_steps"],
                val_accuracy=meta["validation_accuracy"], val_macro_f1=meta["validation_macro_f1"],
                test_accuracy=result["accuracy"], macro_precision=result["macro_precision"],
                macro_recall=result["macro_recall"], macro_f1=result["macro_f1"], test_samples=result["samples"],
                smoke_test=result["smoke_test"], checkpoint_sha256=result["checkpoint_sha256"], manifest_sha256=result["manifest_sha256"]))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, final_dir / "comparison.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--experiments", nargs="+", choices=["A", "B", "C"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="单个权重的指标 JSON 输出路径")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 6):
        parser.error("batch-size 必须大于 0，小样本数至少为 6。")
    if args.output and args.experiments:
        parser.error("多个实验请使用 --results-root，不能指定单一 --output。")
    torch.set_num_threads(2)
    audit = audit_manifest()
    if args.checkpoint or not (args.run_dir or args.experiments):
        metrics = evaluate_checkpoint(args.checkpoint or ROOT / "checkpoints/best_resnet18.pth",
            args.split, args.batch_size, audit, args.max_samples, args.allow_smoke)
        output = args.output or args.results_root / "final" / f"{args.split}_metrics_external.json"
        atomic_json(output, metrics)
        print(f"Saved: {output}")
        return
    runs = [args.run_dir] if args.run_dir else [latest_run(args.results_root, name, args.allow_smoke) for name in args.experiments]
    for run_dir in runs:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status["status"] not in ("completed", "early_stopped"):
            raise ValueError(f"实验未完成：{run_dir}")
        metrics = evaluate_checkpoint(run_dir / "best.pth", args.split, args.batch_size, audit,
                                      args.max_samples, args.allow_smoke)
        metrics.update(run_dir=str(run_dir.resolve()), epochs_trained=status["last_epoch"])
        if metrics["smoke_test"] and args.results_root.resolve() == (ROOT / "results").resolve():
            raise ValueError("小样本结果需使用单独的 --results-root，避免混入正式结果。")
        atomic_json(args.output or run_dir / f"{args.split}_metrics.json", metrics)
        if args.split == "test":
            export_comparison(args.results_root / "final", metrics["training_metadata"]["experiment"], metrics)
    print(f"Saved experiment metrics under {args.results_root}")


if __name__ == "__main__":
    main()
