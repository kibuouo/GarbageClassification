"""在本地权重上继续训练；保存候选模型，不直接覆盖页面正在使用的模型。"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

for variable, value in (("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
    os.environ.setdefault(variable, value)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from evaluate import (
    MEAN, ROOT, STD, ManifestDataset, classification_metrics, evaluate_model,
)
from model import build_model


def augmented_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.inference_mode()
def extract_features(model, dataset, batch_size):
    model.eval()
    extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
    features, targets = [], []
    for images, labels in DataLoader(dataset, batch_size=batch_size):
        features.append(extractor(images))
        targets.append(labels)
    return torch.cat(features), torch.cat(targets)


@torch.inference_mode()
def cached_metrics(model, features, labels, classes):
    predicted = model.fc(features).argmax(1)
    confusion = torch.bincount(labels * len(classes) + predicted,
                               minlength=len(classes) ** 2).reshape(len(classes), -1)
    return classification_metrics(confusion, classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_resnet18.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/finetuned_resnet18.pth")
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.head_epochs < 0 or args.epochs < 0:
        parser.error("batch-size 必须大于 0，轮次不能为负数。")
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("候选模型必须另存，不能覆盖输入模型。")

    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    started = time.monotonic()
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint

    train_plain = ManifestDataset("train", classes)
    train_augmented = ManifestDataset("train", classes, augmented_transform())
    validation = ManifestDataset("val", classes)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    counts = torch.bincount(torch.tensor(train_plain.targets), minlength=len(classes)).float()
    if (counts == 0).any():
        raise ValueError("训练集缺少类别。")
    # 平方根倒数温和补偿少数类，避免极端权重放大标签噪声。
    weights = counts.rsqrt()
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("Preparing validation features...", flush=True)
    val_features, val_labels = extract_features(model, validation, args.batch_size)
    baseline = cached_metrics(model, val_features, val_labels, classes)
    best = dict(baseline)
    history = []
    print(f"Baseline: accuracy={best['accuracy']:.2%}, macro-F1={best['macro_f1']:.4f}", flush=True)
    report_path = ROOT / "results/final/finetuning_history.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(stage, epoch, loss, metrics):
        nonlocal best
        row = dict(stage=stage, epoch=epoch, loss=loss,
                   accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"])
        history.append(row)
        # 两项都不低于原模型，再按准确率、macro-F1 依次选择。
        eligible = (metrics["accuracy"] >= baseline["accuracy"]
                    and metrics["macro_f1"] >= baseline["macro_f1"])
        if eligible and (metrics["accuracy"], metrics["macro_f1"]) > (best["accuracy"], best["macro_f1"]):
            best = dict(metrics)
            torch.save({
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "classes": classes,
                "training_metadata": {
                    "source_checkpoint_sha256": source_hash,
                    "manifest_sha256": hashlib.sha256((ROOT / "data/split.csv").read_bytes()).hexdigest(),
                    "seed": 42, "training_split": "train", "selection_split": "val",
                    "selected_stage": stage, "selected_epoch": epoch,
                    "validation_accuracy": best["accuracy"], "validation_macro_f1": best["macro_f1"],
                    "class_weights": weights.tolist(), "batch_size": args.batch_size,
                    "source_provenance": "原权重缺少训练元数据，独立测试仍需核实其训练来源。",
                },
            }, args.output)
            row["saved"] = True
        report_path.write_text(json.dumps({
            "baseline": baseline, "best": best, "history": history,
            "elapsed_seconds": time.monotonic() - started,
            "source_checkpoint_sha256": source_hash,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{stage} {epoch}: loss={loss:.4f}, val={metrics['accuracy']:.2%}, "
              f"macro-F1={metrics['macro_f1']:.4f}" + (" SAVED" if row.get("saved") else ""), flush=True)

    if args.head_epochs:
        print("Caching training features (original + augmented views)...", flush=True)
        plain_features, labels = extract_features(model, train_plain, args.batch_size)
        print("Original view cached.", flush=True)
        augmented_features, augmented_labels = extract_features(model, train_augmented, args.batch_size)
        train_features = torch.cat([plain_features, augmented_features])
        train_labels = torch.cat([labels, augmented_labels])
        del plain_features, augmented_features
        head_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=64, shuffle=True)
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.head_epochs)
        for epoch in range(1, args.head_epochs + 1):
            total_loss = 0.0
            for features, labels in head_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model.fc(features), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
            scheduler.step()
            metrics = cached_metrics(model, val_features, val_labels, classes)
            record("head", epoch, total_loss / len(train_labels), metrics)
        del train_features, train_labels, head_loader, val_features, val_labels

    if args.epochs:
        # 从本轮最优分类层开始，进一步适配高层视觉特征。
        if args.output.is_file() and any(row.get("saved") for row in history):
            selected = torch.load(args.output, map_location="cpu", weights_only=True)
            model.load_state_dict(selected["model_state"])
            del selected
        else:
            original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(original["model_state"])
            del original
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([
            {"params": model.layer4.parameters(), "lr": 0.0001},
            {"params": model.fc.parameters(), "lr": 0.0005},
        ], weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        train_loader = DataLoader(train_augmented, batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            # 小批次保持全部 BatchNorm 的运行统计量，仿射参数仍可训练。
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            total_loss = 0.0
            for batch_index, (images, labels) in enumerate(train_loader, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                if batch_index % 100 == 0:
                    print(f"layer4 {epoch}: batch {batch_index}/{len(train_loader)}", flush=True)
            scheduler.step()
            metrics, _ = evaluate_model(model, val_loader, classes)
            record("layer4", epoch, total_loss / len(train_augmented), metrics)

    print(f"Finished. Best validation accuracy={best['accuracy']:.2%}, macro-F1={best['macro_f1']:.4f}", flush=True)
    print(f"Candidate: {args.output}; history: {report_path}", flush=True)


if __name__ == "__main__":
    main()
