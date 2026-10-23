"""Training stages, early stopping and real experiment artifacts."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from models.resnet18_model import build_resnet18, set_trainable, training_mode
from .common import CLASSES, ROOT, choose_device, project_path, seed_everything, synchronize, write_csv, write_json
from .data import MEAN, STD, make_loader, validate_split


def run_epoch(model, loader, device, stage, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        training_mode(model, stage)
    loss_sum, targets, predictions = 0.0, [], []
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.detach().argmax(1).cpu().tolist())
    return {"loss": loss_sum / len(targets), "accuracy": accuracy_score(targets, predictions),
            "macro_f1": f1_score(targets, predictions, labels=list(range(len(CLASSES))), average="macro", zero_division=0)}


def make_optimizer(model, stage, config):
    set_trainable(model, stage)
    if stage == "finetune":
        groups = [{"params": model.fc.parameters(), "lr": config.learning_rate},
                  {"params": model.layer4.parameters(), "lr": config.backbone_learning_rate}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.learning_rate}]
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def save_curves(history, output):
    for metric in ("loss", "accuracy"):
        plt.figure(figsize=(8, 4))
        for split in ("train", "val"):
            plt.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{metric}_curve.png", dpi=150)
        plt.close()


def train(config, output_root=ROOT):
    config.validate()
    seed_everything(config.seed)
    device = choose_device(config.device)
    data_dir = project_path(config.data_dir)
    dataset_id = validate_split(data_dir)
    output = output_root / "results" / config.name
    checkpoint_dir = output_root / "checkpoints" / config.name
    for path in (output, checkpoint_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"实验输出已存在：{path}；请更换实验名以保留历史结果")
    train_loader = make_loader(data_dir, "train", config.batch_size, config.num_workers, config.seed, config.image_size)
    val_loader = make_loader(data_dir, "val", config.batch_size, config.num_workers, config.seed, config.image_size)
    # Pretrained weights are only downloaded for the two transfer learning experiments.
    model = build_resnet18(len(CLASSES), pretrained=config.strategy != "scratch").to(device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    history, best, best_epoch, stale, previous_stage = [], -1.0, 0, 0, None
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        stage = "frozen" if config.strategy == "finetune" and epoch <= config.warmup_epochs else config.strategy
        if stage != previous_stage:
            optimizer = make_optimizer(model, stage, config)
            stale = 0
            previous_stage = stage
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, stage, optimizer)
        val_metrics = run_epoch(model, val_loader, device, stage)
        synchronize(device)
        row = {"epoch": epoch, "stage": stage, "seconds": time.perf_counter() - epoch_started,
               **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        write_csv(output / "training_history.csv", history)
        print(f"{config.name} | {epoch}/{config.epochs} | {stage} | val acc={val_metrics['accuracy']:.4f} | val F1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best:
            best, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "classes": list(CLASSES), "config": config.to_dict(),
                        "image_size": config.image_size, "mean": list(MEAN), "std": list(STD),
                        "dataset_id": dataset_id, "epoch": epoch, "val_metrics": val_metrics}, checkpoint_dir / "best_resnet18.pth")
        else:
            stale += 1
        # Warmup always completes before applying the fine-tuning early-stop budget.
        if stale >= config.patience and not (config.strategy == "finetune" and stage == "frozen"):
            break
    elapsed = time.perf_counter() - started
    save_curves(history, output)
    summary = {"name": config.name, "strategy": config.strategy, "dataset_id": dataset_id,
               "seed": config.seed, "device": str(device), "epochs_run": len(history), "best_epoch": best_epoch,
               "best_val_macro_f1": best, "training_seconds": elapsed,
               "parameters": sum(p.numel() for p in model.parameters()),
               "final_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(output / "train_summary.json", summary)
    return checkpoint_dir / "best_resnet18.pth"
