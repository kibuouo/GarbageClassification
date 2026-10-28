"""A/B/C 对照实验：持久化日志、验证集早停和按轮恢复。"""

import argparse
import csv
import json
import logging
import os
import random
import re
import uuid
import platform
from pathlib import Path
from time import monotonic
from datetime import datetime, timedelta, timezone

for variable, value in (("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2")):
    os.environ.setdefault(variable, value)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import MEAN, STD, ManifestDataset, classification_metrics
from experiment_utils import (ROOT, CLASSES, atomic_json, audit_manifest, limit_dataset,
                              load_config, now, validate_config, verify_data_identity, snapshot_source, sha256)
from model import build_model
from feature_cache import feature_datasets

FIELDS = ["epoch", "stage", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
          "val_macro_f1", "lr_head", "lr_layer4", "seconds"]


class ShanghaiFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def save_checkpoint(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def write_history(run_dir, history):
    temporary = run_dir / "history.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(history)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, run_dir / "history.csv")
    atomic_json(run_dir / "history.json", history)


def set_stage(model, config, epoch):
    model.freeze_features(True)
    tuning = config["mode"] == "layer4" and epoch > config["warmup_epochs"]
    if tuning:
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
    return "layer4" if tuning else "head"


def run_epoch(model, loader, device, optimizer=None, accumulation_steps=1, freeze_bn=True, logger=None, feature_inputs=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_total, images_total, group_samples = 0.0, 0, 0
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for batch_index, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            outputs = model.fc(images) if feature_inputs else model(images)
            loss = nn.functional.cross_entropy(outputs, labels, reduction="sum")
            if training:
                loss.backward()
        size = len(labels)
        loss_total += loss.item()
        images_total += size
        group_samples += size
        encoded = labels.detach().cpu() * len(CLASSES) + outputs.detach().argmax(1).cpu()
        confusion += torch.bincount(encoded, minlength=len(CLASSES) ** 2).reshape(confusion.shape)
        if training and (batch_index % accumulation_steps == 0 or batch_index == len(loader)):
            # 按实际样本数归一化，正确处理最后不足一组的微批次。
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
        if logger and batch_index % 100 == 0:
            logger.info("%s batch %d/%d loss=%.4f", "train" if training else "val",
                        batch_index, len(loader), loss_total / images_total)
    return loss_total / images_total, classification_metrics(confusion, CLASSES)


def train_experiment(config, run_dir, audit, device="auto", resume=False, stop_after=None):
    run_dir = Path(run_dir)
    if resume:
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if previous_status["status"] in ("completed", "early_stopped"):
            raise ValueError("该实验已完成；新实验请使用新的 run-id。")
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / "config.json", config)
    logger = logging.getLogger(f"experiment.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(run_dir / "train.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(ShanghaiFormatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    history, best_accuracy, best_epoch, stale, start_epoch, last_saved_epoch = [], -1.0, 0, 0, 1, 0
    try:
        torch.set_num_threads(config["threads"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device(device)
        if resume:
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            verify_data_identity(metadata, audit)
            if metadata.get("code_lf_sha256"):
                import hashlib
                for name, expected in metadata["code_lf_sha256"].items():
                    actual = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    if actual != expected:
                        raise ValueError(f"恢复被拒绝：训练源码 {name} 已发生变化。")
        else:
            metadata = {"experiment": config["name"], "run_id": run_dir.name, "started_at": now(),
                        "device": str(device), "torch_version": str(torch.__version__),
                        "python_version": platform.python_version(), "platform": platform.platform(),
                        "smoke_test": bool(config.get("max_train_samples") or config.get("max_val_samples")),
                        "initialization": "ImageNet ResNet18_Weights.DEFAULT" if config["pretrained"] else "random",
                        **audit, **snapshot_source(run_dir)}
            atomic_json(run_dir / "metadata.json", metadata)
        atomic_json(run_dir / "status.json", {"status": "running", "updated_at": now(), "resumed": resume})
        model = build_model(len(CLASSES), pretrained=config["pretrained"] and not resume, freeze_backbone=True).to(device)
        if not resume and config["pretrained"]:
            from torchvision.models import ResNet18_Weights
            weights_file = Path(torch.hub.get_dir()) / "checkpoints" / ResNet18_Weights.DEFAULT.url.rsplit("/", 1)[-1]
            metadata["imagenet_weights_sha256"] = sha256(weights_file)
            atomic_json(run_dir / "metadata.json", metadata)
        optimizer = torch.optim.AdamW([
            {"params": model.fc.parameters(), "lr": config["head_lr"], "name": "head"},
            {"params": model.layer4.parameters(), "lr": config["layer4_lr"], "name": "layer4"},
        ], weight_decay=config["weight_decay"], foreach=False)
        if resume:
            last = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=True)
            if last["training_metadata"]["config"] != config:
                raise ValueError("恢复被拒绝：config.json 已发生变化。")
            model.load_state_dict(last["model_state"])
            optimizer.load_state_dict(last["optimizer_state"])
            history, best_accuracy, best_epoch, stale = last["history"], last["best_accuracy"], last["best_epoch"], last["stale"]
            start_epoch = last["epoch"] + 1
            last_saved_epoch = last["epoch"]
            torch.set_rng_state(last["torch_rng"])
            random.setstate(last["python_rng"])
            if device.type == "cuda" and last.get("cuda_rng"):
                torch.cuda.set_rng_state_all(last["cuda_rng"])
            # last 先提交、best 后提交；若中断发生在两者之间，可由 last 修复 best。
            if best_epoch == last_saved_epoch:
                save_checkpoint(run_dir / "best.pth", {key: last[key] for key in ("model_state", "classes", "training_metadata")})
            del last
            write_history(run_dir, history)
        already_stopped = config["patience"] and stale >= config["patience"] and (
            config["mode"] == "head" or last_saved_epoch > config["warmup_epochs"])
        if resume and (start_epoch > config["epochs"] or already_stopped):
            atomic_json(run_dir / "status.json", {"status": "early_stopped" if already_stopped else "completed",
                "last_epoch": last_saved_epoch, "best_epoch": best_epoch, "best_accuracy": best_accuracy, "updated_at": now()})
            logger.info("Recovered final checkpoint and completed status: %s", run_dir)
            return run_dir
        preprocessing = [transforms.Resize((224, 224))]
        if config["augmentation"] == "horizontal_flip":
            preprocessing.append(transforms.RandomHorizontalFlip())
        preprocessing.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        train_data = limit_dataset(ManifestDataset("train", CLASSES, transforms.Compose(preprocessing)), config.get("max_train_samples"))
        val_data = limit_dataset(ManifestDataset("val", CLASSES), config.get("max_val_samples"))
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=config["batch_size"], num_workers=0)
        cached_loaders = None
        if config.get("cache_features", False) and not metadata["smoke_test"] and (
                config["mode"] == "head" or start_epoch <= config["warmup_epochs"]):
            cached_train, cached_val, key = feature_datasets(model, config, audit,
                run_dir.parents[2] / "cache", device, logger)
            metadata["feature_cache_key"] = key
            atomic_json(run_dir / "metadata.json", metadata)
            cached_loaders = (DataLoader(cached_train, batch_size=config["batch_size"], shuffle=True, num_workers=0),
                              DataLoader(cached_val, batch_size=config["batch_size"], num_workers=0))
        logger.info("Experiment %s device=%s train=%d val=%d effective_batch=%d resume_epoch=%d",
                    config["name"], device, len(train_data), len(val_data), config["batch_size"] * config["accumulation_steps"], start_epoch)
        outcome = "completed"
        for epoch in range(start_epoch, config["epochs"] + 1):
            began = monotonic()
            stage = set_stage(model, config, epoch)
            if stage == "layer4" and epoch == config["warmup_epochs"] + 1:
                stale = 0
            use_cache = stage == "head" and cached_loaders is not None
            active_train, active_val = cached_loaders if use_cache else (train_loader, val_loader)
            train_loss, train_metrics = run_epoch(model, active_train, device, optimizer,
                config["accumulation_steps"], config["freeze_bn_stats"], logger, feature_inputs=use_cache)
            val_loss, val_metrics = run_epoch(model, active_val, device, feature_inputs=use_cache)
            improved = val_metrics["accuracy"] > best_accuracy + config["min_delta"]
            if improved:
                best_accuracy, best_epoch, stale = val_metrics["accuracy"], epoch, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, stage=stage, train_loss=train_loss, train_accuracy=train_metrics["accuracy"],
                val_loss=val_loss, val_accuracy=val_metrics["accuracy"], val_macro_f1=val_metrics["macro_f1"],
                lr_head=optimizer.param_groups[0]["lr"], lr_layer4=optimizer.param_groups[1]["lr"] if stage == "layer4" else 0.0,
                seconds=monotonic() - began))
            saved = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "classes": CLASSES,
                "training_metadata": {**metadata, "config": config, "epoch": epoch,
                "validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"],
                "selection_split": "val", "training_split": "train"}}
            save_checkpoint(run_dir / "last.pth", {**saved, "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                "best_epoch": best_epoch, "best_accuracy": best_accuracy, "stale": stale, "history": history,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            last_saved_epoch = epoch
            if improved:
                save_checkpoint(run_dir / "best.pth", saved)
            write_history(run_dir, history)
            atomic_json(run_dir / "status.json", {"status": "running", "last_epoch": epoch,
                "best_epoch": best_epoch, "best_accuracy": best_accuracy, "updated_at": now()})
            logger.info("epoch=%d/%d stage=%s loss=%.4f val_loss=%.4f train=%.2f%% val=%.2f%% macro_F1=%.4f%s",
                epoch, config["epochs"], stage, train_loss, val_loss, 100 * train_metrics["accuracy"],
                100 * val_metrics["accuracy"], val_metrics["macro_f1"], " BEST" if improved else "")
            if config["patience"] and stale >= config["patience"] and (config["mode"] == "head" or stage == "layer4"):
                outcome = "early_stopped"
                break
            if stop_after and epoch - start_epoch + 1 >= stop_after and epoch < config["epochs"]:
                outcome = "paused"
                break
        atomic_json(run_dir / "status.json", {"status": outcome, "last_epoch": history[-1]["epoch"],
            "best_epoch": best_epoch, "best_accuracy": best_accuracy, "updated_at": now()})
        logger.info("%s best_epoch=%d best_val=%.2f%% run_dir=%s", outcome, best_epoch, best_accuracy * 100, run_dir)
        return run_dir
    except BaseException as error:
        logger.exception("Experiment interrupted; previous complete epoch remains in last.pth")
        atomic_json(run_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": str(error), "last_epoch": last_saved_epoch, "updated_at": now()})
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--experiment", choices=["A", "B", "C", "all"])
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", type=Path)
    for name in ("epochs", "batch-size", "accumulation-steps", "patience", "warmup-epochs", "max-train-samples", "max-val-samples"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int, help="本次只运行指定轮数，可 --resume 继续")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after 必须为正整数。")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run-id 只允许字母、数字、下划线和连字符。")
    overrides = {key: getattr(args, key) for key in ("epochs", "batch_size", "accumulation_steps", "patience",
        "warmup_epochs", "max_train_samples", "max_val_samples") if getattr(args, key) is not None}
    if any(value < 6 for key, value in overrides.items() if key.startswith("max_")):
        parser.error("小样本验证至少需要 6 张图片。")
    if args.resume and (overrides or args.no_pretrained or args.run_id):
        parser.error("恢复时不能修改配置或 run-id。")
    audit = audit_manifest()
    if args.resume:
        config = load_config(args.resume / "config.json")
        if args.dry_run:
            print(json.dumps({"config": config, "audit": audit}, indent=2))
        else:
            train_experiment(config, args.resume, audit, args.device, True, args.stop_after)
        return
    names = ["A", "B", "C"] if args.experiment == "all" else [args.experiment or "A"]
    for name in names:
        config = load_config(args.config or ROOT / "experiments" / f"{name}.json")
        config.update(overrides)
        if args.no_pretrained:
            config["pretrained"] = False
        validate_config(config)
        if args.dry_run:
            print(json.dumps({"config": config, "audit": audit}, indent=2))
            continue
        run_id = args.run_id or re.sub(r"[^0-9T]", "", now()) + "_" + uuid.uuid4().hex[:6]
        train_experiment(config, args.output_root / "runs" / config["name"] / run_id,
                         audit, args.device, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
