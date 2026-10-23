"""Aggregate measured metrics while checking experiment comparability."""
import json

from .common import write_csv, write_json


def compare(run_dirs, output):
    rows, identities = [], set()
    for folder in run_dirs:
        summary = json.loads((folder / "train_summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if summary["dataset_id"] != metrics["dataset_id"] or config != metrics["config"]:
            raise ValueError(f"{folder} 的训练与评估不是同一次实验")
        identities.add((summary["dataset_id"], config["seed"], config["image_size"], config["batch_size"],
                        config["epochs"], config["patience"], summary["device"], metrics["device"]))
        rows.append({"experiment": summary["name"], "strategy": summary["strategy"],
                     "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                     "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
                     "training_seconds": summary["training_seconds"], "epochs_run": summary["epochs_run"],
                     "best_epoch": summary["best_epoch"], "device": summary["device"]})
    if not rows:
        raise ValueError("请提供至少一个已完成训练和评估的实验目录")
    if len(identities) != 1:
        raise ValueError("数据划分、训练预算或设备不同，不能直接合并为公平对比表")
    write_csv(output / "experiment_comparison.csv", rows)
    write_json(output / "experiment_comparison.json", rows)
    print(f"实验对比已生成：{output}")
    return rows
