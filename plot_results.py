"""根据真实日志与测试指标绘图，不生成虚构曲线或占位指标。"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import ROOT, latest_run


def plot_results(results_root, experiments, allow_smoke=False):
    final_dir = results_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    histories = {}
    selected_runs = {}
    for name in experiments:
        run = latest_run(results_root, name, allow_smoke)
        selected_runs[name] = run.resolve()
        with (run / "history.csv").open(encoding="utf-8-sig", newline="") as file:
            histories[name] = list(csv.DictReader(file))
    report_path = final_dir / "test_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if report:
        missing = set(experiments) - set(report["experiments"])
        if missing:
            raise ValueError(f"缺少实验测试指标：{sorted(missing)}")
        for name in experiments:
            metrics = report["experiments"][name]
            if Path(metrics["run_dir"]).resolve() != selected_runs[name]:
                raise ValueError(f"{name} 最新训练与测试记录不一致，请先重新运行 evaluate.py。")
            if metrics["smoke_test"] and not allow_smoke:
                raise ValueError("拒绝将小样本指标绘制为正式实验图。")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, metric, title in zip(axes.flat, ("train_loss", "val_loss", "train_accuracy", "val_accuracy"),
                                   ("Training loss", "Validation loss", "Training accuracy", "Validation accuracy")):
        for name, rows in histories.items():
            axis.plot([int(row["epoch"]) for row in rows], [float(row[metric]) for row in rows], marker="o", label=name)
        axis.set(title=title, xlabel="Epoch", ylabel="Accuracy" if "accuracy" in metric else "Cross entropy")
        if "accuracy" in metric:
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(title="Experiment")
    figure.suptitle("SMOKE TEST - not experimental results" if allow_smoke else "ResNet-18 experiment comparison")
    figure.savefig(final_dir / "training_curves.png", dpi=160)
    plt.close(figure)
    if report is None:
        print("Training curves saved. Run evaluate.py before drawing test confusion matrices.")
        return
    figure, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 5), squeeze=False, layout="constrained")
    for axis, name in zip(axes.flat, experiments):
        metrics = report["experiments"][name]
        if metrics["smoke_test"] and not allow_smoke:
            raise ValueError("拒绝将小样本指标绘制为正式实验图。")
        matrix = np.asarray(metrics["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        classes = metrics["classes"]
        axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
                 yticklabels=classes, xlabel="Predicted label", ylabel="True label",
                 title=f"{name}: test accuracy {metrics['accuracy']:.2%}")
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.65)
    if allow_smoke:
        figure.suptitle("SMOKE TEST - not experimental results")
    figure.savefig(final_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    # 单页演示用的简洁结果图，全部数值取自实际测试记录。
    x = np.arange(len(experiments))
    figure, axis = plt.subplots(figsize=(10, 5.6), layout="constrained")
    width = 0.34
    for offset, key, label, color in ((-width / 2, "accuracy", "Test accuracy", "#166A8F"),
                                      (width / 2, "macro_f1", "Macro F1", "#D8882B")):
        values = [report["experiments"][name][key] for name in experiments]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=4, fontsize=13)
    axis.set(xticks=x, xticklabels=experiments, ylim=(0, 1.12), xlabel="Experiment", ylabel="Score")
    axis.set_title("SMOKE TEST - not experimental results" if allow_smoke else "Independent test set comparison", fontsize=20)
    axis.tick_params(labelsize=13)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(fontsize=13, loc="upper left", ncol=2)
    figure.savefig(final_dir / "accuracy_comparison.png", dpi=200)
    figure.savefig(final_dir / "accuracy_comparison.svg")
    plt.close(figure)
    print(f"Saved plots: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--experiments", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    plot_results(args.results_root, args.experiments, args.allow_smoke)


if __name__ == "__main__":
    main()
