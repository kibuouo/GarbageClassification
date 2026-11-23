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
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

from experiment_utils import ROOT, latest_run


def plot_results(results_root, experiments, allow_smoke=False, archived=False):
    report_dir = results_root / "final"
    # 跨平台重绘可能使用不同字体，写入独立目录，保留已校验交付件的字节。
    final_dir = results_root / "replotted" if archived else report_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    histories = {}
    selected_runs = {}
    for name in experiments:
        run = report_dir / "training_logs" / name if archived else latest_run(results_root, name, allow_smoke)
        selected_runs[name] = run.resolve()
        with (run / "history.csv").open(encoding="utf-8-sig", newline="") as file:
            histories[name] = list(csv.DictReader(file))
    report_path = report_dir / "test_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if report:
        missing = set(experiments) - set(report["experiments"])
        if missing:
            raise ValueError(f"缺少实验测试指标：{sorted(missing)}")
        for name in experiments:
            metrics = report["experiments"][name]
            metadata = json.loads((selected_runs[name] / "metadata.json").read_text(encoding="utf-8"))
            if (metrics["training_metadata"]["run_id"] != metadata["run_id"]
                    or metrics["training_metadata"].get("code_sha256") != metadata.get("code_sha256")):
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
    # 单页演示用的准确率图，全部数值取自实际独立测试记录。
    font_names = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in font_names), None)
    if chinese_font:
        plt.rcParams["font.family"] = chinese_font
        plt.rcParams["axes.unicode_minus"] = False
    x = np.arange(len(experiments))
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    values = [report["experiments"][name]["accuracy"] for name in experiments]
    bars = axis.bar(x, values, width=0.55, color=[{"A": "#8593A3", "B": "#2486AD", "C": "#229B83"}[name] for name in experiments])
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=8, fontsize=19, fontweight="bold")
    labels = []
    for name in experiments:
        metrics = report["experiments"][name]
        meta = metrics["training_metadata"]
        config = meta["config"]
        method = ("微调 layer4 + 分类层" if config["mode"] == "layer4" else "仅训练分类层") if chinese_font else config["mode"]
        detail = f"训练 {metrics['epochs_trained']} 轮 · 最佳第 {meta['epoch']} 轮" if chinese_font else f"Trained {metrics['epochs_trained']} epochs; best {meta['epoch']}"
        labels.append(f"{name} 组\n{method}\n{detail}" if chinese_font else f"{name}\n{method}\n{detail}")
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), ylabel="测试集准确率" if chinese_font else "Test accuracy")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    title = "垃圾分类：三组实验准确率对比" if chinese_font else "Garbage classification: test accuracy comparison"
    axis.set_title("SMOKE TEST - not experimental results" if allow_smoke else title, fontsize=21, pad=20)
    axis.tick_params(labelsize=12)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    sample_counts = {report["experiments"][name]["samples"] for name in experiments}
    if len(sample_counts) != 1:
        raise ValueError("准确率对比要求各组使用相同数量的测试图片。")
    count = sample_counts.pop()
    note = f"同一独立测试集：{count} 张图片｜按验证集准确率选取各组最佳模型" if chinese_font else f"Same held-out test set: {count} images | Best checkpoints selected by validation accuracy"
    figure.supxlabel(note, fontsize=11)
    figure.savefig(final_dir / "accuracy_comparison.png", dpi=200)
    figure.savefig(final_dir / "accuracy_comparison.svg")
    plt.close(figure)
    print(f"Saved plots: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--experiments", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--archived", action="store_true", help="直接从 final/training_logs 读取随仓库交付的记录")
    args = parser.parse_args()
    plot_results(args.results_root, args.experiments, args.allow_smoke, args.archived)


if __name__ == "__main__":
    main()
