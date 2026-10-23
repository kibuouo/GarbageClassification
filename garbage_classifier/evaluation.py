"""Held-out test evaluation; never selects a model using test scores."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

from .common import synchronize, write_csv, write_json
from .data import make_loader, validate_split
from .inference import Predictor


def evaluate(checkpoint, data_dir, output, batch_size=32, device="auto"):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数")
    dataset_id = validate_split(data_dir)
    predictor = Predictor(checkpoint, device)
    if predictor.checkpoint.get("dataset_id") != dataset_id:
        raise ValueError("模型与测试集的划分标识不一致，不可混用实验数据")
    loader = make_loader(data_dir, "test", batch_size, image_size=predictor.image_size)
    if loader.dataset.classes != predictor.classes:
        raise ValueError("模型与测试集的类别映射不一致")
    labels = list(range(len(predictor.classes)))
    targets, predictions, rows = [], [], []
    synchronize(predictor.device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, batch_targets in loader:
            probabilities = predictor.model(images.to(predictor.device)).softmax(1).cpu()
            for actual, probs in zip(batch_targets.tolist(), probabilities):
                predicted = int(probs.argmax())
                path = loader.dataset.samples[len(rows)][0]
                rows.append({"image": path, "actual": predictor.classes[actual],
                             "predicted": predictor.classes[predicted], "confidence": float(probs[predicted]),
                             "correct": actual == predicted})
                targets.append(actual)
                predictions.append(predicted)
    synchronize(predictor.device)
    elapsed = time.perf_counter() - started
    report = classification_report(targets, predictions, labels=labels, target_names=predictor.classes, output_dict=True, zero_division=0)
    metrics = {"dataset_id": dataset_id, "checkpoint": str(checkpoint), "test_samples": len(targets),
               "accuracy": accuracy_score(targets, predictions),
               "macro_f1": f1_score(targets, predictions, labels=labels, average="macro", zero_division=0),
               "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
               "evaluation_seconds": elapsed, "device": str(predictor.device),
               "config": predictor.checkpoint.get("config", {}), "classification_report": report}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "test_metrics.json", metrics)
    write_csv(output / "test_predictions.csv", rows)
    mistakes = [row for row in rows if not row["correct"]]
    # Always write a header, even when there are no mistakes.
    import csv
    with (output / "misclassified.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(mistakes)
    matrix = confusion_matrix(targets, predictions, labels=labels)
    write_json(output / "confusion_matrix.json", {"classes": predictor.classes, "matrix": matrix.tolist()})
    _, axis = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=predictor.classes).plot(ax=axis, xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close()
    return {k: metrics[k] for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "test_samples")}
