"""独立复算已交付 A/B/C 最佳权重的完整测试预测，不依赖 results/runs。"""

import argparse
import json
from pathlib import Path

from evaluate import evaluate_checkpoint
from experiment_utils import ROOT, atomic_json, audit_manifest, now, sha256, verify_data_identity
import torch


def verify_results(final):
    manifest = json.loads((final / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = final / name
        if not path.resolve().is_relative_to(final.resolve()):
            raise ValueError(f"归档路径越界：{name}")
        if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"归档文件校验失败：{name}")
    report = json.loads((final / "test_metrics.json").read_text(encoding="utf-8"))
    audit = audit_manifest()
    verified = {}
    torch.set_num_threads(2)
    for name in ("A", "B", "C"):
        expected = report["experiments"][name]
        if expected["split"] != "test" or expected["smoke_test"] or expected["samples"] != sum(audit["counts"]["test"].values()):
            raise ValueError(f"{name} 必须是完整的正式测试结果。")
        verify_data_identity(expected, audit)
        checkpoint = final / "checkpoints" / name / "best.pth"
        if sha256(checkpoint) != expected["checkpoint_sha256"]:
            raise ValueError(f"{name} 权重 SHA-256 与测试记录不同。")
        meta = expected["training_metadata"]
        for filename, digest in meta["code_sha256"].items():
            if sha256(final / "training_logs" / name / "source" / filename) != digest:
                raise ValueError(f"{name} 的训练源码 {filename} 不匹配。")
        actual = evaluate_checkpoint(checkpoint, "test", 4, audit)
        for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "confusion_matrix", "classes", "samples"):
            if actual[key] != expected[key]:
                raise ValueError(f"{name} 复算的 {key} 与归档结果不一致。")
        if actual["training_metadata"] != meta:
            raise ValueError(f"{name} 检查点内的训练来源与测试记录不同。")
        drift = 0.0
        if len(actual["predictions"]) != len(expected["predictions"]):
            raise ValueError(f"{name} 逐图预测数量不一致。")
        for left, right in zip(actual["predictions"], expected["predictions"]):
            for key in ("split", "class", "filename", "true", "predicted"):
                if left[key] != right[key]:
                    raise ValueError(f"{name} 逐图预测不一致：{left['filename']}")
            drift = max(drift, abs(left["probability"] - right["probability"]))
        verified[name] = {"checkpoint_sha256": actual["checkpoint_sha256"], "samples": actual["samples"],
                          "accuracy": actual["accuracy"], "macro_f1": actual["macro_f1"],
                          "all_predictions_match": True, "max_probability_difference": drift}
    verification = {"verified_at": now(), "method": "Reload each archived best.pth, rerun all test images and compare every label, confusion matrix and aggregate metric.",
                    "artifact_files_verified": len(manifest["files"]), "data_audit": audit, "experiments": verified}
    atomic_json(final / "evaluation_verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=ROOT / "results/final")
    verify_results(parser.parse_args().final_dir)
