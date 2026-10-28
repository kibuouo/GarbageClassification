"""仅为与当前权重逐字节匹配的正式测试记录提供页面指标。"""

import json
from pathlib import Path

from experiment_utils import sha256


def model_evidence(checkpoint, report_path):
    try:
        digest = sha256(checkpoint)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("split") != "test" or not isinstance(report.get("experiments"), dict):
            return None
        for name, metrics in report["experiments"].items():
            if not isinstance(metrics, dict):
                continue
            metadata = metrics.get("training_metadata", {})
            if not isinstance(metadata, dict):
                continue
            if (metrics.get("checkpoint_sha256") == digest
                    and metrics.get("split") == "test" and metrics.get("smoke_test") is False
                    and metadata.get("training_split") == "train"
                    and metadata.get("selection_split") == "val"
                    and metadata.get("dataset_sha256") == metrics.get("dataset_sha256")
                    and metadata.get("manifest_sha256") == metrics.get("manifest_sha256")
                    and metadata.get("code_sha256")
                    and 0 <= metrics["accuracy"] <= 1 and metrics["samples"] > 0):
                return {"experiment": name, "run_id": metadata["run_id"],
                        "accuracy": metrics["accuracy"], "samples": metrics["samples"],
                        "checkpoint_sha256": digest}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
