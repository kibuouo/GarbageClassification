"""归档可复核的最佳权重、训练源码和日志，可选择按验证集选模后更新页面模型。"""

import argparse
import json
import os
import shutil
import platform
from importlib.metadata import version
from pathlib import Path

from experiment_utils import ROOT, atomic_json, latest_run, now, sha256


def archive_results(results_root, promote=False):
    final = results_root / "final"
    report = json.loads((final / "test_metrics.json").read_text(encoding="utf-8"))
    runs = {}
    for name in ("A", "B", "C"):
        run = latest_run(results_root, name)
        metrics = report["experiments"][name]
        metadata = metrics["training_metadata"]
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if (metadata["run_id"] != run.name or metrics["smoke_test"]
                or sha256(run / "best.pth") != metrics["checkpoint_sha256"]
                or status["best_epoch"] != metadata["epoch"]):
            raise ValueError(f"{name} 的检查点、测试或选模记录不一致，先重新评估。")
        runs[name] = run
    # 测试指标不参与模型选择，验证准确率相同时保留较早字母。
    selected = max(sorted(runs), key=lambda name:
                   report["experiments"][name]["training_metadata"]["validation_accuracy"])
    for name, run in runs.items():
        logs = final / "training_logs" / name
        logs.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "metadata.json", "status.json", "history.csv", "history.json", "train.log"):
            shutil.copyfile(run / filename, logs / filename)
        shutil.copytree(run / "source", logs / "source", dirs_exist_ok=True)
        for filename, digest in report["experiments"][name]["training_metadata"]["code_sha256"].items():
            if sha256(logs / "source" / filename) != digest:
                raise ValueError(f"{name} 源码归档校验失败：{filename}")
        destination = final / "checkpoints" / name / "best.pth"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "best.pth", destination)
    atomic_json(final / "experiment_protocol.json", {
        "recorded_at": now(), "run_ids": {name: run.name for name, run in runs.items()},
        "configs": {name: json.loads((run / "config.json").read_text(encoding="utf-8")) for name, run in runs.items()},
        "manifest_sha256": report["experiments"]["A"]["manifest_sha256"],
        "manifest_hash_algorithm": "sha256-lf-v1",
        "dataset_sha256": report["experiments"]["A"]["dataset_sha256"],
        "code_sha256_by_experiment": {name: report["experiments"][name]["training_metadata"]["code_sha256"] for name in runs},
        "selected_experiment": selected,
        "selection_rule": "Highest validation accuracy; first epoch and earlier experiment letter retained on ties. Test metrics never used for selection.",
        "test_rule": "Evaluate best.pth on all 378 fixed test images only after training and validation selection.",
        "historical_results": "results/archive/ppt_training (original checkpoints unavailable; historical metrics retained unchanged)",
    })
    packages = {name: version(name) for name in ("torch", "torchvision", "Pillow", "numpy", "streamlit", "matplotlib", "streamlit-webrtc")}
    atomic_json(final / "environment.json", {"recorded_at": now(), "python": platform.python_version(),
                "platform": platform.platform(), "packages": packages})
    lines = ["# A/B/C 实验测试结果", "",
             "本次为正式重训，未使用旧页面权重初始化。三组共用固定划分与种子 42，按验证集准确率选模后分别评估全部 378 张测试图片。运行标识见 `experiment_protocol.json`。", "",
             "| 组别 | 训练轮数 | 最佳轮次 | 验证准确率 | 测试准确率 | Macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in sorted(report["experiments"].items()):
        meta = metrics["training_metadata"]
        lines.append(f"| {name} | {metrics['epochs_trained']} | {meta['epoch']} | {meta['validation_accuracy']:.2%} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} |")
    lines.extend(["", f"验证集选出的页面模型为 **{selected} 组**。测试指标没有参与选模或调参。这是固定划分、单随机种子的比较；在看过旧测试结果后进行的重训复核，不是新增的盲测或多次随机种子实验。", "",
                  "三组最佳权重均保存在 `checkpoints/<A|B|C>/best.pth`；每组的实际训练源码保存在 `training_logs/<A|B|C>/source/`。交付权重、源码、日志、指标与图表的大小及 SHA-256 见 `artifact_manifest.json`；随后生成的独立复算与页面检查报告不计入该清单。", "",
                  "旧 `ppt_training` 指标、图表与日志保存在 `../archive/ppt_training/`，原权重未找回，不能把新权重写成旧检查点。旧协议的 `model.py` 从 Git 提交 `dddecc6` 精确找回；与当前版本的差异仅是预训练下载重试和 CPU 加载逻辑，网络结构不变。", "",
                  "新克隆仓库无需训练即可独立加载三组权重、逐图复算测试指标：", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe verify_results.py", ".\\.venv\\Scripts\\python.exe plot_results.py --archived", "```", ""])
    (final / "comparison_notes.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if promote:
        current = ROOT / "checkpoints/best_resnet18.pth"
        if current.exists():
            backup = current.parent / "legacy" / f"best_resnet18_{sha256(current)[:8]}.pth"
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                shutil.copyfile(current, backup)
        source = final / "checkpoints" / selected / "best.pth"
        temporary = current.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, current)
        metrics = report["experiments"][selected]
        digest = sha256(current)
        text = f"""# 已验证的页面模型

`best_resnet18.pth` 与 `results/final/checkpoints/{selected}/best.pth` 完全相同，来自 `{metrics['training_metadata']['run_id']}` 的 {selected} 组最佳检查点，按验证集准确率选择。

- 文件大小：{current.stat().st_size:,} 字节。
- SHA-256：`{digest}`。
- 最佳轮次：{metrics['training_metadata']['epoch']}；验证准确率：{metrics['training_metadata']['validation_accuracy']:.2%}。
- 固定测试集：{metrics['samples']} 张，准确率 {metrics['accuracy']:.2%}，Macro-F1 {metrics['macro_f1']:.4f}。
- 参数字段：`model_state`；来源字段：`training_metadata`，包含配置、数据身份、实际训练源码哈希与初始化权重哈希。
- 类别：`cardboard, glass, metal, paper, plastic, trash`。
- 输入：RGB、224×224、ImageNet 标准化，与 `app.py` 一致；输出 Softmax Top-3。

页面仅在权重 SHA-256 与正式测试记录吻合时显示准确率；替换成其他权重后会显示待验证的演示模型说明。固定测试集准确率不等于每张上传图片都能达到同样表现。

原页面权重实际大小为 **44,794,315 字节**，SHA-256 为 `7e95464ce6928e68b9730d202f7fd372c7016a22775d60a5f367f5dd9888a3d2`，缺少完整训练来源；本机备份保存在 `checkpoints/legacy/`（不提交）。旧 C 组的 92.86% 仅为旧实验历史记录，不作为该旧页面模型的准确率。

三组已交付权重、源码及日志均在 `results/final/`。运行 `python verify_results.py` 可重新加载所有最佳权重并验证全部测试预测。
"""
        (current.parent / "README.md").write_text(text, encoding="utf-8", newline="\n")
    artifacts = {}
    for path in sorted(final.rglob("*")):
        if path.is_file() and path.name not in ("artifact_manifest.json", "evaluation_verification.json", "ui_verification.json"):
            artifacts[path.relative_to(final).as_posix()] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    atomic_json(final / "artifact_manifest.json", {"created_at": now(), "files": artifacts})
    print(f"Archived A/B/C weights, source and logs. Selected by validation: {selected}; promoted={promote}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--promote", action="store_true", help="按验证准确率选择并备份、更新页面模型")
    args = parser.parse_args()
    archive_results(args.results_root, args.promote)
