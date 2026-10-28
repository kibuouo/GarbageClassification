# 课程成果

当前正式交付为 `audit_retrain_20261028`：三组从 ImageNet 初始化，按验证集选模后在同一份 378 张固定测试图片上评估。

| 实验 | 训练轮数 | 最佳轮次 | 测试准确率 | Macro-F1 |
|---|---:|---:|---:|---:|
| A | 5 | 5 | 82.54% | 0.8115 |
| B | 19 | 14 | 85.45% | 0.8519 |
| C | 20 | 20 | 92.86% | 0.9257 |

- `final/checkpoints/<A|B|C>/best.pth`：三组实际最佳权重，SHA-256 与对比表一致。
- `final/training_logs/<A|B|C>/`：配置、元数据、逐轮历史、训练日志、完成状态及 `source/` 实际源码。
- `final/comparison.csv`、`test_metrics.json`：汇总与全部逐图预测。
- `final/artifact_manifest.json`：交付文件大小及 SHA-256。
- `final/evaluation_verification.json`：重新加载归档权重，对全部测试预测、混淆矩阵和指标独立复算的结果。
- `final/data_identity_verification.json`：旧 LF/CRLF 元数据兼容验证摘要。
- `final/environment.json`：本次环境版本。

新克隆仓库可运行 `python verify_results.py`，无需本地 `runs/` 即可复算全部测试结果。`python plot_results.py --archived` 从交付日志重绘到 `replotted/`，不改写交付图表。

训练流程仍将每轮恢复点和优化器写入 `runs/<实验>/<run-id>/`，默认不提交中间检查点。训练、评估、绘图后运行 `python archive_results.py --promote` 可保存三组最佳权重与源码，并按验证准确率选择页面模型；此前页面权重会备份到 `checkpoints/legacy/`。

旧 `ppt_training` 原权重未找回；原始结果、图表、日志与协议逐字节保留在 `archive/ppt_training/`，五份训练源码也已从 Git 精确恢复。旧指标作为历史记录，不冒充本次模型文件。较早的 `validation_baseline.json`、`finetuning_history.json` 也在该历史目录。

`smoke_crossplatform/` 仅为换行兼容和恢复流程的小样本检查，不属于正式实验结果。

本次使用同一固定划分与种子复核旧结果，没有依据测试表现重新调参；不能据此声称多次随机种子运行或新盲测都能达到同样表现。
