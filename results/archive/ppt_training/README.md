# 旧 ppt_training 历史记录

此目录保留 2026-10-25 的原始指标、图表、训练日志和协议。A/B/C 最佳权重未在本机及 OneDrive 同步项目中找到，因此这些文件作为历史记录保存；新模型另用 `audit_retrain_20261028` 完整重训并测试，不冒充旧检查点。

`source/` 中的五份训练源码已从 Git 历史精确找回，逐文件 SHA-256 与原 `experiment_protocol.json` 完全一致。原 `model.py` 来自提交 `dddecc6`；后续 `255c31f` 仅增加预训练权重下载重试与 CPU 加载，没有修改网络结构。源码来源及哈希见 `recovery_audit.json`。

原文件与该目录新增的恢复说明分开保留；`validation_baseline.json` 和 `finetuning_history.json` 是更早的探索记录，不属于 A/B/C 对照实验。旧页面权重大小为 44,794,315 字节，缺少完整训练来源，不能套用旧 C 组的 92.86%。当前可重新加载验证的模型与结果见 `../../final/`。
