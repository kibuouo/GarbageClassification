# 已验证的页面模型

`best_resnet18.pth` 与 `results/final/checkpoints/C/best.pth` 完全相同，来自 `audit_retrain_20261028` 的 C 组最佳检查点，按验证集准确率选择。

- 文件大小：44,795,851 字节。
- SHA-256：`ff863443824f79e8c3b56d000c9a5eaaabe1beee1e143c7a3b50cfc2bf92ed5b`。
- 最佳轮次：20；验证准确率：90.74%。
- 固定测试集：378 张，准确率 92.86%，Macro-F1 0.9257。
- 参数字段：`model_state`；来源字段：`training_metadata`，包含配置、数据身份、实际训练源码哈希与初始化权重哈希。
- 类别：`cardboard, glass, metal, paper, plastic, trash`。
- 输入：RGB、224×224、ImageNet 标准化，与 `app.py` 一致；输出 Softmax Top-3。

页面仅在权重 SHA-256 与正式测试记录吻合时显示准确率；替换成其他权重后会显示待验证的演示模型说明。固定测试集准确率不等于每张上传图片都能达到同样表现。

原页面权重实际大小为 **44,794,315 字节**，SHA-256 为 `7e95464ce6928e68b9730d202f7fd372c7016a22775d60a5f367f5dd9888a3d2`，缺少完整训练来源；本机备份保存在 `checkpoints/legacy/`（不提交）。旧 C 组的 92.86% 仅为旧实验历史记录，不作为该旧页面模型的准确率。

三组已交付权重、源码及日志均在 `results/final/`。运行 `python verify_results.py` 可重新加载所有最佳权重并验证全部测试预测。
