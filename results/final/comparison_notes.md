# A/B/C 实验测试结果

本次为正式重训，未使用旧页面权重初始化。三组共用固定划分与种子 42，按验证集准确率选模后分别评估全部 378 张测试图片。运行标识见 `experiment_protocol.json`。

| 组别 | 训练轮数 | 最佳轮次 | 验证准确率 | 测试准确率 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| A | 5 | 5 | 79.10% | 82.54% | 0.8115 |
| B | 19 | 14 | 83.33% | 85.45% | 0.8519 |
| C | 20 | 20 | 90.74% | 92.86% | 0.9257 |

验证集选出的页面模型为 **C 组**。测试指标没有参与选模或调参。这是固定划分、单随机种子的比较；在看过旧测试结果后进行的重训复核，不是新增的盲测或多次随机种子实验。

三组最佳权重均保存在 `checkpoints/<A|B|C>/best.pth`；每组的实际训练源码保存在 `training_logs/<A|B|C>/source/`。交付权重、源码、日志、指标与图表的大小及 SHA-256 见 `artifact_manifest.json`；随后生成的独立复算与页面检查报告不计入该清单。

旧 `ppt_training` 指标、图表与日志保存在 `../archive/ppt_training/`，原权重未找回，不能把新权重写成旧检查点。旧协议的 `model.py` 从 Git 提交 `dddecc6` 精确找回；与当前版本的差异仅是预训练下载重试和 CPU 加载逻辑，网络结构不变。

新克隆仓库无需训练即可独立加载三组权重、逐图复算测试指标：

```powershell
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe plot_results.py --archived
```
