# 已训练模型

`best_resnet18.pth` 是当前页面可直接加载的 ResNet-18 六分类权重。它随代码提交，组员安装依赖后即可演示。

- 文件大小：44,792,971 字节。
- 文件版本请以当前权重的 SHA-256 为准，可用 PowerShell `Get-FileHash checkpoints/best_resnet18.pth -Algorithm SHA256` 查看；本次 A/B/C 结果提交没有更新页面权重。
- 权重字段：`model_state`。
- 类别字段：`classes`，按 `cardboard, glass, metal, paper, plastic, trash` 排列。
- 输入：RGB 图片，224×224，使用与 `app.py` 相同的 ImageNet 标准化参数。
- 输出：六个类别分数，页面使用 Softmax 转换为概率后显示 Top-3。

模型结构由 `model.py` 定义。现有页面权重仅保存参数和类别映射，课程报告中的轮次与指标应以实际训练记录为依据。新版 `train.py` 支持 A/B/C 实验，分别将最佳模型保存到 `results/runs/<实验>/<run-id>/best.pth`，并记录配置、数据哈希、日志和验证指标；默认不会覆盖本目录中的页面权重。

选择新的页面模型时，先备份现有权重，再将已确认的实验 `best.pth` 复制为本目录的 `best_resnet18.pth`，并更新本说明中的文件大小与 SHA-256。仅在需要共享新的可用模型时提交权重更新，避免提交各轮中间文件。
