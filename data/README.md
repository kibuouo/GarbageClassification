
## 固定划分

在项目根目录运行 `python split.py`，按种子 42、约 70% / 15% / 15% 分别生成 `train/`、`val/`、`test/`。这些图片是原始数据的复制件，不重复提交到 Git。`split.csv` 记录每张图片的分区、类别和文件名。

原始数据中发现以下三组完全相同、类别标注却不同的图片。原文件完整保留在 `raw/`，脚本暂不把这六张图片纳入任何分区：

| 第一张 | 相同内容的另一张 |
| --- | --- |
| glass/glass115.jpg | metal/metal91.jpg |
| glass/glass176.jpg | plastic/plastic152.jpg |
| glass/glass389.jpg | plastic/plastic332.jpg |

最终训练集 1,765 张、验证集 378 张、测试集 378 张。训练时仅对训练集做随机增强；验证集和测试集保持固定预处理。六类图像标签不直接对应生活垃圾四分类。
