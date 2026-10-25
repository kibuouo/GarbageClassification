# 课程项目数据

本仓库提供一份 TrashNet 缩小版原始图片，便于组员在相同数据上协作和复现训练。

## 来源

- 原始项目：[garythung/trashnet](https://github.com/garythung/trashnet)，作者 Gary Thung、Mindy Yang。
- 本地目录：`raw/dataset-resized/`。
- 原始下载文件：[dataset-resized.zip](https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip)。
- 原始 ZIP 的 Git blob SHA：`4184b4e9d359026a49e5d34ac2cead944f53be88`。
- 原始 ZIP 的 SHA-256：`0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`。
- 原始仓库许可随附于 [TRASHNET_LICENSE](TRASHNET_LICENSE)，请保留来源和许可说明。

图片保持原始内容，未为本项目重新编码。原始 ZIP 不重复放进 Git。

| 类别文件夹 | 中文名称 | 原始图片数量 |
| --- | --- | ---: |
| cardboard | 纸板 | 403 |
| glass | 玻璃 | 501 |
| metal | 金属 | 410 |
| paper | 纸张 | 594 |
| plastic | 塑料 | 482 |
| trash | 其他垃圾 | 137 |
| 合计 | | 2,527 |

## 固定划分

在项目根目录运行 `python split.py`，按种子 42、约 70% / 15% / 15% 分别生成 `train/`、`val/`、`test/`。这些图片是原始数据的复制件，不重复提交到 Git。`split.csv` 记录每张图片的分区、类别和文件名。

原始数据中发现以下三组完全相同、类别标注却不同的图片。原文件完整保留在 `raw/`，脚本暂不把这六张图片纳入任何分区：

| 第一张 | 相同内容的另一张 |
| --- | --- |
| glass/glass115.jpg | metal/metal91.jpg |
| glass/glass176.jpg | plastic/plastic152.jpg |
| glass/glass389.jpg | plastic/plastic332.jpg |

最终训练集 1,765 张、验证集 378 张、测试集 378 张。训练时仅对训练集做随机增强；验证集和测试集保持固定预处理。六类图像标签不直接对应生活垃圾四分类。
