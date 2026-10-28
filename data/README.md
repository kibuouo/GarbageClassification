
## 固定划分

在项目根目录运行 `python split.py`，按种子 42、约 70% / 15% / 15% 分别生成 `train/`、`val/`、`test/`。这些图片是原始数据的复制件，不重复提交到 Git。`split.csv` 记录每张图片的分区、类别和文件名。

原始数据中发现以下三组完全相同、类别标注却不同的图片。原文件完整保留在 `raw/`，脚本暂不把这六张图片纳入任何分区：

| 第一张 | 相同内容的另一张 |
| --- | --- |
| glass/glass115.jpg | metal/metal91.jpg |
| glass/glass176.jpg | plastic/plastic152.jpg |
| glass/glass389.jpg | plastic/plastic332.jpg |

最终训练集 1,765 张、验证集 378 张、测试集 378 张。训练时仅对训练集做随机增强；验证集和测试集保持固定预处理。六类图像标签不直接对应生活垃圾四分类。

## 跨平台清单身份

`split.csv` 保留 UTF-8 BOM，统一使用 LF 换行；`.gitattributes` 和 `split.py` 都固定这一规则。审计采用 `sha256-lf-v1`：先将 CRLF/CR 换行规范为 LF，再计算清单 SHA-256；同时单独保存文件原始字节哈希。规范哈希为 `17ae4d9a032b92cd483b84f84beb97aef85b5f208b34ab3029dd4155ba714aaa`。

恢复训练、评估旧权重及合并结果共用同一兼容验证。没有算法版本的旧记录，可匹配由当前清单重建出的 LF 或 CRLF 原始字节哈希（旧 Windows 哈希为 `90c7aa0d616cbcf2ccf7d40f7520d981c83a0ec9d1cd8bf5b92d595aab519634`）。兼容验证始终要求图片内容与划分数据的综合哈希一致，因此不会放过真实的数据或划分变化；未知的算法版本也会拒绝。
