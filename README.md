# 基于迁移学习的垃圾图像分类与智能识别系统

人工智能课程项目，使用 ResNet-18 对垃圾图片进行六分类，通过 Streamlit 提供图片上传、摄像头拍照和 Top-3 概率展示。

仓库已包含 TrashNet 原始图片、固定划分清单和可加载的模型权重。安装依赖后即可启动页面；需要重新训练时，再执行数据划分和训练。

## 1. 项目结构

```text
GarbageClassification/
├── app.py                      # 页面、图片预处理、模型加载和预测
├── model.py                    # ResNet-18 网络结构与主干冻结逻辑
├── train.py                    # 训练、验证及最佳模型保存
├── split.py                    # 固定随机种子划分数据集
├── requirements.txt            # Python 依赖
├── data/
│   ├── raw/dataset-resized/     # 已提交：六类原始图片
│   ├── split.csv               # 已提交：图片划分清单
│   ├── TRASHNET_LICENSE        # 原始数据来源仓库的许可
│   ├── README.md               # 数据来源与划分说明
│   └── train/、val/、test/      # 运行 split.py 后生成，不重复提交
├── checkpoints/
│   ├── best_resnet18.pth        # 已提交：页面使用的模型权重
│   └── README.md               # 模型格式及校验信息
└── results/
    ├── README.md               # 成果存放说明
    └── final/                  # 后续存放最终图表、指标和报告
```

`results/final/` 在需要保存成果时创建。原始 ZIP、虚拟环境、缓存和临时运行文件不提交。以下命令均在项目根目录执行。

## 2. 安装依赖

首次获取项目：

```bash
git clone https://github.com/kibuouo/GarbageClassification.git
cd GarbageClassification
```

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

后续运行项目前，先激活已有虚拟环境即可。

## 3. 启动识别页面

```bash
streamlit run app.py
```

打开终端提示的本机地址，通常是 `http://localhost:8501`。

1. 在左侧选择“上传图片”或“摄像头拍照”。
2. 上传图片，或允许浏览器使用摄像头并拍下照片。
3. 点击“开始预测”，在右侧查看预测类别和 Top-3 概率。

上传支持 JPG、JPEG、PNG 和 WebP。页面使用 CPU 加载 `checkpoints/best_resnet18.pth`，不需要先重新训练，也不会额外保存上传或拍摄的照片。

如果模型文件缺失，预测按钮会禁用，右侧显示明确标注的固定演示数据；该数据与输入图片无关。模型文件损坏或格式不匹配时，页面会显示加载或预测错误。

六类标签为纸板（cardboard）、玻璃（glass）、金属（metal）、纸张（paper）、塑料（plastic）和其他垃圾（trash），不直接对应生活垃圾四分类。真实环境中的背景、光照和拍摄角度可能影响识别结果。

## 4. 数据划分

原始数据来自 [TrashNet](https://github.com/garythung/trashnet)，已放在 `data/raw/dataset-resized/`，共 2,527 张图片。来源、原许可及重复图片说明见 [数据说明](data/README.md)。

需要重新训练且尚未划分数据时，运行：

```bash
python split.py
```

脚本保留原始图片，按类别使用固定种子 42，以约 70% / 15% / 15% 复制到训练集、验证集和测试集。跳过六张内容相同但类别标签冲突的图片后，得到：

| 分区 | 目录 | 图片数量 |
| --- | --- | ---: |
| 训练集 | `data/train/` | 1,765 |
| 验证集 | `data/val/` | 378 |
| 测试集 | `data/test/` | 378 |

`data/split.csv` 记录分区、类别和文件名。相同原始图片与脚本会生成相同划分。任一目标目录已有文件时，脚本会停止；已经划分好的数据直接用于训练，无需重复运行。

当前训练脚本读取训练集和验证集，测试集留作后续最终评估。

## 5. 模型训练

完成数据划分后运行：

```bash
python train.py
```

当前训练方式是在 ImageNet 预训练的 ResNet-18 上冻结主干，只更新最后的六分类层。首次训练可能需要下载 ImageNet 权重。设备自动按 CUDA → Apple MPS → CPU 的顺序选择。

| 配置 | 当前值 |
| --- | --- |
| 图片尺寸 | 224 × 224 |
| 训练轮数 | 5 |
| Batch size | 16 |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| 损失函数 | 交叉熵 |
| 模型保存依据 | 验证集准确率提升 |

训练集使用随机水平翻转，验证集不做随机增强，两者均使用 ImageNet 标准化。训练轮数、批次大小和学习率在 `train.py` 顶部的 `EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE` 中修改；当前脚本没有命令行配置参数。

终端每轮输出训练和验证的损失、准确率，最佳模型保存到 `checkpoints/best_resnet18.pth`。每次启动训练都会重新加载 ImageNet 参数，不会从现有分类模型续训；保存时会覆盖同名权重。需要保留旧模型时，请先备份。

当前尚未实现独立测试集评估、对比实验、训练曲线和混淆矩阵自动生成。验证准确率不能作为测试集成绩；课程报告中的指标应以实际计算结果为准，最终成果可放入 `results/final/`。

## 6. 模型与预测约定

`model.py` 定义网络结构和前向计算，`train.py` 负责更新参数，`app.py` 加载训练结果进行预测。直接运行 `model.py` 不会开始训练。

模型权重包含 `model_state` 和 `classes` 两个字段，类别顺序由训练集文件夹名称确定。页面按权重中的顺序解释输出，使用 RGB、224 × 224 和 ImageNet 标准化预处理图片，再对模型输出应用 Softmax，取概率最高的三个类别。

模型格式和当前权重的校验信息见 [模型说明](checkpoints/README.md)。权重未保存实际训练轮次和准确率，不能仅凭权重文件推断这些成绩。

## 7. 实时摄像头功能 TODO

- [ ] 验证真实摄像头的开始、停止、连续推理性能和浏览器兼容性。
- [ ] 完成验证后开放实时识别入口：实现代码和 `streamlit-webrtc` 依赖已保留，`app.py` 中的 `ENABLE_LIVE_CAMERA` 当前为 `False`，前端保持拍照模式。
