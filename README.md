# 垃圾图像分类

人工智能课程项目，使用 ResNet-18 进行垃圾图像分类，Streamlit 提供上传图片、摄像头拍照和预测展示。

## 项目结构

```text
GarbageClassification/
├── app.py             # 页面、图片处理、模型加载和预测
├── model.py           # ResNet-18 模型定义
├── train.py           # 模型训练
├── split.py           # 数据集划分
├── requirements.txt   # 依赖
├── README.md
├── data/
│   ├── raw/dataset-resized/  # 已提交：TrashNet 原始图片
│   ├── split.csv            # 已提交：固定划分清单
│   ├── TRASHNET_LICENSE     # 原始数据来源仓库的许可
│   └── README.md            # 数据来源与划分说明
├── checkpoints/
│   ├── best_resnet18.pth    # 已提交：当前可用模型
│   └── README.md           # 模型格式与校验信息
└── results/                # 成果说明，最终图表/报告放 final/
```

仓库包含运行页面所需的代码和模型，以及复现训练所需的一份原始数据。`data/train/`、`data/val/`、`data/test/` 由划分脚本生成，不重复提交；原始 ZIP、虚拟环境和缓存也不提交。

## 启动页面

```bash
git clone https://github.com/kibuouo/GarbageClassification.git
cd GarbageClassification
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

页面仅保留图片输入与预览、预测结果、Top-3 概率。左侧选择“上传图片”或“摄像头拍照”，取得图片后点击“开始预测”，右侧查看结果。

摄像头使用方式：在本机打开 `http://localhost:8501`，切换到“摄像头拍照”，允许浏览器访问摄像头，拍下照片后点击“开始预测”。照片不会额外保存到项目目录。

仓库已提供 `checkpoints/best_resnet18.pth`，安装依赖后可以直接打开页面进行识别，无需先下载数据或重新训练。

## 复现数据划分和训练

原始数据来自 [TrashNet](https://github.com/garythung/trashnet)，位于 `data/raw/dataset-resized/`，共 2,527 张、六个类别。来源与原许可见 [数据说明](data/README.md) 和 [TRASHNET_LICENSE](data/TRASHNET_LICENSE)。

首次克隆并安装依赖后，可以运行：

```bash
python split.py
python train.py
```

`split.py` 使用固定种子 42，在每个类别内随机划分。跳过六张内容重复且标签冲突的图片后，生成训练集 1,765 张、验证集 378 张、测试集 378 张，并写出 `data/split.csv`。相同原始图片和脚本应得到仓库中同一份划分清单。目标目录已有文件时，脚本会停止以避免混合不同划分；已经完成划分的组员直接运行 `python train.py` 即可。

`train.py` 默认使用 ImageNet 预训练的 ResNet-18，冻结主干，训练分类层 5 轮，batch size 16、学习率 0.001，优先使用 CUDA、其次 Apple MPS、最后 CPU。首次重新训练可能下载 ImageNet 权重，需要网络。预处理和类别顺序与页面一致；不同设备和软件版本可能产生数值差异。

每轮在验证集上检查准确率，保存最佳模型到 `checkpoints/best_resnet18.pth`。重新训练会更新此文件，原有版本可以通过 Git 历史取回。测试集留作最终评估；验证准确率不等于测试集准确率。

最终图表、报告和指标可放在 `results/final/` 后提交，暂存日志等其他运行产物保持本地。当前仓库没有预填实验图表或准确率。

## TODO

- [ ] 开放实时摄像头识别：实现代码和 `streamlit-webrtc` 依赖已保留，当前前端入口隐藏。后续在 `app.py` 中将 `ENABLE_LIVE_CAMERA` 设为 `True`，即可在现有两种输入方式之外显示“实时摄像头”。
- [ ] 开放前确认真实摄像头的开始/停止、连续推理性能和浏览器兼容性。

保留的实时功能最多每 0.5 秒进行一次推理，不启用麦克风、不录制视频。本地连接不使用外部 STUN/TURN 服务；远程部署还需配置 HTTPS 和适当的 WebRTC 网络连接。当前默认页面不会创建实时视频组件或启动定时结果刷新。

## 模型使用

没有模型时，预测按钮禁用，右侧显示明确标注的固定演示数据，与上传图片无关。有本地权重后自动使用真实预测，无需切换模式。

代码带中文注释。数据准备和训练脚本独立运行，页面加载已有权重进行预测。

## 模型格式

当前模型和后续训练均使用 `checkpoints/best_resnet18.pth`，内容包含：

- `model_state`：ResNet-18 权重。
- `classes`：与输出层编号一致的类别名称列表。

`app.py` 中的预测使用 RGB、224×224 和 ImageNet 标准化，后续训练需使用相同的预处理约定。`model.py` 中的 `build_model()` 默认不下载预训练权重，预测使用 CPU。有模型后，通过同一个页面进行识别。

`model.py` 中的 `GarbageClassifier` 展开了 ResNet-18 的前向计算，并用中文注释标出各阶段的张量尺寸。基础调用方式：

```python
from model import build_model

model = build_model(num_classes=6)  # 仅创建网络，不代表已经学会分类。
model.eval()  # 用于预测；页面会自动调用。
```

后续进行迁移学习时可使用 `build_model(pretrained=True, freeze_backbone=True)`，加载 ImageNet 参数并只训练分类层。冻结主干时，BatchNorm 统计量也保持固定；调用 `model.freeze_features(False)` 可恢复全网络梯度。加载已有本地模型时继续使用默认 `pretrained=False`，避免重复下载。模型输出为类别分数，Softmax 仍由页面执行。

预期使用 TrashNet 六类图像标签；它们不等同于生活垃圾四分类。
