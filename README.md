# 垃圾图像分类

## 项目结构

```text
GarbageClassification/
├── app.py                      # Streamlit 页面与图片预测
├── model.py                    # ResNet-18 网络定义
├── train.py                    # 训练分类层、验证并保存模型
├── split.py                    # 将原始图片划分为训练、验证和测试集
├── requirements.txt            # 运行依赖
├── data/
│   ├── raw/dataset-resized/     # 六类垃圾原始图片
│   ├── split.csv               # 固定划分清单
│   ├── README.md               # 数据来源与划分说明
│   └── TRASHNET_LICENSE        # 原始数据许可
├── checkpoints/
│   └── best_resnet18.pth        # 页面加载的已训练模型
└── results/                    # 成果存放说明，最终文件放入 final/
```

划分数据后，会在 `data/` 下生成 `train/`、`val/` 和 `test/`。这些目录中的图片由原始数据复制得到，不重复提交。

## 运行方法

以下命令在项目根目录执行。仓库包含原始数据和已训练模型，演示识别时只需安装依赖、启动页面。

**首次安装**

```bash
git clone https://github.com/kibuouo/GarbageClassification.git
cd GarbageClassification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 将激活命令替换为 `.venv\Scripts\Activate.ps1`。



**启动页面**

```bash
streamlit run app.py
```

在浏览器打开终端显示的地址，通常为 `http://localhost:8501`。选择“上传图片”或“摄像头拍照”，取得图片后点击“开始预测”，右侧显示预测类别和 Top-3 概率。

支持纸板、玻璃、金属、纸张、塑料、其他垃圾六类图像标签，不直接对应生活垃圾四分类。预测使用 CPU，拍照需要允许浏览器访问摄像头。

**需要重新训练时**

```bash
python split.py
python train.py
```

`split.py` 使用种子 42 生成训练集 1,765 张、验证集 378 张、测试集 378 张，并写出划分清单。目标目录已有文件时会停止；已完成划分则直接运行 `python train.py`。数据来源和划分细节见 [data/README.md](data/README.md)。

`train.py` 默认加载 ImageNet 预训练参数，冻结 ResNet-18 主干，训练分类层 5 轮，batch size 为 16，学习率为 0.001。首次训练可能下载预训练权重，训练设备自动按 CUDA、MPS、CPU 的顺序选择。修改训练参数请编辑 `train.py` 顶部的常量，当前没有命令行配置参数。

训练和验证的损失、准确率输出到终端，按验证准确率保存最佳模型到 `checkpoints/best_resnet18.pth`。重新训练会覆盖此文件，请按需备份；训练完成后重新启动页面使用新模型。当前未提供测试集评估、对比实验和自动绘图功能。

## 实时摄像头功能 TODO
