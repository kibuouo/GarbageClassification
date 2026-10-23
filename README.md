# 基于迁移学习的垃圾图像分类与智能识别系统

人工智能课程项目框架：TrashNet 数据检查与固定分层划分 → ResNet-18 三组对比实验 → 测试集评估 → Streamlit 演示。

代码已连接数据准备、训练、评估和推理流程。真实数据、模型权重和实验结果需要后续实际运行产生；仓库不提供预设准确率或示例实验结果。

## 目录

```text
GarbageClassification/
├── configs/                    # A/B/C 实验参数，JSON
│   ├── A_scratch.json          # 随机初始化，训练全网络
│   ├── B_frozen.json           # ImageNet 预训练，仅训练 fc
│   └── C_finetune.json         # 先训练 fc，再解冻 layer4
├── garbage_classifier/         # 可复用业务代码
│   ├── common.py               # 类别、路径、种子、设备、文件输出
│   ├── config.py               # 配置读取和校验
│   ├── data.py                 # 共享预处理、DataLoader、划分校验
│   ├── engine.py               # 训练阶段、早停、最佳模型和曲线
│   ├── evaluation.py           # 测试集指标、混淆矩阵、误分类记录
│   ├── inference.py            # CLI 和 Web 共用推理接口
│   └── comparison.py           # 已完成实验的汇总和可比性检查
├── models/                     # ResNet-18 构建和参数冻结策略
├── data/{raw,train,val,test}/   # 原始数据和固定划分，本地生成
├── checkpoints/<实验名>/       # best_resnet18.pth，本地生成
├── results/<实验名>/           # 日志、图表和测试指标，本地生成
├── docs/                       # 架构、实验规约和交付计划
├── tests/                      # 临时合成图片上的流程检查
├── prepare_data.py             # 数据检查、固定划分和类别统计
├── train.py                    # 单组训练
├── evaluate.py                 # 单组测试评估
├── run_experiments.py           # 顺序执行 A/B/C 并评估汇总
├── compare_experiments.py       # 直接汇总已有结果
├── predict.py                  # 单张图片预测
├── app.py                      # Streamlit 识别界面
└── requirements.txt
```

所有脚本的相对路径均以项目根目录为基准。以下命令在项目根目录执行。

## 1. 开发环境

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

建议使用 Python 3.11 或更新版本及支持它的 PyTorch 版本。CUDA 机器请根据本机驱动和 CUDA 条件安装匹配的 PyTorch。`auto` 设备按 CUDA → Apple MPS → CPU 的顺序选择；命令行可指定 `--device cpu`、`--device mps` 或 `--device cuda`。

## 2. 准备数据

从 [TrashNet 原始项目](https://github.com/garythung/trashnet) 获取数据并自行解压。脚本读取本地文件，不自动下载数据。推荐布局：

```text
data/raw/dataset-resized/
├── cardboard/*.jpg
├── glass/*.jpg
├── metal/*.jpg
├── paper/*.jpg
├── plastic/*.jpg
└── trash/*.jpg
```

```bash
python prepare_data.py --source data/raw/dataset-resized
```

按类别固定种子 42 分成约 70% / 15% / 15%，每类至少需要三张图片；小类别按整数取整，因此比例可能略有偏差。保留原始文件并复制到 `data/train`、`data/val`、`data/test`。检查损坏图片、六类目录和内容重复图片；遇到重复时停止并指出文件，需人工检查。原始目录须直接包含六个类别，图片放在类别目录的第一层。

输出 `data/split_manifest.json`、`results/dataset/class_distribution.csv` 和类别分布图。清单记录文件 SHA-256 与所属分区，训练和评估前检查样本没有被增加、删除或替换。三组实验共用此清单。输出目录非空时停止，避免覆盖已有划分；需要另一版划分可指定新的 `--output`，并同步调整配置中的 `data_dir`。

## 3. 训练

建议先跑冻结实验，确认整条流程：

```bash
python train.py --config configs/B_frozen.json
python evaluate.py --checkpoint checkpoints/B_frozen/best_resnet18.pth
```

再分别执行 A、C，或一次顺序执行三组：

```bash
python run_experiments.py
```

迁移学习实验首次运行可能下载 ImageNet 预训练权重。默认 224×224、batch size 32、AdamW、最多 20 轮、早停 patience 7，按验证集 Macro-F1 保存最佳模型。C 的前 5 轮训练 fc，再训练 fc 与 layer4，layer4 使用较低学习率；冻结主干的 BatchNorm 统计量保持不变。`checkpoints/<实验名>/best_resnet18.pth` 同时记录类别顺序、预处理参数、配置与划分标识。

结果按实验名隔离；已有输出时不会覆盖。重跑时复制修改配置或使用 `python train.py --config configs/B_frozen.json --name B_frozen_run2`。

## 4. 评估与实验对比

每个实验在 `results/<实验名>/` 生成：

- `config.json`、`training_history.csv`、`train_summary.json`：参数、逐轮 Loss/Accuracy/Macro-F1、阶段、耗时及最佳轮次。
- `loss_curve.png`、`accuracy_curve.png`：真实训练曲线。
- `test_metrics.json`：Accuracy、Macro-F1、Macro-Precision、Macro-Recall 和逐类别报告。
- `confusion_matrix.png` / `.json`：混淆矩阵和数值。
- `test_predictions.csv`、`misclassified.csv`：逐图片预测和错误样例索引。

```bash
python compare_experiments.py results/A_scratch results/B_frozen results/C_finetune
```

汇总到 `results/comparison/experiment_comparison.csv` 和 `.json`。对比前检查划分、种子、图像尺寸、batch size、最大轮次、早停预算和设备一致。测试集只用于最终评价；模型选择依据验证集。耗时包括每轮训练、验证和日志/模型保存，不包含初始权重下载和训练结束后的绘图；跨设备耗时不可直接比较。

## 5. 智能识别

```bash
python predict.py path/to/image.jpg --checkpoint checkpoints/B_frozen/best_resnet18.pth
streamlit run app.py
```

页面会加载所选模型，支持图片上传、预览、Top-3 概率、处理耗时、低置信度提示和会话记录 CSV 导出。没有模型时会明确提示准备数据和训练。历史记录保存在当前会话内，长期保存请导出 CSV。只加载本项目训练或可信来源的权重文件。

六类是 TrashNet 图像标签，不能直接映射为中国生活垃圾四分类。低置信度提示不是可靠的未知类别检测；真实场景泛化需要另外采集环境照片评估。

## 检查和后续交付

```bash
python -m unittest discover -s tests -v
```

检查使用临时合成图片与随机初始化模型，验证数据划分、冻结规则、训练到评估和推理流程；合成数据结果不作为课程实验成绩。架构和实验规范见 [docs/architecture.md](docs/architecture.md)，四周计划及验收项见 [docs/project_plan.md](docs/project_plan.md)。报告、PPT、演示视频和摄像头/Grad-CAM 扩展留到获得真实模型和实验结果后完成。
