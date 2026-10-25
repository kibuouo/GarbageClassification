# 垃圾图像分类

## 项目结构

```text
GarbageClassification/
├── app.py             # 页面与预测
├── model.py           # ResNet-18 模型
├── train.py           # 模型训练
├── split.py           # 数据集划分
├── requirements.txt   # 依赖
├── data/              # 原始图片、划分清单与来源说明
├── checkpoints/       # 已训练模型 best_resnet18.pth
└── results/           # 最终图表、指标和报告放入 final/
```

## 运行方法

仓库已包含训练好的模型，安装依赖后即可使用。

```bash
git clone https://github.com/kibuouo/GarbageClassification.git
cd GarbageClassification
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell 激活虚拟环境使用 `.venv\Scripts\Activate.ps1`。

打开 `http://localhost:8501`，上传图片或允许摄像头访问并拍照，再点击“开始预测”，查看分类结果和 Top-3 概率。

## 实时摄像头功能 TODO

- [ ] 开放实时摄像头识别入口：实现代码已保留，当前前端隐藏。
- [ ] 验证真实摄像头的开始、停止、连续识别性能和浏览器兼容性。
