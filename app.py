"""垃圾图像分类页面。启动方式：streamlit run app.py。"""

import pickle
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from model import build_model
from model_evidence import model_evidence

# 相对项目目录查找权重，避免启动位置不同导致找不到文件。
ROOT = Path(__file__).resolve().parent
EFFICIENT_CHECKPOINT = ROOT / "checkpoints" / "best_efficientnet_b0.pth"
CHECKPOINT = EFFICIENT_CHECKPOINT if EFFICIENT_CHECKPOINT.is_file() else ROOT / "checkpoints" / "best_resnet18.pth"
TEST_REPORT = (ROOT / "results/efficientnet_b0/final/test_metrics.json" if CHECKPOINT == EFFICIENT_CHECKPOINT
               else ROOT / "results/final/test_metrics.json")
LABELS = {
    "cardboard": "纸板",
    "glass": "玻璃",
    "metal": "金属",
    "paper": "纸张",
    "plastic": "塑料",
    "trash": "其他垃圾",
}

# 暂无模型时用固定数据展示布局；不会分析图片，也不会下载权重。
DEMO_RESULTS = [
    {"label": "plastic", "probability": 0.82},
    {"label": "glass", "probability": 0.12},
    {"label": "metal", "probability": 0.06},
]

# 预测使用的图片尺寸和标准化规则，后续训练时应保持一致。
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 转换为 [通道, 高, 宽] 格式的浮点张量。
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


@st.cache_resource
def load_model(path, modified_ns):
    """加载本地权重；文件时间戳改变时更新缓存。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]  # 类别顺序必须与训练时一致。
    model = build_model(len(classes), architecture=checkpoint.get("training_metadata", {}).get("architecture", "resnet18"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # 切换到预测状态，固定 BatchNorm 等层的行为。
    return model, classes


@st.cache_data
def current_model_evidence(path, modified_ns, report_modified_ns):
    return model_evidence(path, TEST_REPORT)


def predict_image(image, model, classes):
    """预处理图片，并返回概率最高的三个类别。"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    with torch.inference_mode():  # 预测无需计算梯度。
        # 添加批次维度：[3, 224, 224] → [1, 3, 224, 224]。
        logits = model(TRANSFORM(image).unsqueeze(0))
        probabilities = logits.softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(3, len(classes)))
    return [
        {"label": classes[index], "probability": score}
        for index, score in zip(indices.tolist(), scores.tolist())
    ]


def show_result(results):
    """显示分类名称及 Top-3 概率条。"""
    best = results[0]
    st.metric("类别", LABELS.get(best["label"], best["label"]))
    st.write("**Top-3 概率**")
    for result in results:
        name = LABELS.get(result["label"], result["label"])
        st.progress(
            result["probability"], text=f"{name}  ·  {result['probability']:.1%}"
        )


def main():
    st.set_page_config(page_title="垃圾图像分类", layout="wide")
    st.title("垃圾图像分类")
    has_model = CHECKPOINT.is_file()
    evidence = current_model_evidence(CHECKPOINT, CHECKPOINT.stat().st_mtime_ns,
        TEST_REPORT.stat().st_mtime_ns if TEST_REPORT.exists() else None) if has_model else None
    if evidence:
        st.caption(f"当前模型：{'EfficientNet-B0' if CHECKPOINT == EFFICIENT_CHECKPOINT else 'ResNet-18'} · {evidence['experiment']} 组 · 固定测试集准确率 {evidence['accuracy']:.2%}"
                   f"（{evidence['samples']} 张图片）。单张图片的概率不代表准确率。")
    elif has_model:
        st.caption("当前为演示模型，尚无可核验的独立测试指标。")
    left, right = st.columns([1.2, 1], gap="large")
    image, clicked = None, False

    with left:
        st.subheader("选择图片")
        sources = ["上传图片", "摄像头拍照"]
        source = st.radio(
            "图片来源",
            sources,
            horizontal=True,
            label_visibility="collapsed",
        )
        if source == "上传图片":
            input_file = st.file_uploader(
                "选择图片",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="uploaded_image",
            )
        else:
            # 拍照和上传都返回文件对象，后续复用同一套预测逻辑。
            input_file = st.camera_input("拍摄垃圾图片", key="camera_image")
            st.caption("允许摄像头访问，拍照后点击“开始预测”。")
        if input_file is not None:
            try:
                with Image.open(input_file) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                # 拍照组件已显示照片，避免再次显示相同的预览。
                if source == "上传图片":
                    st.image(image, use_container_width=True)
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ):
                st.error("图片无法读取，请重新选择。")
        clicked = st.button(
            "开始预测", type="primary", disabled=image is None or not has_model
        )

    with right:
        st.subheader("预测结果")
        if not has_model:
            st.caption("暂无模型，以下为固定演示数据，与上传图片无关。")
            show_result(DEMO_RESULTS)
        elif clicked:
            try:
                with st.spinner("正在预测…"):
                    model, classes = load_model(
                        CHECKPOINT, CHECKPOINT.stat().st_mtime_ns
                    )
                    results = predict_image(image, model, classes)
                show_result(results)
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                st.error(f"预测失败：{error}")
        else:
            st.caption("选择或拍摄图片后，点击“开始预测”。")


if __name__ == "__main__":
    main()
