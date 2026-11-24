"""垃圾图像分类页面。启动方式：streamlit run app.py。"""

import pickle
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from model_evidence import model_evidence

# 相对项目目录查找权重，避免启动位置不同导致找不到文件。
ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_resnet18.pth"
TEST_REPORT = ROOT / "results" / "final" / "test_metrics.json"
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

@st.cache_resource
def load_model(path, modified_ns):
    """加载本地权重；文件时间戳改变时更新缓存。"""
    # 首次预测时再导入推理依赖，上传和拍照界面可以先显示。
    import torch

    from model import build_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]  # 类别顺序必须与训练时一致。
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # 切换到预测状态，固定 BatchNorm 等层的行为。
    return model, classes


@st.cache_data
def current_model_evidence(path, modified_ns, report_modified_ns):
    return model_evidence(path, TEST_REPORT)


def predict_image(image, model, classes):
    """预处理图片，并返回概率最高的三个类别。"""
    import torch
    from torchvision import transforms

    # 图片尺寸和标准化规则与训练保持一致。
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = ImageOps.exif_transpose(image).convert("RGB")
    with torch.inference_mode():  # 预测无需计算梯度。
        # 添加批次维度：[3, 224, 224] → [1, 3, 224, 224]。
        logits = model(transform(image).unsqueeze(0))
        probabilities = logits.softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(3, len(classes)))
    return [
        {"label": classes[index], "probability": score}
        for index, score in zip(indices.tolist(), scores.tolist())
    ]


def show_result(results):
    """显示分类名称及 Top-3 概率条。"""
    best = results[0]
    st.metric("最可能类别", LABELS.get(best["label"], best["label"]))
    st.markdown("**前三类预测概率**")
    for result in results:
        name = LABELS.get(result["label"], result["label"])
        st.progress(
            result["probability"], text=f"{name}  ·  {result['probability']:.1%}"
        )
    st.caption("以上为这张图片的分类概率。")


def render_page():
    st.title("垃圾图像分类")
    st.markdown("上传图片或用摄像头拍照，识别常见垃圾的类别。")
    has_model = CHECKPOINT.is_file()
    left, right = st.columns([1.2, 1], gap="large")

    with left, st.container(border=True):
        st.subheader("选择图片")
        st.caption("尽量对准一件垃圾，让它占据画面的大部分。")
        source = st.segmented_control(
            "图片来源",
            ["上传图片", "摄像头拍照"],
            default="上传图片",
            required=True,
            label_visibility="collapsed",
        )
        if source == "摄像头拍照":
            input_file = st.camera_input("拍摄垃圾图片", key="camera_image")
        else:
            input_file = st.file_uploader(
                "选择要识别的图片",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="uploaded_image",
            )
        image = None
        if input_file is not None:
            try:
                with Image.open(input_file) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ):
                st.error("图片无法读取，请重新选择。")
        clicked = st.button(
            "开始识别",
            type="primary",
            icon=":material/search:",
            width="stretch",
            disabled=image is None or not has_model,
        )
        if image is not None and source != "摄像头拍照":
            preview = image.copy()
            preview.thumbnail((480, 290))
            st.image(preview, caption="上传预览", alt="待识别的垃圾图片", width="content")

    with right, st.container(border=True):
        st.subheader("识别结果")
        if not has_model:
            st.warning("暂无模型权重，以下为演示结果。")
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
                ImportError,
                RuntimeError,
                ValueError,
                KeyError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                st.error(f"预测失败：{error}")
        else:
            if image is None:
                st.info("选择一张图片后，这里会显示预测类别和前三类的概率。", icon=":material/image_search:")
            else:
                st.info("图片已就绪。点击「开始识别」查看结果。", icon=":material/check_circle:")

    with st.expander("模型与测试数据", icon=":material/analytics:"):
        st.caption("识别类别　" + "、".join(LABELS.values()))
        if has_model:
            evidence = current_model_evidence(
                CHECKPOINT,
                CHECKPOINT.stat().st_mtime_ns,
                TEST_REPORT.stat().st_mtime_ns if TEST_REPORT.exists() else None,
            )
            if evidence:
                st.write(
                    f"**ResNet-18 · {evidence['experiment']} 组**  \n"
                    f"固定测试集准确率 **{evidence['accuracy']:.2%}**（{evidence['samples']} 张图片）"
                )
            else:
                st.caption("ResNet-18 · 暂无可核验的测试集指标")


def main():
    st.set_page_config(page_title="垃圾图像分类", page_icon=":material/recycling:", layout="wide")
    with st.container(horizontal_alignment="center"), st.container(width=1200):
        render_page()


if __name__ == "__main__":
    main()
