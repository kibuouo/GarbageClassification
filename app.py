"""Local Streamlit demonstration; run with streamlit run app.py."""
import csv
import io
import pickle
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from garbage_classifier.common import LABELS_ZH, ROOT
from garbage_classifier.inference import Predictor


@st.cache_resource
def load_predictor(path, modified_ns):
    # The modification timestamp invalidates the cache after a model is replaced.
    return Predictor(path)


def history_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def main():
    st.set_page_config(page_title="垃圾图像智能识别", page_icon="♻️", layout="wide")
    st.title("♻️ 垃圾图像智能识别")
    st.caption("基于 ResNet-18 迁移学习 · 人工智能课程实验系统")
    st.info("识别纸板、玻璃、金属、纸张、塑料和其他垃圾六种图像标签。标签不等同于生活垃圾四分类，请按当地规定投放。")
    st.session_state.setdefault("history", [])
    checkpoints = sorted((ROOT / "checkpoints").rglob("*.pth"))
    with st.sidebar:
        st.header("识别设置")
        selected = st.selectbox("模型", checkpoints, format_func=lambda p: str(p.relative_to(ROOT / "checkpoints"))) if checkpoints else None
        threshold = st.slider("低置信度提示阈值", 0.0, 1.0, 0.60, 0.05)
        st.caption("概率不是正确率。模型可能无法识别训练类别以外的物体。")
    predictor = None
    if selected is None:
        st.warning("尚未找到训练模型。准备 TrashNet 数据并运行 train.py 后，即可在这里选择模型。")
    else:
        try:
            predictor = load_predictor(selected, selected.stat().st_mtime_ns)
            st.caption(f"模型已加载 · 设备 {predictor.device}")
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
            st.error(f"模型加载失败：{error}")
    upload = st.file_uploader("上传垃圾图片", type=("jpg", "jpeg", "png", "webp"))
    if upload is not None:
        try:
            with Image.open(upload) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error("图片无法读取，请上传有效的 JPG、PNG 或 WebP 图片。")
        else:
            preview, result_area = st.columns(2)
            preview.image(image, caption=upload.name, use_container_width=True)
            with result_area:
                if st.button("开始识别", type="primary", disabled=predictor is None):
                    try:
                        with st.spinner("正在识别…"):
                            result = predictor.predict(image)
                        label = LABELS_ZH.get(result["label"], result["label"])
                        st.success(f"预测类别：{label}")
                        confidence, latency = st.columns(2)
                        confidence.metric("预测概率", f"{result['confidence']:.1%}")
                        latency.metric("处理耗时", f"{result['elapsed_ms']:.1f} ms")
                        for row in result["top3"]:
                            st.progress(row["probability"], text=f"{LABELS_ZH.get(row['label'], row['label'])}：{row['probability']:.1%}")
                        if result["confidence"] < threshold:
                            st.warning("预测不确定，建议调整光照、背景和拍摄角度后重试。")
                        st.session_state.history.append({"time": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "filename": upload.name, "model": str(selected.relative_to(ROOT / "checkpoints")),
                            "label": result["label"], "confidence": result["confidence"],
                            "elapsed_ms": round(result["elapsed_ms"], 2), "uncertain": result["confidence"] < threshold})
                    except (RuntimeError, ValueError, OSError) as error:
                        st.error(f"识别失败：{error}")
    st.subheader("本次会话的识别记录")
    if st.session_state.history:
        st.dataframe(st.session_state.history, use_container_width=True)
        st.download_button("导出 CSV", history_csv(st.session_state.history), file_name="classification_history.csv", mime="text/csv")
    else:
        st.caption("识别完成后会在这里记录结果，可导出 CSV 保存。刷新或关闭会话后，记录可能清空。")


if __name__ == "__main__":
    main()
