"""垃圾图像分类页面。启动方式：streamlit run app.py。"""

import pickle
from pathlib import Path
from threading import Lock
from time import monotonic

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from torchvision import transforms

from model import build_model

# 相对项目目录查找权重，避免启动位置不同导致找不到文件。
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_resnet18.pth"
# TODO: 实时识别代码暂时保留，后续开放入口时将此开关设为 True。
ENABLE_LIVE_CAMERA = False
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
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # 切换到预测状态，固定 BatchNorm 等层的行为。
    return model, classes


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


class LivePredictor:
    """在视频线程中计算结果，用锁向页面线程安全地传递数据。"""

    def __init__(self, model, classes, interval=3.0):
        # interval 秒内只计算一次，避免视频帧积压。
        self.model, self.classes = model, classes
        self.interval = interval
        self.lock = Lock()
        self.running = False
        self.busy = False
        self.generation = 0
        self.last_started = 0.0
        self.results = None
        self.error = None

    def set_running(self, running):
        # 开始、停止或切换输入时清除旧结果；丢弃停止前尚未完成的推理。
        with self.lock:
            if self.running != running:
                self.running = running
                self.generation += 1
                self.last_started = 0.0
                self.results = None
                self.error = None

    def stop(self):
        self.set_running(False)

    def process_frame(self, frame):
        # 视频预览保留原帧；最多每 3 秒识别一次，避免积压每一帧。
        with self.lock:
            now = monotonic()
            if not self.running or self.busy or now - self.last_started < self.interval:
                return frame
            self.busy = True
            self.last_started = now
            generation = self.generation

        results, error = None, None
        try:
            # 这里只做计算，不能在视频回调线程中调用 st.* 页面函数。
            results = predict_image(frame.to_image(), self.model, self.classes)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            with self.lock:
                self.busy = False
                if self.running and generation == self.generation:
                    self.results, self.error = results, error
        return frame

    def snapshot(self):
        with self.lock:
            return self.results, self.error


@st.fragment(run_every=0.5)
def show_live_result(processor):
    """只刷新右侧结果，不重复创建左侧的视频连接。"""
    results, error = processor.snapshot()
    if error:
        st.error(f"实时预测失败：{error}")
    elif results:
        show_result(results)
    else:
        st.caption("等待摄像头画面…")


def main():
    st.set_page_config(page_title="垃圾图像分类", layout="wide")
    st.title("垃圾图像分类")
    has_model = CHECKPOINT.is_file()
    left, right = st.columns([1.2, 1], gap="large")
    image, clicked, live_context, processor = None, False, None, None

    with left:
        st.subheader("选择图片")
        sources = ["上传图片", "摄像头拍照"]
        if ENABLE_LIVE_CAMERA:
            sources.append("实时摄像头")
        source = st.radio(
            "图片来源",
            sources,
            horizontal=True,
            label_visibility="collapsed",
        )
        if source in ("上传图片", "摄像头拍照"):
            # 离开实时模式时停止推理，组件移除后由浏览器释放视频连接。
            previous = st.session_state.get("live_processor")
            if previous is not None:
                previous.stop()
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
        elif not has_model:
            st.info("尚未找到训练模型，暂时无法进行实时识别。")
        else:
            try:
                stamp = CHECKPOINT.stat().st_mtime_ns
                model, classes = load_model(CHECKPOINT, stamp)
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                st.error(f"模型加载失败：{error}")
            else:
                # 每个浏览器会话单独保存视频状态，页面刷新时不反复初始化。
                if st.session_state.get("live_model_stamp") != stamp:
                    previous = st.session_state.get("live_processor")
                    if previous is not None:
                        previous.stop()
                    st.session_state.live_processor = LivePredictor(model, classes)
                    st.session_state.live_model_stamp = stamp
                processor = st.session_state.live_processor
                live_context = webrtc_streamer(
                    key=f"live-camera-{stamp}",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=processor.process_frame,
                    on_video_ended=processor.stop,
                    async_processing=True,
                    # 本机浏览器直连本机服务，不使用外部中继。
                    rtc_configuration={"iceServers": []},
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 15, "max": 20},
                        },
                        "audio": False,  # 仅请求摄像头，不请求麦克风。
                    },
                    media_toggle_controls=False,
                    translations={
                        "start": "开始识别",
                        "stop": "停止识别",
                        "select_device": "选择摄像头",
                        "device_ask_permission": "请允许浏览器访问摄像头",
                        "device_access_denied": "摄像头访问被拒绝，请在浏览器中允许访问",
                    },
                )
                processor.set_running(live_context.state.playing)
                st.caption("点击“开始识别”后自动更新结果；点击“停止识别”关闭摄像头。")

    with right:
        st.subheader("预测结果")
        if source == "实时摄像头":
            if live_context is not None and live_context.state.playing:
                show_live_result(processor)
            else:
                st.caption("启动摄像头后，预测类别和 Top-3 概率会自动更新。")
        elif not has_model:
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
