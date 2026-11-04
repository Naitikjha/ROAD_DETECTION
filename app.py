"""
app.py — Interactive Web Demo for Pothole Detection System
Course: Introduction to Deep Learning (B.Tech Sem V)
Framework: PyTorch 2.9 (CPU)
Features:
  - Image Detection (Dataset samples + Custom uploads)
  - Video Detection with 1-D Kalman Filter probability smoothing
  - Grad-CAM heatmap visualization and derived bounding boxes
  - Evaluation curves & Viva cheat sheet
"""

import streamlit as st
import json
import time
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

st.set_page_config(page_title="Pothole Detection Demo", page_icon="🕳️", layout="wide")

CURRENT_DIR  = Path(__file__).parent.resolve()
MODEL_PATH   = CURRENT_DIR / "pothole_final.pth"
THRESH_PATH  = CURRENT_DIR / "threshold.json"
CONFIG_PATH  = CURRENT_DIR / "model_config.json"
METRICS_PATH = CURRENT_DIR / "metrics_summary.json"
DATA_POTHOLES = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data\potholes")
DATA_NORMAL   = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data\normal")
SAMPLE_VIDEO  = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\sample_pothole_video.mp4")

# ─────────────────────────── Model & Architecture ───────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64))

        self.layer2_block1  = ResidualBlock(64, 128, stride=2)
        self.layer2_b2_conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.layer2_b2_bn1   = nn.BatchNorm2d(128)
        self.layer2_b2_relu  = nn.ReLU(inplace=True)
        self.last_conv       = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.layer2_b2_bn2   = nn.BatchNorm2d(128)
        self.layer2_b2_relu2 = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2_block1(x)
        identity = x
        out = self.layer2_b2_relu(self.layer2_b2_bn1(self.layer2_b2_conv1(x)))
        out = self.layer2_b2_bn2(self.last_conv(out))
        out += identity
        x   = self.layer2_b2_relu2(out)
        x = self.pool(x)
        return self.fc(x)

# ─────────────────────────── Kalman Filter ───────────────────────────
class KalmanFilter1D:
    def __init__(self, Q: float = 0.01, R: float = 0.1):
        self.Q = Q
        self.R = R
        self.x = 0.5
        self.P = 1.0

    def update(self, z: float) -> float:
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return float(self.x)

    def reset(self, initial: float = 0.5):
        self.x = initial
        self.P = 1.0

# ─────────────────────────── Grad-CAM ───────────────────────────
def gradcam(model: nn.Module, image_tensor: torch.Tensor, target_layer: nn.Module, img_size: int) -> np.ndarray:
    model.eval()
    img = image_tensor.unsqueeze(0)
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["grad"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    output = model(img)
    model.zero_grad()
    output.backward()

    h1.remove()
    h2.remove()

    feat = activations["feat"].squeeze(0)
    grad = gradients["grad"].squeeze(0)
    weights = grad.mean(dim=(1, 2))

    cam = torch.zeros(feat.shape[1:])
    for i, w in enumerate(weights):
        cam = cam + w * feat[i]
    cam = torch.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cv2.resize(cam, (img_size, img_size))

def bbox_from_heatmap(cam: np.ndarray, img_size: int, percentile: int = 50):
    thresh = np.percentile(cam, percentile)
    binary = (cam >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img_size - 1, img_size - 1
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h

def overlay_cam(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img_np + alpha * heatmap_colored, 0, 255).astype(np.uint8)

# ─────────────────────────── Load Resources ───────────────────────────
@st.cache_resource
def load_all():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    img_size = cfg.get("img_size", 128)
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])

    transform = T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    model = ResNetCNN()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(THRESH_PATH) as f:
        th_data = json.load(f)
    threshold = float(th_data["threshold"])

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return model, transform, threshold, img_size, metrics

model, transform, default_threshold, IMG_SIZE, metrics = load_all()

# ─────────────────────────── Header & Navigation ───────────────────────────
st.title("🛣️ Pothole Detection System")
st.caption("Deep Learning Practical Project (B.Tech Sem V) — From Scratch (PyTorch) with Grad-CAM & 1-D Kalman Filter")

tabs = st.tabs([
    "📷 Image Detection",
    "🎥 Video Detection (Kalman Filter)",
    "📊 Model Evaluation & Curves",
    "🧠 Grad-CAM Explainability",
    "📖 Viva Cheat Sheet"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Image Detection
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Single Image Pothole Detection with Grad-CAM")
    col_input, col_config = st.columns([2, 1])

    with col_config:
        st.markdown("### Settings")
        selected_threshold = st.slider(
            "Decision Threshold (Validation F1-optimized)",
            min_value=0.10, max_value=0.90, value=default_threshold, step=0.01,
            key="img_thresh"
        )
        st.info(f"Optimal threshold chosen on validation set: **{default_threshold:.2f}**")
        input_mode = st.radio("Choose Input Source:", ["Sample Images from Dataset", "Upload Custom Image"], key="img_mode")

    sample_image = None
    if input_mode == "Sample Images from Dataset":
        pothole_samples = list(DATA_POTHOLES.glob("*.jpg"))[:8] if DATA_POTHOLES.exists() else []
        normal_samples  = list(DATA_NORMAL.glob("*.jpg"))[:8] if DATA_NORMAL.exists() else []
        
        sample_dict = {f"Pothole: {p.name}": p for p in pothole_samples}
        sample_dict.update({f"Normal Road: {p.name}": p for p in normal_samples})

        selected_sample_name = st.selectbox("Select a test image from dataset:", list(sample_dict.keys()))
        if selected_sample_name:
            sample_image = Image.open(sample_dict[selected_sample_name]).convert("RGB")
    else:
        uploaded_file = st.file_uploader("Upload road image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="img_uploader")
        if uploaded_file:
            sample_image = Image.open(uploaded_file).convert("RGB")

    if sample_image:
        tensor = transform(sample_image)
        with torch.no_grad():
            logit = model(tensor.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
        cam = gradcam(model, tensor, model.last_conv, IMG_SIZE)

        img_resized = np.array(sample_image.resize((IMG_SIZE, IMG_SIZE)))
        overlay = overlay_cam(img_resized, cam)
        x1, y1, x2, y2 = bbox_from_heatmap(cam, IMG_SIZE)
        overlay_bbox = overlay.copy()
        if prob >= selected_threshold:
            cv2.rectangle(overlay_bbox, (x1, y1), (x2, y2), (255, 255, 0), 2)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if prob >= selected_threshold:
                st.error(f"### ⚠️ POTHOLE DETECTED\n**Confidence:** `{prob:.2%}` (Threshold: `{selected_threshold:.2f}`)")
            else:
                st.success(f"### ✅ NORMAL ROAD\n**Confidence:** `{prob:.2%}` (Threshold: `{selected_threshold:.2f}`)")
            st.progress(prob)

        with res_col2:
            st.markdown(
                f"**Localisation Mode:** Weakly Supervised (Grad-CAM).\n"
                f"- Heatmap highlights regions driving the model's decision.\n"
                f"- Bounding box derived from 50th percentile contour activation."
            )

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(img_resized, caption="1. Input Road Image (128x128)", use_container_width=True)
        with col_img2:
            st.image(cam, caption="2. Grad-CAM Activation Heatmap", clamp=True, use_container_width=True)
        with col_img3:
            st.image(overlay_bbox, caption="3. Heatmap Overlay + Bounding Box", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Video Detection (Kalman Filter)
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Road Video Pothole Detection with 1-D Kalman Filter Smoothing")
    st.markdown(
        """
        In video sequences, raw CNN frame-by-frame predictions can exhibit temporal flicker.
        A **1-D Kalman Filter** maintains a hidden state of the true road condition,
        damping high-frequency measurement noise ($R$) while adapting to road state changes ($Q$).
        """
    )

    v_col1, v_col2 = st.columns([2, 1])

    with v_col2:
        st.markdown("### Video & Filter Settings")
        video_thresh = st.slider("Detection Threshold", 0.10, 0.90, default_threshold, 0.01, key="v_thresh")
        use_kalman = st.checkbox("Enable 1-D Kalman Smoothing", value=True)
        kalman_q = st.slider("Process Noise Q (state drift rate)", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        kalman_r = st.slider("Measurement Noise R (sensor noise)", 0.01, 0.5, 0.10, 0.02, format="%.2f")
        frame_stride = st.selectbox("Processing Stride", [1, 2, 3, 5], index=1, help="Stride 2 processes every 2nd frame for 2x faster performance")
        max_frames_to_process = st.number_input("Max frames to process (0 = all frames)", min_value=0, max_value=2000, value=120, step=30)

        video_source_mode = st.radio("Choose Video Source:", ["Sample Road Video", "Upload Video File"], key="v_source")

    video_path_to_use = None
    if video_source_mode == "Sample Road Video":
        if SAMPLE_VIDEO.exists():
            video_path_to_use = str(SAMPLE_VIDEO)
            st.info(f"Using sample video: `{SAMPLE_VIDEO.name}` ({SAMPLE_VIDEO.stat().st_size / (1024*1024):.1f} MB)")
        else:
            st.warning("Sample video not found on disk. Please upload a video.")
    else:
        uploaded_video = st.file_uploader("Upload MP4/AVI/MOV video", type=["mp4", "avi", "mov", "mkv"], key="v_uploader")
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path_to_use = tfile.name
            st.success("Video uploaded successfully!")

    if video_path_to_use:
        st.video(video_path_to_use)

        if st.button("🚀 Run Video Pothole Detection", type="primary"):
            cap = cv2.VideoCapture(video_path_to_use)
            if not cap.isOpened():
                st.error("Cannot open video file.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if max_frames_to_process > 0:
                    frames_limit = min(total_frames, int(max_frames_to_process))
                else:
                    frames_limit = total_frames

                kf = KalmanFilter1D(Q=kalman_q, R=kalman_r)
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                disp_col1, disp_col2 = st.columns([3, 2])
                with disp_col1:
                    live_image_placeholder = st.empty()
                with disp_col2:
                    telemetry_placeholder = st.empty()
                    chart_placeholder = st.empty()

                raw_probs, smooth_probs, frame_indices = [], [], []
                pothole_frames_detected = 0
                frame_count = 0
                processed_count = 0

                # Prepare output writer
                output_video_path = CURRENT_DIR / "annotated_video_demo.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps / frame_stride, (IMG_SIZE, IMG_SIZE))

                start_time = time.time()

                while cap.isOpened() and frame_count < frames_limit:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_stride == 0:
                        processed_count += 1
                        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                        tensor = transform(pil_img)
                        with torch.no_grad():
                            logit = model(tensor.unsqueeze(0))
                        raw_prob = torch.sigmoid(logit).item()

                        if use_kalman:
                            smoothed_prob = kf.update(raw_prob)
                        else:
                            smoothed_prob = raw_prob

                        decisive_prob = smoothed_prob

                        cam = gradcam(model, tensor, model.last_conv, IMG_SIZE)
                        img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
                        overlay = overlay_cam(img_resized, cam)

                        is_pothole = decisive_prob >= video_thresh
                        if is_pothole:
                            pothole_frames_detected += 1
                            x1, y1, x2, y2 = bbox_from_heatmap(cam, IMG_SIZE)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            status_label = f"POTHOLE p={decisive_prob:.2f}"
                            label_color = (255, 50, 50)
                        else:
                            status_label = f"NORMAL p={decisive_prob:.2f}"
                            label_color = (50, 255, 50)

                        cv2.putText(overlay, status_label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)

                        # Write to video
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        out_writer.write(overlay_bgr)

                        raw_probs.append(raw_prob)
                        smooth_probs.append(smoothed_prob)
                        frame_indices.append(frame_count)

                        # Live preview update every few frames
                        if processed_count % 3 == 0 or frame_count == frames_limit - 1:
                            live_image_placeholder.image(overlay, caption=f"Frame {frame_count}/{frames_limit} — {status_label}", use_container_width=True)
                            telemetry_placeholder.markdown(
                                f"**Processed:** `{frame_count}/{frames_limit}` frames  \n"
                                f"**Pothole Frames:** `{pothole_frames_detected}` (`{100*pothole_frames_detected/max(processed_count,1):.1f}%`)  \n"
                                f"**Raw Prob:** `{raw_prob:.3f}` | **Smoothed:** `{smoothed_prob:.3f}`"
                            )
                            df_chart = pd.DataFrame({
                                "Raw CNN": raw_probs[-30:],
                                "Kalman Smoothed": smooth_probs[-30:]
                            })
                            chart_placeholder.line_chart(df_chart, height=160)

                    frame_count += 1
                    progress_bar.progress(min(frame_count / frames_limit, 1.0))
                    status_text.text(f"Analyzing frame {frame_count} / {frames_limit}...")

                cap.release()
                out_writer.release()
                total_time = time.time() - start_time

                st.success(f"🎉 Video analysis complete! Processed {processed_count} frames in {total_time:.1f}s ({processed_count/max(total_time,0.1):.1f} FPS).")

                # Full Probability Curves across the entire video
                st.markdown("### 📈 Temporal Probability Trajectory (Kalman Filter vs Raw CNN)")
                st.caption("Notice how the 1-D Kalman Filter removes spurious high-frequency spikes and stabilizes decision thresholds.")
                full_df = pd.DataFrame({
                    "Frame": frame_indices,
                    "Raw CNN Probability": raw_probs,
                    "Kalman Smoothed Probability": smooth_probs,
                    "Decision Threshold": [video_thresh] * len(frame_indices)
                }).set_index("Frame")
                st.line_chart(full_df)

                # Download button for annotated video
                if output_video_path.exists():
                    with open(output_video_path, "rb") as vf:
                        st.download_button(
                            label="⬇️ Download Annotated Video (MP4)",
                            data=vf.read(),
                            file_name="pothole_annotated_video.mp4",
                            mime="video/mp4"
                        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Model Evaluation & Curves
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📊 Empirical Test Set Results (N = 106 Unseen Images)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Test Accuracy (Model B)", f"{metrics['model_b_test']['accuracy']:.2%}")
    m_col2.metric("Precision", f"{metrics['model_b_test']['precision']:.2%}")
    m_col3.metric("Recall", f"{metrics['model_b_test']['recall']:.2%}")
    m_col4.metric("ROC-AUC", f"{metrics['model_b_test']['roc_auc']:.4f}")

    st.markdown("### Comparison Table")
    st.table({
        "Model": ["Majority-Class Baseline", "Model A: Vanilla CNN", "Model B: ResNet CNN (BatchNorm)"],
        "Parameters": ["—", f"{metrics['model_params']['model_a']:,}", f"{metrics['model_params']['model_b']:,}"],
        "Accuracy": [f"{metrics['majority_baseline_acc']:.2%}", f"{metrics['model_a_test']['accuracy']:.2%}", f"{metrics['model_b_test']['accuracy']:.2%}"],
        "Precision": ["—", f"{metrics['model_a_test']['precision']:.2%}", f"{metrics['model_b_test']['precision']:.2%}"],
        "Recall": ["100.0%", f"{metrics['model_a_test']['recall']:.2%}", f"{metrics['model_b_test']['recall']:.2%}"],
        "F1 Score": ["—", f"{metrics['model_a_test']['f1']:.4f}", f"{metrics['model_b_test']['f1']:.4f}"],
        "ROC-AUC": ["0.5000", f"{metrics['model_a_test']['roc_auc']:.4f}", f"{metrics['model_b_test']['roc_auc']:.4f}"],
        "F1 95% Bootstrap CI": ["—", f"[{metrics['model_a_test']['f1_ci_lo']:.4f}, {metrics['model_a_test']['f1_ci_hi']:.4f}]", f"[{metrics['model_b_test']['f1_ci_lo']:.4f}, {metrics['model_b_test']['f1_ci_hi']:.4f}]"]
    })

    st.markdown("### Training & Evaluation Plots")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if (CURRENT_DIR / "training_curves.png").exists():
            st.image(str(CURRENT_DIR / "training_curves.png"), caption="Loss & Accuracy Curves (Train vs Validation)")
        if (CURRENT_DIR / "roc_curves.png").exists():
            st.image(str(CURRENT_DIR / "roc_curves.png"), caption="ROC Curves (Model A vs Model B)")
    with p_col2:
        if (CURRENT_DIR / "confusion_matrices.png").exists():
            st.image(str(CURRENT_DIR / "confusion_matrices.png"), caption="Test Confusion Matrices")
        if (CURRENT_DIR / "pr_curves.png").exists():
            st.image(str(CURRENT_DIR / "pr_curves.png"), caption="Precision-Recall Curves")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Explainable AI & Error Analysis
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🧠 Grad-CAM Explainability & Failure Analysis")
    st.markdown(
        """
        Grad-CAM computes the gradients of the output score with respect to the feature maps of `last_conv`,
        providing spatial localization without bounding box training.
        """
    )
    if (CURRENT_DIR / "gradcam_tp.png").exists():
        st.markdown("#### True Positives (Correctly Detected Potholes)")
        st.image(str(CURRENT_DIR / "gradcam_tp.png"), caption="True Positives with Grad-CAM localization")

    err_col1, err_col2 = st.columns(2)
    with err_col1:
        if (CURRENT_DIR / "gradcam_fp.png").exists():
            st.markdown("#### False Positives (Normal Road Flagged as Pothole)")
            st.image(str(CURRENT_DIR / "gradcam_fp.png"), caption="False Positives: triggered by sharp road shadows or surface markings")
    with err_col2:
        if (CURRENT_DIR / "gradcam_fn.png").exists():
            st.markdown("#### False Negatives (Potholes Missed)")
            st.image(str(CURRENT_DIR / "gradcam_fn.png"), caption="False Negatives: smooth/shallow potholes with low edge contrast")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5: Viva Cheat Sheet
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📖 Viva Cheat Sheet & Questions")
    if (CURRENT_DIR / "viva_cheatsheet.md").exists():
        with open(CURRENT_DIR / "viva_cheatsheet.md", encoding="utf-8") as vf:
            st.markdown(vf.read())
