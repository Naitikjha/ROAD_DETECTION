"""
app.py — Presentation-Grade Interactive Web Application
Project: Road Pothole Detection & Weakly-Supervised Localization System
Course: Introduction to Deep Learning (B.Tech Sem V)
Framework: PyTorch 2.9 (CPU)
Features:
  - Executive-grade Dark Dashboard UI
  - Real-time Image Saliency Inspection (Grad-CAM)
  - Road Video Temporal Screening with 1-D Kalman Filter
  - Empirical Validation & Benchmark Analytics
  - Explainable AI (XAI) Diagnostic Suite
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

# ─────────────────────────── Page Configuration ───────────────────────────
st.set_page_config(
    page_title="Road Surface Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── Global Paths ───────────────────────────
CURRENT_DIR   = Path(__file__).parent.resolve()
MODEL_PATH    = CURRENT_DIR / "pothole_final.pth"
THRESH_PATH   = CURRENT_DIR / "threshold.json"
CONFIG_PATH   = CURRENT_DIR / "model_config.json"
METRICS_PATH  = CURRENT_DIR / "metrics_summary.json"
DATA_POTHOLES = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data\potholes")
DATA_NORMAL   = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data\normal")
SAMPLE_VIDEO  = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\sample_pothole_video.mp4")

# ─────────────────────────── Custom Premium CSS ───────────────────────────
st.markdown("""
<style>
    /* Global reset & typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide standard Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Main Container background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #060913 100%);
        color: #E2E8F0;
    }

    /* Custom Header Card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .hero-sub {
        color: #94A3B8;
        font-size: 0.98rem;
        margin-top: 0.4rem;
        margin-bottom: 0.9rem;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .meta-badge {
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
        font-family: 'JetBrains Mono', monospace;
    }
    .meta-badge.emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.25);
    }
    .meta-badge.purple {
        background: rgba(168, 85, 247, 0.12);
        color: #C084FC;
        border-color: rgba(168, 85, 247, 0.25);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Status Alert Banners */
    .alert-pothole {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #FCA5A5;
    }
    .alert-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 1px solid rgba(52, 211, 153, 0.5);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #6EE7B7;
    }

    /* Styled Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }

    /* Image Display Frames */
    .img-box {
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        text-align: center;
        padding: 0.5rem;
    }
    .img-caption {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Model Architecture (from scratch) ───────────────────────────
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

        self.layer2_block1   = ResidualBlock(64, 128, stride=2)
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

# ─────────────────────────── 1-D Kalman Filter ───────────────────────────
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

# ─────────────────────────── Cache Loading ───────────────────────────
@st.cache_resource
def load_assets():
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

model, transform, default_threshold, IMG_SIZE, metrics = load_assets()

# ─────────────────────────── Executive Hero Banner ───────────────────────────
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="hero-title">🛣️ Road Pothole Detection & Localization System</h1>
            <div class="hero-sub">Deep Learning Practical Project (B.Tech Sem V) • Autonomous Road Surface Inspection</div>
            <div class="badge-container">
                <span class="meta-badge">PyTorch 2.9 (From Scratch)</span>
                <span class="meta-badge emerald">ResNet + BatchNorm</span>
                <span class="meta-badge purple">Grad-CAM XAI</span>
                <span class="meta-badge">1-D Kalman Filter</span>
                <span class="meta-badge emerald">Test Acc: 91.51%</span>
                <span class="meta-badge purple">ROC-AUC: 0.9669</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Navigation Tabs (Viva Removed) ───────────────────────────
tabs = st.tabs([
    "📷 Image Saliency Inspection",
    "🎥 Road Video Stream & Kalman Tracking",
    "📊 Empirical Benchmarks & Curves",
    "🧠 Grad-CAM Explainability Diagnostics"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Image Saliency Inspection
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_main, col_sidebar = st.columns([2.2, 1.0])

    with col_sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Inspection Controls")
        
        selected_threshold = st.slider(
            "Decision Threshold (τ)",
            min_value=0.10, max_value=0.90, value=default_threshold, step=0.01,
            help="Optimal threshold determined on validation set via F1 maximization."
        )
        st.caption(f"🎯 Validation-optimal threshold: **`{default_threshold:.2f}`**")

        input_mode = st.radio("Input Source Selection", ["Benchmark Dataset Samples", "Upload External Image"])
        st.markdown('</div>', unsafe_allow_html=True)

    sample_image = None
    with col_sidebar:
        if input_mode == "Benchmark Dataset Samples":
            pothole_samples = list(DATA_POTHOLES.glob("*.jpg"))[:12] if DATA_POTHOLES.exists() else []
            normal_samples  = list(DATA_NORMAL.glob("*.jpg"))[:12] if DATA_NORMAL.exists() else []
            
            sample_dict = {f"🕳️ Pothole Sample: {p.name}": p for p in pothole_samples}
            sample_dict.update({f"✅ Normal Road: {p.name}": p for p in normal_samples})

            selected_sample_name = st.selectbox("Select Test Image:", list(sample_dict.keys()))
            if selected_sample_name:
                sample_image = Image.open(sample_dict[selected_sample_name]).convert("RGB")
        else:
            uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                sample_image = Image.open(uploaded_file).convert("RGB")

    with col_main:
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

            # Decision Banner
            is_detected = prob >= selected_threshold
            if is_detected:
                st.markdown(f"""
                <div class="alert-pothole">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:1.3rem; font-weight:800;">⚠️ POTHOLE CRATER DETECTED</span><br>
                            <span style="font-size:0.9rem;">Weakly-supervised spatial bounding box derived from gradient saliency maps.</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.8rem; font-weight:800; font-family:'JetBrains Mono';">{prob:.1%}</span><br>
                            <span style="font-size:0.75rem; text-transform:uppercase;">Model Confidence</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-normal">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:1.3rem; font-weight:800;">✅ NORMAL ROAD SURFACE</span><br>
                            <span style="font-size:0.9rem;">Surface continuity confirmed within permissible texture variance.</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.8rem; font-weight:800; font-family:'JetBrains Mono';">{prob:.1%}</span><br>
                            <span style="font-size:0.75rem; text-transform:uppercase;">Pothole Likelihood</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.markdown('<div class="img-box">', unsafe_allow_html=True)
                st.image(img_resized, use_container_width=True)
                st.markdown('<div class="img-caption">1. Raw Photographic Feed (128x128)</div></div>', unsafe_allow_html=True)
            with col_v2:
                st.markdown('<div class="img-box">', unsafe_allow_html=True)
                st.image(cam, clamp=True, use_container_width=True)
                st.markdown('<div class="img-caption">2. Grad-CAM Activation Energy</div></div>', unsafe_allow_html=True)
            with col_v3:
                st.markdown('<div class="img-box">', unsafe_allow_html=True)
                st.image(overlay_bbox, use_container_width=True)
                st.markdown('<div class="img-caption">3. Weakly-Supervised Bounding Contour</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Video Stream & Kalman Tracking
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_v_main, col_v_ctrl = st.columns([2.2, 1.0])

    with col_v_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Video Tracking Controls")
        video_thresh = st.slider("Detection Threshold", 0.10, 0.90, default_threshold, 0.01, key="v_thresh_tab")
        use_kalman = st.checkbox("Enable 1-D Kalman Filter", value=True, help="Eliminates frame-to-frame probability flickering")
        kalman_q = st.slider("Process Noise Q", 0.001, 0.1, 0.01, 0.005, format="%.3f")
        kalman_r = st.slider("Sensor Noise R", 0.01, 0.5, 0.10, 0.02, format="%.2f")
        frame_stride = st.selectbox("Processing Stride", [1, 2, 3, 5], index=1, help="Stride 2 processes every 2nd frame for 2x faster performance")
        max_frames = st.number_input("Max Frame Budget", min_value=30, max_value=2000, value=120, step=30)
        
        video_source_mode = st.radio("Video Source", ["Preloaded Video Sample", "Upload Video File"], key="v_mode")
        
        chosen_video_path = None
        if video_source_mode == "Preloaded Video Sample":
            v_options = {}
            if (CURRENT_DIR / "sample_test_short.mp4").exists():
                v_options["⚡ Quick Test Clip (120 frames, 4.8s)"] = CURRENT_DIR / "sample_test_short.mp4"
            if (CURRENT_DIR / "sample_road_video.mp4").exists():
                v_options["📹 Full Dashcam Video (692 frames, 27.7s)"] = CURRENT_DIR / "sample_road_video.mp4"
            elif SAMPLE_VIDEO.exists():
                v_options["📹 Full Dashcam Video (692 frames, 27.7s)"] = SAMPLE_VIDEO
            
            if v_options:
                selected_v_key = st.selectbox("Select Sample Video:", list(v_options.keys()))
                chosen_video_path = str(v_options[selected_v_key])
        else:
            up_v = st.file_uploader("Upload Video File (MP4, MOV)", type=["mp4", "mov", "avi"])
            if up_v:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tf.write(up_v.read())
                chosen_video_path = tf.name

        st.markdown('</div>', unsafe_allow_html=True)

    with col_v_main:
        if chosen_video_path:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"**Loaded Footage:** `{Path(chosen_video_path).name}`")
            st.video(chosen_video_path)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Execute Autonomous Video Screening", type="primary"):
                cap = cv2.VideoCapture(chosen_video_path)
                if not cap.isOpened():
                    st.error("Error opening video stream.")
                else:
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    limit = min(total_f, int(max_frames)) if max_frames > 0 else total_f

                    kf = KalmanFilter1D(Q=kalman_q, R=kalman_r)
                    prog_bar = st.progress(0.0)
                    prog_txt = st.empty()

                    v_disp_col1, v_disp_col2 = st.columns([1.5, 1.0])
                    with v_disp_col1:
                        frame_live_view = st.empty()
                    with v_disp_col2:
                        telemetry_box = st.empty()
                        live_chart_box = st.empty()

                    raw_list, smooth_list, frame_ids = [], [], []
                    pothole_count = 0
                    f_idx = 0
                    proc_idx = 0

                    output_v_path = CURRENT_DIR / "annotated_video_demo.mp4"
                    writer = cv2.VideoWriter(str(output_v_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / frame_stride, (IMG_SIZE, IMG_SIZE))
                    t_start = time.time()

                    while cap.isOpened() and f_idx < limit:
                        ret, bgr = cap.read()
                        if not ret: break

                        if f_idx % frame_stride == 0:
                            proc_idx += 1
                            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                            ten = transform(pil)
                            with torch.no_grad():
                                lg = model(ten.unsqueeze(0))
                            r_prob = torch.sigmoid(lg).item()
                            s_prob = kf.update(r_prob) if use_kalman else r_prob
                            dec_prob = s_prob

                            cm = gradcam(model, ten, model.last_conv, IMG_SIZE)
                            ov = overlay_cam(np.array(pil.resize((IMG_SIZE, IMG_SIZE))), cm)

                            is_pot = dec_prob >= video_thresh
                            if is_pot:
                                pothole_count += 1
                                bx1, by1, bx2, by2 = bbox_from_heatmap(cm, IMG_SIZE)
                                cv2.rectangle(ov, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                                tag = f"POTHOLE {dec_prob:.2f}"
                                clr = (255, 50, 50)
                            else:
                                tag = f"NORMAL {dec_prob:.2f}"
                                clr = (50, 255, 50)

                            cv2.putText(ov, tag, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
                            writer.write(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

                            raw_list.append(r_prob)
                            smooth_list.append(s_prob)
                            frame_ids.append(f_idx)

                            if proc_idx % 3 == 0 or f_idx == limit - 1:
                                frame_live_view.image(ov, caption=f"Frame {f_idx}/{limit} • {tag}", use_container_width=True)
                                telemetry_box.markdown(f"""
                                <div class="glass-card" style="padding:0.8rem;">
                                    <b>Analyzed:</b> <code>{f_idx}/{limit}</code> frames<br>
                                    <b>Pothole Detections:</b> <code>{pothole_count}</code> ({100*pothole_count/max(proc_idx,1):.1f}%)<br>
                                    <b>Raw CNN Prob:</b> <code>{r_prob:.3f}</code><br>
                                    <b>Kalman Smoothed:</b> <code>{s_prob:.3f}</code>
                                </div>
                                """, unsafe_allow_html=True)
                                live_chart_box.line_chart(pd.DataFrame({
                                    "Raw": raw_list[-25:],
                                    "Kalman": smooth_list[-25:]
                                }), height=150)

                        f_idx += 1
                        prog_bar.progress(min(f_idx / limit, 1.0))
                        prog_txt.caption(f"Analyzing road surface in frame {f_idx} of {limit}...")

                    cap.release()
                    writer.release()
                    dur = time.time() - t_start

                    st.success(f"✅ Video processing complete in {dur:.1f}s ({proc_idx/max(dur,0.1):.1f} FPS).")
                    
                    st.markdown("### 📈 Kalman Filter Temporal Smoothing Analysis")
                    st.caption("Visualizing the dampening of high-frequency spurious noise across road video frames.")
                    plot_df = pd.DataFrame({
                        "Frame Index": frame_ids,
                        "Raw CNN Likelihood": raw_list,
                        "Kalman Smoothed Likelihood": smooth_list,
                        "Decision Threshold": [video_thresh] * len(frame_ids)
                    }).set_index("Frame Index")
                    st.line_chart(plot_df)

                    if output_v_path.exists():
                        with open(output_v_path, "rb") as vf:
                            st.download_button(
                                label="⬇️ Download Annotated Video File",
                                data=vf.read(),
                                file_name="pothole_annotated_feed.mp4",
                                mime="video/mp4"
                            )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Empirical Benchmarks & Curves
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🏆 Comprehensive Evaluation (Held-Out Test Partition, N = 106)")
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{metrics['model_b_test']['accuracy']:.1%}</div>
            <div class="kpi-label">Classification Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{metrics['model_b_test']['precision']:.1%}</div>
            <div class="kpi-label">Precision Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{metrics['model_b_test']['recall']:.1%}</div>
            <div class="kpi-label">Pothole Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{metrics['model_b_test']['roc_auc']:.4f}</div>
            <div class="kpi-label">Area Under ROC (AUC)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📐 Architecture Comparison: Vanilla CNN vs. ResNet (BatchNorm)")
    st.table({
        "Model Architecture": ["Majority-Class Baseline", "Model A: Vanilla CNN", "Model B: ResNet CNN + BatchNorm"],
        "Parameter Budget": ["0", f"{metrics['model_params']['model_a']:,}", f"{metrics['model_params']['model_b']:,}"],
        "Accuracy": [f"{metrics['majority_baseline_acc']:.2%}", f"{metrics['model_a_test']['accuracy']:.2%}", f"{metrics['model_b_test']['accuracy']:.2%}"],
        "Precision": ["—", f"{metrics['model_a_test']['precision']:.2%}", f"{metrics['model_b_test']['precision']:.2%}"],
        "Recall": ["100.0%", f"{metrics['model_a_test']['recall']:.2%}", f"{metrics['model_b_test']['recall']:.2%}"],
        "F1 Score": ["—", f"{metrics['model_a_test']['f1']:.4f}", f"{metrics['model_b_test']['f1']:.4f}"],
        "ROC-AUC": ["0.5000", f"{metrics['model_a_test']['roc_auc']:.4f}", f"{metrics['model_b_test']['roc_auc']:.4f}"],
        "F1 95% Bootstrap CI": ["—", f"[{metrics['model_a_test']['f1_ci_lo']:.4f}, {metrics['model_a_test']['f1_ci_hi']:.4f}]", f"[{metrics['model_b_test']['f1_ci_lo']:.4f}, {metrics['model_b_test']['f1_ci_hi']:.4f}]"]
    })
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 Empirical Training & Discriminative Curves")
    grid_c1, grid_c2 = st.columns(2)
    with grid_c1:
        if (CURRENT_DIR / "training_curves.png").exists():
            st.image(str(CURRENT_DIR / "training_curves.png"), caption="Training & Validation Trajectory (Loss & Accuracy)", use_container_width=True)
        if (CURRENT_DIR / "roc_curves.png").exists():
            st.image(str(CURRENT_DIR / "roc_curves.png"), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
    with grid_c2:
        if (CURRENT_DIR / "confusion_matrices.png").exists():
            st.image(str(CURRENT_DIR / "confusion_matrices.png"), caption="Confusion Matrix Distributions", use_container_width=True)
        if (CURRENT_DIR / "pr_curves.png").exists():
            st.image(str(CURRENT_DIR / "pr_curves.png"), caption="Precision-Recall Curves", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Explainable AI (Grad-CAM)
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🧠 Explainable AI: Activation Saliency & Failure Analysis")
    st.markdown(
        """
        **Grad-CAM** interrogates the final convolutional layer (`self.last_conv`) by calculating the gradients
        of the target class output with respect to spatial feature activations. This provides transparent attribution
        of features driving positive and negative classifications without bounding box supervision.
        """
    )
    
    if (CURRENT_DIR / "gradcam_tp.png").exists():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ✅ True Positive Detections (Accurate Pothole Saliency)")
        st.image(str(CURRENT_DIR / "gradcam_tp.png"), caption="True Positives: Model concentrates spatial energy strictly on crater edges & depth cues", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    err1, err2 = st.columns(2)
    with err1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ False Positives (Look-Alikes & Road Markings)")
        if (CURRENT_DIR / "gradcam_fp.png").exists():
            st.image(str(CURRENT_DIR / "gradcam_fp.png"), caption="False Positives: High contrast tree shadows or asphalt seam cracks mimic pothole edges", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with err2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 False Negatives (Shallow / Blended Depressions)")
        if (CURRENT_DIR / "gradcam_fn.png").exists():
            st.image(str(CURRENT_DIR / "gradcam_fn.png"), caption="False Negatives: Eroded, low-depth potholes without sharp perimeter contrast", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
