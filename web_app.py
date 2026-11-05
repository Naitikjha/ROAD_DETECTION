"""
web_app.py — Autonomous Road Vision & Pothole Screening Dashboard
Modern presentation-grade FastAPI web application.
"""

import os
import io
import json
import base64
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ─────────────────────────── Paths & Configurations ───────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
MODEL_PATH   = BASE_DIR / "pothole_final.pth"
CONFIG_PATH  = BASE_DIR / "model_config.json"
THRESH_PATH  = BASE_DIR / "threshold.json"
METRICS_PATH = BASE_DIR / "metrics_summary.json"
DATA_DIR     = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data")
POTHOLES_DIR = DATA_DIR / "potholes"
NORMAL_DIR   = DATA_DIR / "normal"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Model Definition ───────────────────────────
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

# ─────────────────────────── Grad-CAM ───────────────────────────
def compute_gradcam(model: nn.Module, image_tensor: torch.Tensor, target_layer: nn.Module, img_size: int):
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

def extract_bbox(cam: np.ndarray, img_size: int, percentile: int = 50):
    thresh = np.percentile(cam, percentile)
    binary = (cam >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img_size - 1, img_size - 1, 0
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area = cv2.contourArea(largest)
    return x, y, x + w, y + h, int(area)

def build_overlay(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.42):
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img_np + alpha * heatmap_colored, 0, 255).astype(np.uint8)

# ─────────────────────────── Asset Loading ───────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
IMG_SIZE = cfg.get("img_size", 128)
MEAN     = cfg.get("mean", [0.485, 0.456, 0.406])
STD      = cfg.get("std", [0.229, 0.224, 0.225])

with open(THRESH_PATH) as f:
    thresh_data = json.load(f)
DEFAULT_THRESHOLD = float(thresh_data.get("threshold", 0.31))

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

transform = T.Compose([
    T.Resize(IMG_SIZE + 32),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

model = ResNetCNN()
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

# ─────────────────────────── FastAPI App ───────────────────────────
app = FastAPI(title="Road Surface Pothole Detection Intelligence Suite")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def pil_to_base64(img_np: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

def cam_to_base64(cam: np.ndarray) -> str:
    cam_u8 = (cam * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics": METRICS
    })

@app.get("/api/samples")
async def get_samples():
    """Return list of sample images for instant 1-click testing."""
    samples = []
    if POTHOLES_DIR.exists():
        for p in sorted(POTHOLES_DIR.glob("*.jpg"))[:8]:
            samples.append({
                "id": f"pothole_{p.name}",
                "label": "Pothole",
                "filename": p.name,
                "path": str(p),
                "type": "pothole"
            })
    if NORMAL_DIR.exists():
        for p in sorted(NORMAL_DIR.glob("*.jpg"))[:8]:
            samples.append({
                "id": f"normal_{p.name}",
                "label": "Normal Road",
                "filename": p.name,
                "path": str(p),
                "type": "normal"
            })
    return JSONResponse(samples)

@app.get("/api/sample-image/{sample_type}/{filename}")
async def get_sample_image(sample_type: str, filename: str):
    folder = POTHOLES_DIR if sample_type == "pothole" else NORMAL_DIR
    target = folder / filename
    if target.exists():
        return FileResponse(str(target), media_type="image/jpeg")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.get("/api/metrics")
async def get_metrics():
    return JSONResponse(METRICS)

@app.get("/api/plot/{plot_name}")
async def get_plot(plot_name: str):
    plot_file = BASE_DIR / plot_name
    if plot_file.exists():
        return FileResponse(str(plot_file), media_type="image/png")
    return JSONResponse({"error": "Plot not found"}, status_code=404)

@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None),
    threshold: float = Form(DEFAULT_THRESHOLD)
):
    t0 = time.time()
    if file and file.filename:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    elif sample_path and Path(sample_path).exists():
        pil_img = Image.open(sample_path).convert("RGB")
    else:
        return JSONResponse({"error": "No valid image provided."}, status_code=400)

    # Preprocessing & Inference
    tensor = transform(pil_img)
    with torch.no_grad():
        logit = model(tensor.unsqueeze(0))
    prob = torch.sigmoid(logit).item()

    # Grad-CAM Saliency
    cam = compute_gradcam(model, tensor, model.last_conv, IMG_SIZE)
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay = build_overlay(img_resized, cam)

    x1, y1, x2, y2, area = extract_bbox(cam, IMG_SIZE)
    overlay_bbox = overlay.copy()
    is_pothole = bool(prob >= threshold)

    if is_pothole:
        cv2.rectangle(overlay_bbox, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(overlay_bbox, f"CRATER {prob:.0%}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    elapsed_ms = (time.time() - t0) * 1000

    return JSONResponse({
        "status": "success",
        "is_pothole": is_pothole,
        "probability": round(prob, 4),
        "threshold": round(threshold, 2),
        "latency_ms": round(elapsed_ms, 1),
        "bbox": {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "area_px": area,
            "width": x2 - x1, "height": y2 - y1
        },
        "images": {
            "original": pil_to_base64(img_resized),
            "heatmap": cam_to_base64(cam),
            "overlay": pil_to_base64(overlay_bbox)
        }
    })

@app.post("/api/predict-video-frame")
async def predict_video_frame(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    kalman_state: Optional[str] = Form(None),
    kalman_q: float = Form(0.01),
    kalman_r: float = Form(0.10)
):
    """Processes an individual video frame sent from client-side video player."""
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = transform(pil_img)
    with torch.no_grad():
        logit = model(tensor.unsqueeze(0))
    raw_prob = torch.sigmoid(logit).item()

    # Kalman filter update
    kf = KalmanFilter1D(Q=kalman_q, R=kalman_r)
    if kalman_state:
        try:
            st_dict = json.loads(kalman_state)
            kf.x = float(st_dict.get("x", 0.5))
            kf.P = float(st_dict.get("P", 1.0))
        except Exception:
            pass

    smoothed_prob = kf.update(raw_prob)
    cam = compute_gradcam(model, tensor, model.last_conv, IMG_SIZE)
    x1, y1, x2, y2, area = extract_bbox(cam, IMG_SIZE)

    return JSONResponse({
        "raw_prob": round(raw_prob, 4),
        "smoothed_prob": round(smoothed_prob, 4),
        "is_pothole": bool(smoothed_prob >= threshold),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "kalman_state": {"x": kf.x, "P": kf.P}
    })
