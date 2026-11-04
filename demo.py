#!/usr/bin/env python3
"""
demo.py — Pothole Detection Demo
==================================
Course: Introduction to Deep Learning (B.Tech Sem V)
Project: Pothole Detection System

Usage:
  # Single image:
  python demo.py --input path/to/image.jpg

  # Video:
  python demo.py --input path/to/video.mp4 --output annotated_output.mp4

Outputs:
  - Annotated image/video with Grad-CAM heatmap overlay and bounding box
  - Printed probability and detection label

Localisation disclaimer:
  Bounding boxes are derived from Grad-CAM (weakly supervised).
  No ground-truth box annotations were used for training.
  Box-level metrics such as mAP are NOT reported.
"""

import argparse
import json
import sys
import os
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Safe UTF-8 console output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────── Kalman Filter (manual) ──────────────────────
class KalmanFilter1D:
    """
    Simple 1-D Kalman filter for smoothing per-frame probabilities.
    Syllabus topic: Kalman filter for smoothing.

    State model:
        x_t = x_{t-1} + process_noise       (probability drifts slowly)
        z_t = x_t     + measurement_noise   (observed raw prob)

    Parameters:
        Q : process noise variance  (lower = smoother but slower to react)
        R : measurement noise variance (higher = trust model less per frame)
    """
    def __init__(self, Q: float = 0.01, R: float = 0.1):
        self.Q = Q          # process noise
        self.R = R          # measurement noise
        self.x = 0.5        # initial state estimate (neutral probability)
        self.P = 1.0        # initial estimation error covariance

    def update(self, z: float) -> float:
        """
        Update the filter with a new observation z (raw probability).
        Returns the smoothed probability estimate.
        """
        # ── Predict step ──
        # State prediction: x_pred = x (identity transition)
        # Covariance prediction:
        self.P = self.P + self.Q

        # ── Update step ──
        K      = self.P / (self.P + self.R)       # Kalman gain
        self.x = self.x + K * (z - self.x)        # state update
        self.P = (1.0 - K) * self.P               # covariance update

        return float(self.x)

    def reset(self, initial: float = 0.5):
        self.x = initial
        self.P = 1.0


# ─────────────────────────── Model Definitions ───────────────────────────
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
    """
    Model B: Custom CNN with BatchNorm + residual blocks (from scratch).
    Attribute self.last_conv is the final convolutional layer for Grad-CAM.
    """
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


# ─────────────────────────── Grad-CAM ────────────────────────────────────
def gradcam(model: nn.Module, image_tensor: torch.Tensor,
            target_layer: nn.Module) -> np.ndarray:
    """
    Implements Grad-CAM from scratch (no external library).
    Syllabus topic: Explainable AI for CNNs.

    Returns heatmap as (IMG_SIZE, IMG_SIZE) numpy float array in [0, 1].
    """
    model.eval()
    img = image_tensor.unsqueeze(0)

    activations: dict = {}
    gradients:   dict = {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["grad"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    output = model(img)   # shape (1, 1) logit
    model.zero_grad()
    output.backward()

    h1.remove()
    h2.remove()

    feat    = activations["feat"].squeeze(0)   # (C, H, W)
    grad    = gradients["grad"].squeeze(0)     # (C, H, W)
    weights = grad.mean(dim=(1, 2))            # (C,)

    cam = torch.zeros(feat.shape[1:])
    for i, w in enumerate(weights):
        cam = cam + w * feat[i]
    cam = torch.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam


def overlay_heatmap(original_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Overlay jet-colourmap Grad-CAM on a BGR image (OpenCV format)."""
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    orig_resized = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))
    overlay = cv2.addWeighted(orig_resized, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def bbox_from_heatmap(cam: np.ndarray, percentile: int = 50):
    """
    Derive bounding box from thresholded Grad-CAM.
    Threshold at given percentile, find contours, return largest rect.
    """
    threshold = np.percentile(cam, percentile)
    binary    = (cam >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, IMG_SIZE - 1, IMG_SIZE - 1
    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


# ─────────────────────────── Utilities ───────────────────────────────────
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
IMG_SIZE = 128   # default; overridden by model_config.json if present

def make_preprocess(img_size):
    return T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

preprocess = make_preprocess(IMG_SIZE)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_model(model_path: str) -> ResNetCNN:
    model = ResNetCNN()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_image(model: ResNetCNN, pil_image: Image.Image) -> tuple:
    """
    Run inference on a PIL image.
    Returns (probability, cam_heatmap_numpy).
    """
    tensor = preprocess(pil_image)
    with torch.no_grad():
        logit = model(tensor.unsqueeze(0))
    prob = torch.sigmoid(logit).item()
    cam  = gradcam(model, tensor, model.last_conv)
    return prob, cam


def annotate_frame(frame_bgr: np.ndarray, prob: float,
                   cam: np.ndarray, threshold: float) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap, bounding box, and probability label
    onto a BGR frame. Returns annotated BGR frame.
    """
    overlay = overlay_heatmap(frame_bgr, cam, alpha=0.4)

    if prob >= threshold:
        x1, y1, x2, y2 = bbox_from_heatmap(cam)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"POTHOLE  p={prob:.3f}"
        label_color = (0, 0, 255)
    else:
        label = f"NORMAL   p={prob:.3f}"
        label_color = (0, 220, 0)

    cv2.putText(overlay, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, label_color, 2, cv2.LINE_AA)
    return overlay


# ─────────────────────────── Main Functions ──────────────────────────────
def process_image(model: ResNetCNN, threshold: float,
                  input_path: str, output_path: str) -> None:
    pil_img = Image.open(input_path).convert("RGB")
    prob, cam = predict_image(model, pil_img)

    # Convert PIL to BGR for OpenCV operations
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    annotated = annotate_frame(frame_bgr, prob, cam, threshold)

    cv2.imwrite(output_path, annotated)

    detection = "[POTHOLE DETECTED]" if prob >= threshold else "[NORMAL ROAD]"
    print(f"\n{'-'*50}")
    print(f"Input   : {input_path}")
    print(f"Result  : {detection}")
    print(f"p(pothole) = {prob:.4f}  |  threshold = {threshold:.2f}")
    print(f"Output  : {output_path}")
    print(f"{'-'*50}\n")
    print("NOTE: Localisation is weakly supervised via Grad-CAM.")
    print("      No box-level metrics (mAP) are claimed.\n")


def process_video(model: ResNetCNN, threshold: float,
                  input_path: str, output_path: str,
                  kalman_Q: float = 0.01, kalman_R: float = 0.1) -> None:
    """
    Process a video file frame by frame.
    Per-frame probabilities are smoothed with a 1-D Kalman filter
    (Syllabus: Kalman filter for smoothing).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (IMG_SIZE, IMG_SIZE))

    kf = KalmanFilter1D(Q=kalman_Q, R=kalman_R)
    frame_idx = 0
    pothole_frames = 0

    print(f"\n{'-'*50}")
    print(f"Video  : {input_path}")
    print(f"Frames : {total}  |  FPS: {fps:.1f}")
    print(f"Kalman : Q={kalman_Q}, R={kalman_R}")
    print(f"{'-'*50}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert BGR→RGB for PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        raw_prob, cam = predict_image(model, pil_img)

        # Kalman smoothing
        smoothed_prob = kf.update(raw_prob)

        annotated = annotate_frame(frame_bgr, smoothed_prob, cam, threshold)

        # Add smoothing indicator
        cv2.putText(annotated,
                    f"raw={raw_prob:.3f}  smooth={smoothed_prob:.3f}",
                    (8, IMG_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(annotated)

        if smoothed_prob >= threshold:
            pothole_frames += 1

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total} | "
                  f"raw={raw_prob:.3f} smooth={smoothed_prob:.3f}")

    cap.release()
    writer.release()

    print(f"\nDone. Pothole frames: {pothole_frames}/{frame_idx} "
          f"({100*pothole_frames/max(frame_idx,1):.1f}%)")
    print(f"Output: {output_path}")
    print("NOTE: Localisation is weakly supervised via Grad-CAM.")
    print("      No box-level metrics (mAP) are claimed.\n")


# ─────────────────────────── CLI Entry Point ─────────────────────────────
def main():
    # Locate project root (directory of this script)
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Pothole Detection Demo — image or video inference with Grad-CAM")
    parser.add_argument("--input",  required=True,
                        help="Path to input image or video file")
    parser.add_argument("--output", default=None,
                        help="Path to annotated output file (image or mp4)")
    parser.add_argument("--model",
                        default=str(script_dir / "pothole_final.pth"),
                        help="Path to model weights (.pth)")
    parser.add_argument("--threshold_file",
                        default=str(script_dir / "threshold.json"),
                        help="Path to threshold.json")
    parser.add_argument("--kalman_q", type=float, default=0.01,
                        help="Kalman process noise (video only)")
    parser.add_argument("--kalman_r", type=float, default=0.1,
                        help="Kalman measurement noise (video only)")
    args = parser.parse_args()

    # Load threshold
    if not Path(args.threshold_file).exists():
        print(f"ERROR: threshold.json not found at {args.threshold_file}")
        print("Run train_and_evaluate.py first to generate it.")
        sys.exit(1)
    with open(args.threshold_file) as f:
        tj = json.load(f)
    threshold = float(tj["threshold"])
    print(f"Loaded threshold={threshold:.2f} (val F1={tj.get('val_f1', '?')})")

    # Load model config if present (to set IMG_SIZE)
    global IMG_SIZE, preprocess
    config_path = Path(args.model).parent / "model_config.json"
    if config_path.exists():
        with open(config_path) as cf:
            mcfg = json.load(cf)
        IMG_SIZE = mcfg.get("img_size", IMG_SIZE)
        preprocess = make_preprocess(IMG_SIZE)
        print(f"Loaded model config: img_size={IMG_SIZE}")

    # Load model
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Run train_and_evaluate.py first to generate pothole_final.pth.")
        sys.exit(1)
    model = load_model(args.model)
    print(f"Loaded model from {args.model}")

    input_path = args.input
    ext = Path(input_path).suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        out = args.output or str(Path(input_path).stem + "_annotated.mp4")
        process_video(model, threshold, input_path, out,
                      kalman_Q=args.kalman_q, kalman_R=args.kalman_r)
    else:
        out = args.output or str(Path(input_path).stem + "_annotated.jpg")
        process_image(model, threshold, input_path, out)


if __name__ == "__main__":
    main()

