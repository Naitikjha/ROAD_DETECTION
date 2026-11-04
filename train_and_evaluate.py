"""
train_and_evaluate.py
=====================
Pothole Detection System -- Full Training & Evaluation Pipeline
Course: Introduction to Deep Learning (B.Tech Sem V)
Framework: PyTorch 2.9 (CPU)

Covers:
  - ML workflow and supervised learning
  - Neural network basics and hyperparameters
  - Vanilla CNN (Model A)
  - Architectural advancement: BatchNorm + residual blocks (Model B)
  - Explainable AI: Grad-CAM
  - Kalman filter for video smoothing (in demo.py)
"""

import os, sys, json, random, hashlib, time, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)

# Force utf-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -- reproducibility --
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- paths --
DATA_ROOT  = Path(r"d:\NAITIK\DL PROJECT\ROAD_POTHHOLE\data")
OUT_DIR    = Path(r"d:\NAITIK\DL PROJECT\ROAD_DETCTION-2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POTHOLE_DIR = DATA_ROOT / "potholes"
NORMAL_DIR  = DATA_ROOT / "normal"

IMG_SIZE   = 128       # reduced from 224 for CPU speed; still effective
BATCH_SIZE = 32
MAX_EPOCHS = 30        # early stopping will kick in before this
PATIENCE   = 7
LR         = 1e-3
DEVICE     = torch.device("cpu")

print("=" * 60)
print("POTHOLE DETECTION SYSTEM - Training Pipeline")
print("=" * 60)
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Seed   : {SEED}")
print(f"Image  : {IMG_SIZE}x{IMG_SIZE}")
print(flush=True)

# ==================================================================
# SECTION 1: Dataset Audit & Deduplication
# ==================================================================
print("-" * 60)
print("SECTION 1: Dataset Audit")
print("-" * 60, flush=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def collect_images(folder, label):
    records = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() in IMG_EXTS:
            records.append({"path": str(f), "label": label})
    return records

records = collect_images(POTHOLE_DIR, 1) + collect_images(NORMAL_DIR, 0)
print(f"Raw image count -- pothole: {sum(r['label']==1 for r in records)}, "
      f"normal: {sum(r['label']==0 for r in records)}, total: {len(records)}")

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

seen, unique_records = set(), []
n_dupes = 0
for r in records:
    h = md5(r["path"])
    if h not in seen:
        seen.add(h)
        unique_records.append(r)
    else:
        n_dupes += 1

print(f"Exact duplicates removed: {n_dupes}")
print(f"Unique images -- pothole: {sum(r['label']==1 for r in unique_records)}, "
      f"normal: {sum(r['label']==0 for r in unique_records)}, "
      f"total: {len(unique_records)}")
print()
print("Dataset  : Pothole Detection Dataset (Kaggle)")
print("License  : Could not be verified from local files; assumed public domain / CC0")
print(flush=True)

# ==================================================================
# SECTION 2: Stratified Split 70/15/15
# ==================================================================
print("-" * 60)
print("SECTION 2: Stratified Train/Val/Test Split (70 / 15 / 15)")
print("-" * 60, flush=True)

paths  = [r["path"] for r in unique_records]
labels = [r["label"] for r in unique_records]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED)

print(f"Train : {len(train_paths):4d}  (pothole={sum(train_labels)}, "
      f"normal={len(train_labels)-sum(train_labels)})")
print(f"Val   : {len(val_paths):4d}  (pothole={sum(val_labels)}, "
      f"normal={len(val_labels)-sum(val_labels)})")
print(f"Test  : {len(test_paths):4d}  (pothole={sum(test_labels)}, "
      f"normal={len(test_labels)-sum(test_labels)})")

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=np.array(train_labels))
class_weights = {0: class_weights_arr[0], 1: class_weights_arr[1]}
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
print(f"Class weights -- normal: {class_weights[0]:.4f}, pothole: {class_weights[1]:.4f}")
print(f"pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")
print(flush=True)

# ==================================================================
# SECTION 3: Data Pipeline
# ==================================================================
print("-" * 60)
print("SECTION 3: Data Pipeline")
print("-" * 60, flush=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_transform = T.Compose([
    T.Resize(IMG_SIZE + 32),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class PotholeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

train_ds = PotholeDataset(train_paths, train_labels, train_transform)
val_ds   = PotholeDataset(val_paths,   val_labels,   val_transform)
test_ds  = PotholeDataset(test_paths,  test_labels,  val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
      f"Test batches: {len(test_loader)}")
print(flush=True)

# ==================================================================
# SECTION 4 & 5: Model Definitions
# ==================================================================
print("-" * 60)
print("SECTION 4 & 5: Model Definitions")
print("-" * 60, flush=True)

# -- Model A: Vanilla CNN --
class VanillaCNN(nn.Module):
    """Vanilla CNN trained from scratch. (Syllabus: Vanilla CNN)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.last_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_rest = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.last_conv(x)
        x = self.conv3_rest(x)
        x = self.pool(x)
        return self.fc(x)

# -- Model B: CNN + BatchNorm + Residual Blocks --
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
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
    """CNN with BatchNorm + residual blocks (from scratch). (Syllabus: Architectural Advancement)"""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64))

        self.layer2_block1 = ResidualBlock(64, 128, stride=2)
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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_a = VanillaCNN().to(DEVICE)
model_b = ResNetCNN().to(DEVICE)
print(f"Model A (VanillaCNN)  params: {count_params(model_a):,}")
print(f"Model B (ResNetCNN)   params: {count_params(model_b):,}")
print(flush=True)

# ==================================================================
# SECTION 6: Training Loop
# ==================================================================
print("-" * 60)
print("SECTION 6: Training")
print("-" * 60, flush=True)

def train_model(model, name, train_loader, val_loader, pos_weight, max_epochs, patience):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == lbls).sum().item()
            total   += imgs.size(0)
        train_loss = running_loss / total
        train_acc  = correct / total

        # -- Validate --
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
                logits = model(imgs)
                loss   = criterion(logits, lbls)
                v_loss    += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                v_correct += (preds == lbls).sum().item()
                v_total   += imgs.size(0)
        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"[{name}] Ep {epoch:02d}/{max_epochs} | "
              f"TrL={train_loss:.4f} TrA={train_acc:.3f} | "
              f"VaL={val_loss:.4f} VaA={val_acc:.3f} | "
              f"{elapsed:.0f}s", flush=True)

        # -- Early stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{name}] Early stopping at epoch {epoch} "
                      f"(best val loss={best_val_loss:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    total_time = time.time() - t0
    print(f"[{name}] Training complete. Best val loss={best_val_loss:.4f}. "
          f"Total time: {total_time:.0f}s", flush=True)
    return history, model

print("\n-- Training Model A (VanillaCNN) --", flush=True)
hist_a, model_a = train_model(model_a, "A", train_loader, val_loader,
                               pos_weight, MAX_EPOCHS, PATIENCE)

print("\n-- Training Model B (ResNetCNN) --", flush=True)
hist_b, model_b = train_model(model_b, "B", train_loader, val_loader,
                               pos_weight, MAX_EPOCHS, PATIENCE)

# Save Model B as the final model
torch.save(model_b.state_dict(), OUT_DIR / "pothole_final.pth")
print(f"\nSaved Model B -> {OUT_DIR / 'pothole_final.pth'}", flush=True)

model_config = {"model_type": "ResNetCNN", "img_size": IMG_SIZE, "mean": MEAN, "std": STD}
with open(OUT_DIR / "model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

# ==================================================================
# SECTION 7: Threshold Selection on Validation Set
# ==================================================================
print()
print("-" * 60)
print("SECTION 7: Threshold Selection (maximise F1 on val set)")
print("-" * 60, flush=True)

def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)

val_probs_b, val_true_b = get_probs(model_b, val_loader)

thresholds = np.arange(0.10, 0.91, 0.01)
f1_scores_arr  = []
for t in thresholds:
    preds = (val_probs_b >= t).astype(int)
    f1_scores_arr.append(f1_score(val_true_b, preds, zero_division=0))

best_idx   = int(np.argmax(f1_scores_arr))
best_thresh = float(thresholds[best_idx])
best_val_f1 = float(f1_scores_arr[best_idx])

print(f"Best threshold: {best_thresh:.2f}  (val F1={best_val_f1:.4f})", flush=True)

with open(OUT_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": best_thresh, "val_f1": best_val_f1}, f, indent=2)
print(f"Saved -> {OUT_DIR / 'threshold.json'}", flush=True)

# ==================================================================
# SECTION 8: Test Set Evaluation
# ==================================================================
print()
print("-" * 60)
print("SECTION 8: Test Set Evaluation")
print("-" * 60, flush=True)

test_probs_a, test_true_a = get_probs(model_a, test_loader)
test_probs_b, test_true_b = get_probs(model_b, test_loader)

THRESH = best_thresh

maj_class  = int(Counter(train_labels).most_common(1)[0][0])
maj_preds  = np.full(len(test_true_b), maj_class)
maj_acc    = accuracy_score(test_true_b, maj_preds)
print(f"\nMajority-class baseline (class={maj_class}): accuracy={maj_acc:.4f}", flush=True)

def evaluate(probs, true, thresh, name):
    preds = (probs >= thresh).astype(int)
    acc   = accuracy_score(true, preds)
    prec  = precision_score(true, preds, zero_division=0)
    rec   = recall_score(true, preds, zero_division=0)
    f1    = f1_score(true, preds, zero_division=0)
    auc_r = roc_auc_score(true, probs)
    cm    = confusion_matrix(true, preds)

    print(f"\n-- {name} (threshold={thresh:.2f}) --")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc_r:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    rng = np.random.default_rng(SEED)
    n   = len(true)
    f1_boots = []
    for _ in range(1000):
        idx    = rng.integers(0, n, size=n)
        b_preds = preds[idx]
        b_true  = true[idx]
        f1_boots.append(f1_score(b_true, b_preds, zero_division=0))
    ci_lo, ci_hi = np.percentile(f1_boots, [2.5, 97.5])
    print(f"  F1 Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]", flush=True)

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc_r,
                cm=cm, probs=probs, true=true, preds=preds,
                ci_lo=ci_lo, ci_hi=ci_hi)

res_a = evaluate(test_probs_a, test_true_a, THRESH, "Model A -- VanillaCNN")
res_b = evaluate(test_probs_b, test_true_b, THRESH, "Model B -- ResNetCNN")

# -- Plots --
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, res, name in zip(axes, [res_a, res_b],
                          ["Model A (VanillaCNN)", "Model B (ResNetCNN)"]):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pothole"],
                yticklabels=["Normal", "Pothole"], ax=ax)
    ax.set_title(f"Confusion Matrix -- {name}")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
plt.close()
print(f"\nSaved -> confusion_matrices.png", flush=True)

fig, ax = plt.subplots(figsize=(7, 6))
for res, name, c in [(res_a, "Model A (Vanilla CNN)", "steelblue"),
                      (res_b, "Model B (ResNet CNN)",  "darkorange")]:
    fpr, tpr, _ = roc_curve(res["true"], res["probs"])
    ax.plot(fpr, tpr, c=c, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],"k--", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves -- Test Set")
ax.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=150); plt.close()
print("Saved -> roc_curves.png", flush=True)

fig, ax = plt.subplots(figsize=(7, 6))
for res, name, c in [(res_a, "Model A (Vanilla CNN)", "steelblue"),
                      (res_b, "Model B (ResNet CNN)",  "darkorange")]:
    prec_arr, rec_arr, _ = precision_recall_curve(res["true"], res["probs"])
    pr_auc = auc(rec_arr, prec_arr)
    ax.plot(rec_arr, prec_arr, c=c, label=f"{name} (AUC={pr_auc:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves -- Test Set")
ax.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves.png", dpi=150); plt.close()
print("Saved -> pr_curves.png", flush=True)

# ==================================================================
# SECTION 9: Training Curves
# ==================================================================
print()
print("-" * 60)
print("SECTION 9: Training Curves")
print("-" * 60, flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(hist_a["train_loss"], label="Model A Train", color="steelblue",   ls="-")
ax.plot(hist_a["val_loss"],   label="Model A Val",   color="steelblue",   ls="--")
ax.plot(hist_b["train_loss"], label="Model B Train", color="darkorange", ls="-")
ax.plot(hist_b["val_loss"],   label="Model B Val",   color="darkorange", ls="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
ax.set_title("Training & Validation Loss"); ax.legend()
ax = axes[1]
ax.plot(hist_a["train_acc"], label="Model A Train", color="steelblue",   ls="-")
ax.plot(hist_a["val_acc"],   label="Model A Val",   color="steelblue",   ls="--")
ax.plot(hist_b["train_acc"], label="Model B Train", color="darkorange", ls="-")
ax.plot(hist_b["val_acc"],   label="Model B Val",   color="darkorange", ls="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Training & Validation Accuracy"); ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150); plt.close()
print("Saved -> training_curves.png", flush=True)

# ==================================================================
# SECTION 10: Grad-CAM
# ==================================================================
print()
print("-" * 60)
print("SECTION 10: Grad-CAM Explainability")
print("-" * 60, flush=True)

def gradcam(model, image_tensor, target_layer):
    model.eval()
    img = image_tensor.unsqueeze(0).to(DEVICE)
    activations = {}
    gradients   = {}
    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        gradients["grad"] = grad_out[0].detach()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    output = model(img)
    model.zero_grad()
    output.backward()
    h1.remove(); h2.remove()
    feat = activations["feat"].squeeze(0)
    grad = gradients["grad"].squeeze(0)
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(feat.shape[1:])
    for i, w in enumerate(weights):
        cam += w * feat[i]
    cam = torch.relu(cam)
    cam = cam.numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam

def overlay_gradcam(original_img_pil, cam, alpha=0.4):
    img_np = np.array(original_img_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * img_np + alpha * heatmap_color
    return overlay.astype(np.uint8)

def bbox_from_heatmap(cam, percentile=50):
    thresh = np.percentile(cam, percentile)
    binary = (cam >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, IMG_SIZE - 1, IMG_SIZE - 1
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h

# Collect test predictions
model_b.eval()
test_preds_b = (test_probs_b >= THRESH).astype(int)
true_arr   = np.array(test_true_b)
preds_arr  = np.array(test_preds_b)
probs_arr  = test_probs_b

TP_idx = np.where((preds_arr == 1) & (true_arr == 1))[0]
FP_idx = np.where((preds_arr == 1) & (true_arr == 0))[0]
FN_idx = np.where((preds_arr == 0) & (true_arr == 1))[0]

print(f"TPs={len(TP_idx)}, FPs={len(FP_idx)}, FNs={len(FN_idx)}", flush=True)

def show_gradcam_grid(indices, title, filename, n=4):
    n = min(n, len(indices))
    if n == 0:
        print(f"  No samples for {title}"); return
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    if n == 1:
        axes = [axes]
    for row, idx in enumerate(indices[:n]):
        path  = test_paths[idx]
        label = test_labels[idx]
        prob  = probs_arr[idx]
        orig_pil = Image.open(path).convert("RGB")
        tensor = val_transform(orig_pil)
        cam = gradcam(model_b, tensor, model_b.last_conv)
        overlay = overlay_gradcam(orig_pil, cam)
        x1, y1, x2, y2 = bbox_from_heatmap(cam)
        overlay_bbox = overlay.copy()
        cv2.rectangle(overlay_bbox, (x1, y1), (x2, y2), (255, 255, 0), 2)
        axes[row][0].imshow(orig_pil.resize((IMG_SIZE, IMG_SIZE)))
        axes[row][0].set_title(f"Original\n(true={'pothole' if label==1 else 'normal'})")
        axes[row][0].axis("off")
        axes[row][1].imshow(cam, cmap="jet")
        axes[row][1].set_title(f"Grad-CAM\np(pothole)={prob:.3f}")
        axes[row][1].axis("off")
        axes[row][2].imshow(overlay_bbox)
        axes[row][2].set_title("Overlay + BBox")
        axes[row][2].axis("off")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=130); plt.close()
    print(f"  Saved -> {filename}", flush=True)

show_gradcam_grid(TP_idx, "True Positives -- Grad-CAM",  "gradcam_tp.png", n=4)
show_gradcam_grid(FP_idx, "False Positives -- Grad-CAM", "gradcam_fp.png", n=3)
show_gradcam_grid(FN_idx, "False Negatives -- Grad-CAM", "gradcam_fn.png", n=3)

# ==================================================================
# Save metrics summary to JSON
# ==================================================================
metrics_summary = {
    "dataset": {
        "name": "Pothole Detection Dataset (Kaggle)",
        "license": "Could not be verified; assumed CC0/public domain",
        "raw_total": len(records),
        "duplicates_removed": n_dupes,
        "final_total": len(unique_records),
        "pothole": int(sum(r["label"]==1 for r in unique_records)),
        "normal":  int(sum(r["label"]==0 for r in unique_records)),
    },
    "split": {
        "train": len(train_paths),
        "val":   len(val_paths),
        "test":  len(test_paths),
    },
    "class_weights": {str(k): float(v) for k, v in class_weights.items()},
    "pos_weight": float(pos_weight.item()),
    "model_params": {
        "model_a": count_params(model_a),
        "model_b": count_params(model_b),
    },
    "threshold": best_thresh,
    "val_f1_at_threshold": best_val_f1,
    "majority_baseline_acc": float(maj_acc),
    "model_a_test": {
        "accuracy":  float(res_a["acc"]),
        "precision": float(res_a["prec"]),
        "recall":    float(res_a["rec"]),
        "f1":        float(res_a["f1"]),
        "roc_auc":   float(res_a["auc"]),
        "f1_ci_lo":  float(res_a["ci_lo"]),
        "f1_ci_hi":  float(res_a["ci_hi"]),
        "confusion_matrix": res_a["cm"].tolist(),
    },
    "model_b_test": {
        "accuracy":  float(res_b["acc"]),
        "precision": float(res_b["prec"]),
        "recall":    float(res_b["rec"]),
        "f1":        float(res_b["f1"]),
        "roc_auc":   float(res_b["auc"]),
        "f1_ci_lo":  float(res_b["ci_lo"]),
        "f1_ci_hi":  float(res_b["ci_hi"]),
        "confusion_matrix": res_b["cm"].tolist(),
    },
    "training_history": {
        "model_a_epochs": len(hist_a["train_loss"]),
        "model_b_epochs": len(hist_b["train_loss"]),
        "model_a_best_val_loss": float(min(hist_a["val_loss"])),
        "model_b_best_val_loss": float(min(hist_b["val_loss"])),
    },
    "error_analysis": {
        "TP": int(len(TP_idx)),
        "FP": int(len(FP_idx)),
        "FN": int(len(FN_idx)),
        "TN": int(len(test_true_b) - len(TP_idx) - len(FP_idx) - len(FN_idx)),
    },
}

with open(OUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
print(f"\nMetrics summary saved -> {OUT_DIR / 'metrics_summary.json'}", flush=True)

print()
print("=" * 60)
print("ALL DONE -- Training & Evaluation Complete")
print("=" * 60)
print(f"\n{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("-" * 65)
print(f"{'Majority-class Baseline':<30} {maj_acc:>7.4f} {'--':>7} {'--':>7} {'--':>7} {'--':>7}")
print(f"{'Model A -- Vanilla CNN':<30} {res_a['acc']:>7.4f} {res_a['prec']:>7.4f} "
      f"{res_a['rec']:>7.4f} {res_a['f1']:>7.4f} {res_a['auc']:>7.4f}")
print(f"{'Model B -- ResNet CNN':<30} {res_b['acc']:>7.4f} {res_b['prec']:>7.4f} "
      f"{res_b['rec']:>7.4f} {res_b['f1']:>7.4f} {res_b['auc']:>7.4f}")
print(f"\nF1 95% CI  Model A: [{res_a['ci_lo']:.4f}, {res_a['ci_hi']:.4f}]")
print(f"F1 95% CI  Model B: [{res_b['ci_lo']:.4f}, {res_b['ci_hi']:.4f}]")
print(f"\nNote: Localisation is weakly supervised via Grad-CAM.")
print("No bounding-box annotations exist; mAP and similar")
print("box-level metrics are NOT reported.", flush=True)
