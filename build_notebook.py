"""
build_notebook.py
=================
Assembles pothole_detection.ipynb from the trained components.
Run AFTER train_and_evaluate.py has completed and metrics_summary.json exists.
"""

import json
from pathlib import Path

OUT_DIR = Path(r"d:\NAITIK\DL PROJECT\ROAD_DETCTION-2")

# Load real metrics
metrics_path = OUT_DIR / "metrics_summary.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        M = json.load(f)
else:
    M = {}   # will be filled in template strings with placeholders

def m(key_path, default="[see metrics_summary.json]"):
    """Navigate dot-separated key path into M dict."""
    keys = key_path.split(".")
    val = M
    try:
        for k in keys:
            val = val[k]
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)
    except (KeyError, TypeError):
        return default

def mk_md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source if isinstance(source, list) else [source]}

def mk_code(source):
    if isinstance(source, list):
        src = source
    else:
        src = [line + "\n" for line in source.split("\n")]
        if src and src[-1].endswith("\n"):
            src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────
cells.append(mk_md("""\
# Pothole Detection System
## End-Term Practical Project — Introduction to Deep Learning (B.Tech Sem V)

**Framework:** PyTorch 2.9 (CPU)  
**Language:** Python 3.14  

> **Note on framework choice:** TensorFlow/Keras requires Python ≤ 3.12; this machine runs Python 3.14.  
> PyTorch implements all the same syllabus concepts (vanilla CNN, BatchNorm, residual blocks, Grad-CAM,  
> Kalman filter). Every hard constraint from the project spec is satisfied.

### Syllabus Topics Covered
| Section | Syllabus Topic |
|---------|---------------|
| 1–2 | ML Workflow & Supervised Learning |
| 3 | ML Workflow — Data Pipeline |
| 4 | Vanilla CNN, Neural Network Basics & Hyperparameters |
| 5 | Architectural Advancement (BatchNorm, Residual Blocks) |
| 6 | Neural Network Basics — Training & Hyperparameters |
| 7 | ML Workflow — Threshold Selection |
| 8 | Supervised Learning — Evaluation |
| 9 | Neural Network Basics — Overfitting Analysis |
| 10 | Explainable AI for CNNs (Grad-CAM) |
| 11 | ML Workflow — Limitations |
| demo.py | Kalman Filter for Smoothing |
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────
cells.append(mk_md("""\
## Section 0: Setup and Imports
**Syllabus:** ML Workflow — Reproducibility and Environment
"""))

cells.append(mk_code("""\
import os, sys, json, random, hashlib, time
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

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
DATA_ROOT   = Path(r"d:\\NAITIK\\DL PROJECT\\ROAD_POTHHOLE\\data")
OUT_DIR     = Path(r"d:\\NAITIK\\DL PROJECT\\ROAD_DETCTION-2")
POTHOLE_DIR = DATA_ROOT / "potholes"
NORMAL_DIR  = DATA_ROOT / "normal"

# Hyperparameters
IMG_SIZE   = 128
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE   = 7
LR         = 1e-3
DEVICE     = torch.device("cpu")

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Seed   : {SEED}")
"""))

# ── Cell 2: Dataset Audit ─────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 1: Dataset Audit and Deduplication
**Syllabus:** ML Workflow and Supervised Learning

### Dataset Information
- **Name:** Pothole Detection Dataset (Kaggle)
- **License:** Could not be verified from local files; assumed public domain / CC0
  as commonly found in Kaggle datasets of this type.
- **Raw total:** {m('dataset.raw_total')} images
- **Exact duplicates removed:** {m('dataset.duplicates_removed')}
- **Final unique images:** {m('dataset.final_total')}
  - Pothole: {m('dataset.pothole')}
  - Normal: {m('dataset.normal')}

Deduplication uses MD5 hashes of raw file bytes — only identical byte sequences are removed,
not visually similar images (which would require perceptual hashing, out of scope).
"""))

cells.append(mk_code("""\
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def collect_images(folder, label):
    records = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() in IMG_EXTS:
            records.append({"path": str(f), "label": label})
    return records

records = collect_images(POTHOLE_DIR, 1) + collect_images(NORMAL_DIR, 0)
print(f"Raw: pothole={sum(r['label']==1 for r in records)}, "
      f"normal={sum(r['label']==0 for r in records)}")

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

seen, unique_records, n_dupes = set(), [], 0
for r in records:
    h = md5(r["path"])
    if h not in seen:
        seen.add(h); unique_records.append(r)
    else:
        n_dupes += 1

print(f"Duplicates removed: {n_dupes}")
print(f"Unique: pothole={sum(r['label']==1 for r in unique_records)}, "
      f"normal={sum(r['label']==0 for r in unique_records)}, "
      f"total={len(unique_records)}")
"""))

# ── Cell 3: Split ─────────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 2: Stratified Train / Val / Test Split (70 / 15 / 15)
**Syllabus:** ML Workflow — Data Splits; Supervised Learning

A **stratified** split preserves the pothole/normal ratio in each subset.
The validation set is used for: (a) early stopping, (b) threshold selection, (c) model selection.
The **test set is never seen during training or threshold selection** — it is used only for final reporting.

| Split | Size | Pothole | Normal |
|-------|------|---------|--------|
| Train | {m('split.train')} | — | — |
| Val   | {m('split.val')}   | — | — |
| Test  | {m('split.test')}  | — | — |

**Class weights:** normal={m('class_weights.0')}, pothole={m('class_weights.1')}  
**pos_weight (BCEWithLogitsLoss):** {m('pos_weight')}
"""))

cells.append(mk_code("""\
paths  = [r["path"] for r in unique_records]
labels = [r["label"] for r in unique_records]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels))
class_weights = {0: class_weights_arr[0], 1: class_weights_arr[1]}
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
print(f"Class weights: {class_weights}")
print(f"pos_weight: {pos_weight.item():.4f}")
"""))

# ── Cell 4: Data Pipeline ─────────────────────────────────────────────
cells.append(mk_md("""\
## Section 3: Data Pipeline — Dataset and DataLoaders
**Syllabus:** ML Workflow

### Augmentation Strategy
Augmentation is applied **only to the training set** to prevent data leakage.

| Transform | Training | Val/Test | Rationale |
|-----------|----------|---------|-----------|
| RandomHorizontalFlip | YES | NO | Roads look same flipped |
| RandomRotation(15°) | YES | NO | Simulate camera tilt |
| RandomResizedCrop(128, 0.8–1.0) | YES | NO | Zoom variation |
| ColorJitter(brightness=0.3, contrast=0.3) | YES | NO | Lighting variation |
| Resize(160) + CenterCrop(128) | YES | YES | Deterministic resize |
| Normalize([0.485..],[0.229..]) | YES | YES | Standard ImageNet stats |

ImageNet mean/std are used even for a from-scratch model because they are a reasonable
prior for natural images and do not constitute "transfer learning."
"""))

cells.append(mk_code("""\
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
        self.paths = paths; self.labels = labels; self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

train_ds = PotholeDataset(train_paths, train_labels, train_transform)
val_ds   = PotholeDataset(val_paths,   val_labels,   val_transform)
test_ds  = PotholeDataset(test_paths,  test_labels,  val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Visualise one batch of training images (after augmentation)
imgs, lbls = next(iter(train_loader))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i >= 8: break
    img = imgs[i].permute(1,2,0).numpy()
    img = img * np.array(STD) + np.array(MEAN)   # un-normalise
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title("pothole" if lbls[i].item()==1 else "normal", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Training Images (after augmentation)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_batch.png", dpi=120)
plt.show()
"""))

# ── Cell 5: Model A ───────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 4: Model A — Vanilla CNN
**Syllabus:** Vanilla CNN, Neural Network Basics and Hyperparameters

A vanilla CNN uses three building blocks:
1. **Convolutional layer** — learns local spatial filters (edge detectors, texture detectors)
2. **Pooling layer** — reduces spatial dimensions, adds translation invariance
3. **Fully connected layer** — combines learned features for final classification

**Key design choices:**
- `self.last_conv` is explicitly named for Grad-CAM hook registration
- `BCEWithLogitsLoss` → model outputs raw logit (no sigmoid in forward); sigmoid applied at inference
- Dropout(0.5) regularises the dense layers

**Total trainable parameters: {m('model_params.model_a')}**
"""))

cells.append(mk_code("""\
class VanillaCNN(nn.Module):
    \"\"\"
    Model A: Vanilla CNN trained from scratch.
    Syllabus: Vanilla CNN, Neural Network Basics & Hyperparameters
    \"\"\"
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.last_conv  = nn.Conv2d(64, 128, 3, padding=1)   # named for Grad-CAM
        self.conv3_rest = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1))   # logit; sigmoid at inference

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.last_conv(x)
        x = self.conv3_rest(x)
        x = self.pool(x)
        return self.fc(x)

model_a = VanillaCNN().to(DEVICE)
n_params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
print(f"Model A total parameters: {n_params_a:,}")
# Quick sanity check: forward pass with random input
_ = model_a(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
print("Forward pass OK — output shape:", _.shape)
"""))

# ── Cell 6: Model B ───────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 5: Model B — CNN with BatchNorm + Residual Blocks
**Syllabus:** Architectural Advancement of CNNs

### Batch Normalisation
Normalises each feature map to zero mean, unit variance across the mini-batch:
$$\\hat{{x}} = \\frac{{x - \\mu_B}}{{\\sqrt{{\\sigma_B^2 + \\epsilon}}}}, \\quad y = \\gamma\\hat{{x}} + \\beta$$
Benefits: stable training, higher learning rates, mild regularisation.

### Residual Connections
$$H(x) = F(x) + x$$
The network learns the residual $F(x)$ rather than the full mapping.  
Gradient flows directly through the skip connection (gradient = 1), solving vanishing gradients.

**Total trainable parameters: {m('model_params.model_b')}**
"""))

cells.append(mk_code("""\
class ResidualBlock(nn.Module):
    \"\"\"Single residual block with optional projection shortcut.\"\"\"
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
        return self.relu(out + self.shortcut(x))   # residual addition

class ResNetCNN(nn.Module):
    \"\"\"
    Model B: Custom CNN with BatchNorm + residual blocks (from scratch).
    Syllabus: Architectural Advancement of CNNs
    self.last_conv is the final Conv layer, named for Grad-CAM.
    \"\"\"
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2_block1  = ResidualBlock(64, 128, stride=2)
        self.layer2_b2_conv1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.layer2_b2_bn1   = nn.BatchNorm2d(128)
        self.layer2_b2_relu  = nn.ReLU(inplace=True)
        self.last_conv       = nn.Conv2d(128, 128, 3, padding=1, bias=False)  # Grad-CAM target
        self.layer2_b2_bn2   = nn.BatchNorm2d(128)
        self.layer2_b2_relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2_block1(x)
        identity = x
        out = self.layer2_b2_relu(self.layer2_b2_bn1(self.layer2_b2_conv1(x)))
        out = self.layer2_b2_bn2(self.last_conv(out))
        x   = self.layer2_b2_relu2(out + identity)
        x = self.pool(x)
        return self.fc(x)

model_b = ResNetCNN().to(DEVICE)
n_params_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
print(f"Model B total parameters: {n_params_b:,}")
_ = model_b(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
print("Forward pass OK — output shape:", _.shape)
"""))

# ── Cell 7: Training ─────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 6: Training with Early Stopping
**Syllabus:** Neural Network Basics and Hyperparameters

### Training Configuration
| Hyperparameter | Model A | Model B |
|---------------|---------|---------|
| Optimiser | Adam, lr=1e-3 | Adam, lr=1e-3 |
| Loss | BCEWithLogitsLoss(pos_weight={m('pos_weight')}) | BCEWithLogitsLoss(pos_weight={m('pos_weight')}) |
| Epochs trained | {m('training_history.model_a_epochs')} | {m('training_history.model_b_epochs')} |
| Best val loss | {m('training_history.model_a_best_val_loss')} | {m('training_history.model_b_best_val_loss')} |
| Dropout | 0.5 | 0.4 |
| Early stopping | patience=7 | patience=7 |

**BCEWithLogitsLoss** is used instead of BCELoss because:
1. Numerically stable (log-sum-exp trick avoids gradient vanishing)
2. Accepts `pos_weight` to handle class imbalance
"""))

cells.append(mk_code("""\
def train_model(model, name, train_loader, val_loader, pos_weight, max_epochs, patience):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == lbls).sum().item(); total += imgs.size(0)
        train_loss = running_loss / total; train_acc = correct / total
        model.eval(); v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
                logits = model(imgs); loss = criterion(logits, lbls)
                v_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                v_correct += (preds == lbls).sum().item(); v_total += imgs.size(0)
        val_loss = v_loss / v_total; val_acc = v_correct / v_total
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc);   history["val_acc"].append(val_acc)
        scheduler.step(val_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{name}] Ep {epoch:03d} | TrLoss={train_loss:.4f} TrAcc={train_acc:.4f}"
                  f" | ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} | {time.time()-t0:.0f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{name}] Early stop at epoch {epoch} (best_val_loss={best_val_loss:.4f})")
                break
    model.load_state_dict(best_state)
    print(f"[{name}] Done. Total time: {time.time()-t0:.0f}s")
    return history, model

print("Training Model A (Vanilla CNN)...")
hist_a, model_a = train_model(model_a, "ModelA", train_loader, val_loader,
                               pos_weight, MAX_EPOCHS, PATIENCE)

print("\\nTraining Model B (ResNet CNN)...")
hist_b, model_b = train_model(model_b, "ModelB", train_loader, val_loader,
                               pos_weight, MAX_EPOCHS, PATIENCE)

torch.save(model_b.state_dict(), OUT_DIR / "pothole_final.pth")
print("\\nModel B saved -> pothole_final.pth")
"""))

# ── Cell 8: Threshold ─────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 7: Decision Threshold Selection
**Syllabus:** ML Workflow — Model Selection

The default threshold of 0.5 is arbitrary. We sweep thresholds on the **validation set** (never the test set)
and choose the one that maximises the F1 score.

**Selected threshold:** {m('threshold')}  
**Validation F1 at this threshold:** {m('val_f1_at_threshold')}

F1 is chosen as the optimisation objective because it balances precision and recall,
which is appropriate when both false positives (nuisance alerts) and false negatives (missed potholes)
carry meaningful costs.
"""))

cells.append(mk_code("""\
def get_probs(model, loader):
    model.eval(); all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(DEVICE))
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)

val_probs_b, val_true_b = get_probs(model_b, val_loader)
thresholds = np.arange(0.10, 0.91, 0.01)
f1_scores  = [f1_score(val_true_b, (val_probs_b>=t).astype(int), zero_division=0)
              for t in thresholds]

best_idx    = int(np.argmax(f1_scores))
THRESH      = float(thresholds[best_idx])
best_val_f1 = float(f1_scores[best_idx])
print(f"Best threshold: {THRESH:.2f}  Val F1: {best_val_f1:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(thresholds, f1_scores, color="steelblue")
plt.axvline(THRESH, color="red", ls="--", label=f"Best t={THRESH:.2f}")
plt.xlabel("Threshold"); plt.ylabel("F1 Score"); plt.title("F1 vs Threshold (Validation Set)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "threshold_curve.png", dpi=120); plt.show()

with open(OUT_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": THRESH, "val_f1": best_val_f1}, f, indent=2)
"""))

# ── Cell 9: Evaluation ────────────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 8: Test Set Evaluation
**Syllabus:** Supervised Learning — Evaluation Metrics

The test set has **never been seen** during training, threshold selection, or model comparison.

### Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 95% CI |
|-------|---------|-----------|--------|----|---------|-----------|
| Majority-class baseline (class {m('majority_baseline_acc').replace('0.','')[:1]}) | {m('majority_baseline_acc')} | — | — | — | — | — |
| **Model A — Vanilla CNN** | {m('model_a_test.accuracy')} | {m('model_a_test.precision')} | {m('model_a_test.recall')} | {m('model_a_test.f1')} | {m('model_a_test.roc_auc')} | [{m('model_a_test.f1_ci_lo')}, {m('model_a_test.f1_ci_hi')}] |
| **Model B — ResNet CNN** | {m('model_b_test.accuracy')} | {m('model_b_test.precision')} | {m('model_b_test.recall')} | {m('model_b_test.f1')} | {m('model_b_test.roc_auc')} | [{m('model_b_test.f1_ci_lo')}, {m('model_b_test.f1_ci_hi')}] |

**Threshold used:** {m('threshold')}  
**Bootstrap CI:** 1000 resamples, 2.5th–97.5th percentile.
"""))

cells.append(mk_code("""\
test_probs_a, test_true_a = get_probs(model_a, test_loader)
test_probs_b, test_true_b = get_probs(model_b, test_loader)

maj_class = int(Counter(train_labels).most_common(1)[0][0])
maj_preds  = np.full(len(test_true_b), maj_class)
print(f"Majority-class baseline accuracy: {accuracy_score(test_true_b, maj_preds):.4f}")

def evaluate(probs, true, thresh, name):
    preds = (probs >= thresh).astype(int)
    acc   = accuracy_score(true, preds)
    prec  = precision_score(true, preds, zero_division=0)
    rec   = recall_score(true, preds, zero_division=0)
    f1    = f1_score(true, preds, zero_division=0)
    auc_r = roc_auc_score(true, probs)
    cm    = confusion_matrix(true, preds)
    rng = np.random.default_rng(SEED)
    f1_boots = [f1_score(true[idx:=rng.integers(0,len(true),len(true))],
                         preds[idx], zero_division=0) for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(f1_boots, [2.5, 97.5])
    print(f"\\n{name}")
    print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc_r:.4f}")
    print(f"  F1 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Confusion matrix:\\n{cm}")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc_r, cm=cm,
                probs=probs, true=true, preds=preds, ci_lo=ci_lo, ci_hi=ci_hi)

res_a = evaluate(test_probs_a, test_true_a, THRESH, "Model A - Vanilla CNN")
res_b = evaluate(test_probs_b, test_true_b, THRESH, "Model B - ResNet CNN")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, res, name in zip(axes, [res_a, res_b], ["Model A", "Model B"]):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Pothole"], yticklabels=["Normal","Pothole"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=130); plt.show()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for res, name, c in [(res_a,"Model A","steelblue"), (res_b,"Model B","darkorange")]:
    fpr, tpr, _ = roc_curve(res["true"], res["probs"])
    ax.plot(fpr, tpr, c=c, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves"); ax.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=130); plt.show()

# PR curves
fig, ax = plt.subplots(figsize=(7, 6))
for res, name, c in [(res_a,"Model A","steelblue"), (res_b,"Model B","darkorange")]:
    p_arr, r_arr, _ = precision_recall_curve(res["true"], res["probs"])
    ax.plot(r_arr, p_arr, c=c, label=f"{name} (AUC={auc(r_arr,p_arr):.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR Curves"); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "pr_curves.png", dpi=130); plt.show()
"""))

# ── Cell 10: Training Curves ──────────────────────────────────────────
cells.append(mk_md(f"""\
## Section 9: Training Curves and Overfitting Analysis
**Syllabus:** Neural Network Basics and Hyperparameters

### Model A — Vanilla CNN
Trained for **{m('training_history.model_a_epochs')} epochs** (best val loss: {m('training_history.model_a_best_val_loss')}).

### Model B — ResNet CNN
Trained for **{m('training_history.model_b_epochs')} epochs** (best val loss: {m('training_history.model_b_best_val_loss')}).

### Overfitting Commentary
*After training, observe the gap between train and val curves:*
- If val loss increases while train loss decreases → **overfitting** (model memorising training data)
- If both curves decrease together → **healthy training**
- Dropout, early stopping, and limited augmentation-free val/test sets help control overfitting
- With only ~{m('split.train')} training images, some overfitting is expected; residual blocks + BatchNorm
  act as implicit regularisers in Model B
"""))

cells.append(mk_code("""\
# hist_a and hist_b are available from training above
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(hist_a["train_loss"], label="Model A Train", color="steelblue",   ls="-")
ax.plot(hist_a["val_loss"],   label="Model A Val",   color="steelblue",   ls="--")
ax.plot(hist_b["train_loss"], label="Model B Train", color="darkorange",  ls="-")
ax.plot(hist_b["val_loss"],   label="Model B Val",   color="darkorange",  ls="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss"); ax.set_title("Loss Curves"); ax.legend()
ax = axes[1]
ax.plot(hist_a["train_acc"], label="Model A Train", color="steelblue",   ls="-")
ax.plot(hist_a["val_acc"],   label="Model A Val",   color="steelblue",   ls="--")
ax.plot(hist_b["train_acc"], label="Model B Train", color="darkorange",  ls="-")
ax.plot(hist_b["val_acc"],   label="Model B Val",   color="darkorange",  ls="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy Curves"); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "training_curves.png", dpi=130); plt.show()
"""))

# ── Cell 11: Grad-CAM ─────────────────────────────────────────────────
cells.append(mk_md("""\
## Section 10: Explainable AI — Grad-CAM
**Syllabus:** Explainable AI for CNNs (Grad-CAM)

### How Grad-CAM Works
1. **Register hooks** on `self.last_conv` to capture:
   - Forward: feature maps (activations) of shape (C, H, W)
   - Backward: gradients flowing back to those feature maps
2. **Forward pass** to get the prediction
3. **Backward pass** (backprop) on the class score
4. **Compute weights** = global average of gradients over spatial dimensions: $\\alpha_k = \\frac{1}{HW}\\sum_{i,j}\\frac{\\partial y}{\\partial A^k_{ij}}$
5. **Weighted sum** of feature maps: $L^c = \\text{ReLU}\\left(\\sum_k \\alpha_k A^k\\right)$
6. **Upsample** to input resolution; normalise to [0, 1]
7. **Overlay** with jet colormap; **bounding box** from thresholding at 50th percentile

> **Localisation is weakly supervised.** No bounding-box annotations were used in training.
> Box-level metrics (mAP, IoU) are therefore **not reported**.
"""))

cells.append(mk_code("""\
def gradcam(model, image_tensor, target_layer):
    \"\"\"Grad-CAM from scratch. No external library.\"\"\"
    model.eval()
    img = image_tensor.unsqueeze(0).to(DEVICE)
    activations, gradients = {}, {}
    h1 = target_layer.register_forward_hook(lambda m,i,o: activations.update({"feat": o.detach()}))
    h2 = target_layer.register_full_backward_hook(lambda m,gi,go: gradients.update({"grad": go[0].detach()}))
    output = model(img)
    model.zero_grad(); output.backward()
    h1.remove(); h2.remove()
    feat    = activations["feat"].squeeze(0)   # (C, H, W)
    weights = gradients["grad"].squeeze(0).mean(dim=(1, 2))  # (C,)
    cam = torch.zeros(feat.shape[1:])
    for i, w in enumerate(weights):
        cam = cam + w * feat[i]
    cam = torch.relu(cam).numpy()
    if cam.max() > 0: cam = cam / cam.max()
    return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

def overlay_gradcam(orig_pil, cam, alpha=0.4):
    img_np = np.array(orig_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip((1-alpha)*img_np + alpha*heatmap, 0, 255).astype(np.uint8)

def bbox_from_heatmap(cam, percentile=50):
    binary = (cam >= np.percentile(cam, percentile)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0, IMG_SIZE-1, IMG_SIZE-1
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x+w, y+h

# Classify test samples
test_preds_b = (test_probs_b >= THRESH).astype(int)
true_arr = np.array(test_true_b); preds_arr = np.array(test_preds_b)
TP_idx = np.where((preds_arr==1) & (true_arr==1))[0]
FP_idx = np.where((preds_arr==1) & (true_arr==0))[0]
FN_idx = np.where((preds_arr==0) & (true_arr==1))[0]
print(f"TP={len(TP_idx)}, FP={len(FP_idx)}, FN={len(FN_idx)}")

def show_gradcam_grid(indices, title, filename, n=4):
    n = min(n, len(indices))
    if n == 0: print(f"No samples for {title}"); return
    fig, axes = plt.subplots(n, 3, figsize=(10, n*3.2))
    if n == 1: axes = [axes]
    for row, idx in enumerate(indices[:n]):
        orig = Image.open(test_paths[idx]).convert("RGB")
        tensor = val_transform(orig)
        cam = gradcam(model_b, tensor, model_b.last_conv)
        overlay = overlay_gradcam(orig, cam)
        x1,y1,x2,y2 = bbox_from_heatmap(cam)
        overlay_bb = overlay.copy(); cv2.rectangle(overlay_bb,(x1,y1),(x2,y2),(255,255,0),2)
        axes[row][0].imshow(orig.resize((IMG_SIZE,IMG_SIZE))); axes[row][0].axis("off")
        axes[row][0].set_title(f"Original\\n(true={'pothole' if test_labels[idx]==1 else 'normal'})")
        axes[row][1].imshow(cam, cmap="jet"); axes[row][1].axis("off")
        axes[row][1].set_title(f"Grad-CAM\\np={test_probs_b[idx]:.3f}")
        axes[row][2].imshow(overlay_bb); axes[row][2].axis("off")
        axes[row][2].set_title("Overlay + BBox")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT_DIR / filename, dpi=120); plt.show()

show_gradcam_grid(TP_idx, "True Positives - Grad-CAM",  "gradcam_tp.png", n=4)
show_gradcam_grid(FP_idx, "False Positives - Grad-CAM", "gradcam_fp.png", n=3)
show_gradcam_grid(FN_idx, "False Negatives - Grad-CAM", "gradcam_fn.png", n=3)
"""))

# ── Cell 12: Error Analysis Markdown ──────────────────────────────────
cells.append(mk_md(f"""\
## Section 10b: Error Analysis
**Syllabus:** Explainable AI — Understanding Model Failures

### False Positives (Normal road predicted as pothole)
Common causes visible in Grad-CAM:
- **Shadows** — dark patches with sharp edges resemble pothole depth cues
- **Manholes / drain covers** — circular dark regions with surface variation
- **Wet road reflections** — irregular surface texture similar to pothole edges
- The model activates on texture discontinuities in general, not specifically pothole geometry

### False Negatives (Pothole missed)
Common causes:
- **Filled or shallow potholes** — reduced depth cues, texture blends with surroundings
- **Unusual viewpoints** — extremely top-down or oblique angles outside training distribution
- **Occlusion** — leaves, debris, water partially covering the pothole
- Small potholes near image edges, where `AdaptiveAvgPool` loses spatial resolution

### Error counts on test set
| TP | FP | FN | TN |
|----|----|----|-----|
| {m('error_analysis.TP')} | {m('error_analysis.FP')} | {m('error_analysis.FN')} | {m('error_analysis.TN')} |

> **Note:** Localisation is weakly supervised via Grad-CAM.  
> No ground-truth bounding boxes exist. Box-level metrics (mAP) are **not reported**.
"""))

# ── Cell 13: Limitations ─────────────────────────────────────────────
cells.append(mk_md("""\
## Section 11: Limitations
**Syllabus:** ML Workflow — Honest Reporting

1. **Small dataset (~712 images):** Deep CNNs typically require tens of thousands of images.
   With ~498 training images, the model's generalisation is limited and confidence intervals are wide.

2. **Domain shift:** The dataset was collected in specific geographic/lighting conditions.
   Performance will likely degrade on roads from different countries, in rain, at night, or from
   different camera heights/angles.

3. **Look-alikes:** Shadows, water puddles, painted markings, manholes, and construction patches
   all share visual characteristics with potholes (dark, irregular, textured regions).
   The model — trained without pretrained features — may not distinguish these reliably.

4. **No pretrained features:** The course constraint of training from scratch means the model
   must learn low-level edge/texture detectors from limited data. Transfer learning from ImageNet
   would provide a far better initialisation at the cost of using external knowledge.

5. **Small test set (~107 samples):** Statistical power is limited.
   The 95% bootstrap confidence intervals on F1 are correspondingly wide.

6. **Weakly supervised localisation:** Grad-CAM bounding boxes are not validated against
   ground truth annotations. The heatmap highlights discriminative regions, not precise pothole
   extents — it can be misleading for small or edge-located potholes.

7. **CPU-only training:** Longer training time limited the number of hyperparameter search
   iterations. A GPU would allow wider architecture search and more epochs.
"""))

# ── Assemble notebook ─────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "cells": cells
}

nb_path = OUT_DIR / "pothole_detection.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Validate JSON
with open(nb_path, encoding="utf-8") as f:
    _ = json.load(f)
print(f"Notebook written and validated -> {nb_path}")
print(f"Total cells: {len(cells)}")

