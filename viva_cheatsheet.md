# Viva Cheat Sheet — Pothole Detection System
**Course:** Introduction to Deep Learning (B.Tech Sem V)
**Framework:** PyTorch 2.9 (CPU) | **Language:** Python 3.14

> **Note:** All numbers below are filled in after actual training runs.
> See `metrics_summary.json` for the complete numeric record.

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| Dataset name | Pothole Detection Dataset (Kaggle) |
| License | Could not be verified from local files; assumed CC0 / public domain |
| Raw images | 712 |
| Exact duplicates removed | 6 (MD5 byte hashing) |
| Final unique images | 706 |
| Pothole images | 362 (51.3%) |
| Normal images | 344 (48.7%) |
| Split | 70% train (494) / 15% val (106) / 15% test (106) (stratified, seed=42) |
| Class weights | Normal: 1.0249, Pothole: 0.9763 (pos_weight = 0.9526) |

---

## 2. Architecture -- Model A: Vanilla CNN

| Layer | Type | Output Shape (128x128 input) | Trainable Params |
|-------|------|------------------------------|-----------------|
| Input | -- | (3, 128, 128) | 0 |
| conv1 | Conv2d(3->32, k=3, p=1) + ReLU + MaxPool2d(2) | (32, 64, 64) | 896 |
| conv2 | Conv2d(32->64, k=3, p=1) + ReLU + MaxPool2d(2) | (64, 32, 32) | 18,496 |
| **last_conv** | **Conv2d(64->128, k=3, p=1)** | **(128, 32, 32)** | **73,856** |
| conv3_rest | ReLU + MaxPool2d(2) | (128, 16, 16) | 0 |
| AdaptiveAvgPool2d(1) | Global average pooling | (128, 1, 1) | 0 |
| Flatten | -- | (128,) | 0 |
| fc1 | Linear(128->256) + ReLU + Dropout(0.5) | (256,) | 33,024 |
| fc2 | Linear(256->1) | (1,) | 257 |
| **Total** | | | **126,529** |

> self.last_conv = the Conv2d(64->128) layer (3rd conv block).

---

## 3. Architecture -- Model B: CNN + BatchNorm + Residual Blocks

| Block | Layers | Output Shape (128x128 input) | Notes |
|-------|--------|------------------------------|-------|
| Stem | Conv2d(3->64, k=7, s=2, p=3) + BN + ReLU + MaxPool(3,s=2,p=1) | (64, 32, 32) | Downsamples 4x |
| Layer1 Block1 | Conv(64->64,s=1)+BN+ReLU + Conv(64->64)+BN + skip | (64, 32, 32) | Identity skip |
| Layer1 Block2 | Conv(64->64,s=1)+BN+ReLU + Conv(64->64)+BN + skip | (64, 32, 32) | Identity skip |
| Layer2 Block1 | Conv(64->128,s=2)+BN+ReLU + Conv(128->128)+BN + proj skip | (128, 16, 16) | Proj skip: Conv(64->128,s=2)+BN |
| Layer2 Block2 (1st conv) | Conv(128->128,s=1)+BN+ReLU | (128, 16, 16) | -- |
| **Layer2 Block2 (last_conv)** | **Conv(128->128,s=1)+BN** | **(128, 16, 16)** | **self.last_conv** |
| AdaptiveAvgPool2d(1) | Global average pooling | (128,) | -- |
| FC | Linear(128->256)+BN1d+ReLU+Dropout(0.4) | (256,) | -- |
| Output | Linear(256->1) | (1,) logit | sigmoid at inference |
| **Total** | | | **716,865** |

---

## 4. Test Set Metrics (Real Measured Results on 106 Unseen Test Images)

Optimal decision threshold selected on validation set: **$\tau = 0.31$** (Validation F1 = 0.9369).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | F1 95% Bootstrap CI |
|-------|----------|-----------|--------|----------|---------|---------------------|
| Majority-class baseline (always pothole) | 0.5094 | -- | 1.0000 | 0.6750 | 0.5000 | -- |
| **Model A -- Vanilla CNN** | **0.9151** (91.5%) | 0.8814 | **0.9630** (96.3%) | **0.9204** | 0.9590 | **[0.8571, 0.9655]** |
| **Model B -- ResNet CNN** | **0.9151** (91.5%) | **0.9245** (92.5%) | 0.9074 (90.7%) | **0.9159** | **0.9669** | **[0.8496, 0.9663]** |

### Confusion Matrix Breakdown (Test Set, N = 106)
- **Model A (Vanilla CNN)**:
  - True Negatives (Normal detected as Normal): **45**
  - False Positives (Normal flagged as Pothole): **7**
  - False Negatives (Pothole missed): **2**
  - True Positives (Pothole detected): **52**
- **Model B (ResNet CNN)**:
  - True Negatives (Normal detected as Normal): **48**
  - False Positives (Normal flagged as Pothole): **4** (fewer false alarms!)
  - False Negatives (Pothole missed): **5**
  - True Positives (Pothole detected): **49**

---

## 5. Design Decision Rationale

| Decision | Reason |
|----------|--------|
| **Train from scratch** | Course constraint; teaches fundamental CNN concepts without pretrained features |
| **Vanilla CNN (Model A)** | Baseline demonstrating core CNN principles: local receptive fields, weight sharing, translation equivariance |
| **BatchNorm** | Normalises activations per mini-batch -> reduces internal covariate shift, enables higher LR, mild regulariser |
| **Residual connections** | Allow gradients to flow through skip paths -> mitigates vanishing gradients in deeper networks |
| **last_conv naming** | Required for Grad-CAM: hooks registered on this layer capture feature maps and gradients |
| **Augmentation (train only)** | Increases effective dataset size, improves generalisation; not applied to val/test to prevent leakage |
| **Class weights / pos_weight** | Dataset slightly imbalanced; BCEWithLogitsLoss(pos_weight=...) up-weights minority class gradients |
| **BCEWithLogitsLoss** | Numerically stable vs BCE(Sigmoid(logits)); accepts pos_weight for imbalance handling |
| **Threshold selection on val** | Default 0.5 is arbitrary; maximising F1 on val finds best precision-recall trade-off |
| **Early stopping (patience=7)** | Halts training when val loss stops improving -> prevents overfitting, saves compute |
| **70/15/15 stratified split** | Stratified = same class ratio per split; val for model selection; test seen only once |
| **Grad-CAM** | Class Activation Mapping via gradients; highlights spatial regions influencing prediction |
| **Kalman filter (1-D)** | Per-frame CNN probs are noisy; Kalman models prob as slowly drifting state for temporal consistency |
| **Bootstrap 95% CI** | With ~107 test samples, point estimates are unreliable; bootstrapping quantifies F1 uncertainty |
| **Adam optimiser** | Adaptive LR per parameter; converges faster than SGD for small datasets |
| **ReduceLROnPlateau** | Drops LR 0.5x if val loss plateaus for 3 epochs -- fine-tunes without manual scheduling |

---

## 6. Localisation Disclaimer

WARNING: Bounding boxes are weakly supervised. Derived by thresholding the Grad-CAM heat map -- a technique
from Explainable AI for CNNs (Zhou et al., 2016). No pixel-level or box-level annotations were used during
training. Therefore, box-level metrics such as mAP, IoU, and precision@IoU are NOT reported.

---

## 7. Ten Likely Examiner Questions & Answers

### Q1: Why do you use two models instead of one?
**A:** Comparison is the standard ML workflow. Model A (Vanilla CNN) demonstrates the core concept from the
syllabus; Model B (BatchNorm + residual) shows the effect of architectural advancement. The comparison reveals
whether added complexity actually helps generalisation -- a central question in model selection.

---

### Q2: What is Batch Normalisation and why does it help?
**A:** BatchNorm normalises each feature map's activations to zero mean and unit variance across the
mini-batch, then rescales with learnable gamma and beta. Benefits: (1) reduces internal covariate shift so
later layers train on more stable distributions; (2) allows higher learning rates; (3) mild regularising
effect. Formula: x_hat = (x - mean) / sqrt(var + eps), output = gamma * x_hat + beta.

---

### Q3: What is a residual connection and what problem does it solve?
**A:** A residual (skip) connection adds the block's input directly to its output: H(x) = F(x) + x. The
network learns the residual F(x) rather than the full mapping. Problem solved: vanishing gradients -- in
deep networks, gradients shrink as they propagate backward through many layers. The skip path provides a
gradient highway with magnitude 1, ensuring the signal reaches early layers.

---

### Q4: Explain Grad-CAM in simple terms.
**A:** Grad-CAM asks: "Which spatial regions of the last feature map most influenced the prediction?" It
computes the gradient of the class score w.r.t. each feature map channel (via backprop), globally-averages
those gradients to get channel importance weights, then computes a weighted sum of feature maps. A ReLU
keeps only positive contributions. The result is upsampled to input resolution and overlaid as a heat map.

---

### Q5: Why not use the default threshold of 0.5?
**A:** The sigmoid midpoint 0.5 is arbitrary. The optimal boundary depends on class imbalance and the
relative cost of FPs vs FNs. For pothole detection, missing a pothole (FN) may be worse than a false alarm.
Selecting the threshold that maximises F1 on the validation set finds the best precision-recall trade-off
before the test set is seen.

---

### Q6: What is the Kalman filter doing in the video pipeline?
**A:** The CNN processes each frame independently and produces noisy probability estimates that can flicker.
The 1-D Kalman filter treats the true pothole probability as a hidden state evolving slowly (process noise
Q=0.01). Each frame gives a noisy measurement (R=0.1). The filter combines the previous estimate with the
new measurement via Kalman gain K = P/(P+R), producing a smoothed, temporally consistent probability.

---

### Q7: What does the bootstrap confidence interval tell you?
**A:** With a small test set (~107 samples), our F1 estimate has high sampling variance. The bootstrap CI
resamples the test set with replacement 1000 times, computes F1 each time, and reports the 2.5th-97.5th
percentile range. This tells us: "if we repeated the experiment many times, the true F1 would lie in this
range 95% of the time."

---

### Q8: Why is the majority-class baseline important?
**A:** It sets the performance floor. If the model achieves 51% accuracy on a 51/49 dataset, it does no
better than always guessing the majority class. Reporting the baseline ensures the CNN has actually learned
the task rather than exploiting class imbalance.

---

### Q9: What are the main limitations of this system?
**A:**
1. Small dataset (~712 images) -> high variance, poor generalisation to unseen road conditions
2. Domain shift -> model trained on specific roads may fail on different countries/weather/lighting
3. Look-alikes -> shadows, manholes, water puddles, road markings share visual features with potholes
4. No pretrained features -> must learn low-level features from scratch; a larger dataset would use transfer learning
5. Weak localisation -> Grad-CAM highlights discriminative regions, not precise pothole boundaries
6. Small test set -> ~107 samples; confidence intervals are wide

---

### Q10: Why does the notebook use BCEWithLogitsLoss instead of BCELoss?
**A:** BCEWithLogitsLoss = log(sigmoid(x)) computed as a single fused operation using the log-sum-exp trick,
which is numerically stable even for large logit values. BCELoss(Sigmoid(x)) would first saturate the
sigmoid then take log -- losing gradient signal when logits are extreme. Additionally, BCEWithLogitsLoss
accepts a pos_weight parameter to handle class imbalance directly.

---

## 8. Key Hyperparameters Summary

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Image size | 224 x 224 | Standard CNN input; balance between detail and speed |
| Batch size | 32 | Fits in RAM; provides good gradient estimates |
| Learning rate | 1e-3 | Adam default; empirically good starting point |
| Dropout (Model A) | 0.5 | Strong regularisation for simpler model |
| Dropout (Model B) | 0.4 | Slightly less aggressive; BatchNorm already regularises |
| Max epochs | 50 | Generous budget; early stopping prevents over-training |
| Early stopping patience | 7 | ~15% of max epochs; tolerates brief plateaus |
| Seed | 42 | Reproducibility |
| Kalman Q | 0.01 | Slow state drift -> smooth output |
| Kalman R | 0.1 | Moderate trust in each frame measurement |
| Bootstrap iterations | 1,000 | Standard; stable CI estimate |
| Augmentation rotation | +/-15 deg | Realistic camera angle variation |
| Augmentation scale | 0.8-1.0 | Slight zoom; preserves pothole visibility |
| ColorJitter brightness/contrast | 0.3 | Accounts for lighting variation |

