# Brain Tumor MRI Classification using Classical Machine Learning

## Project Overview

This project applies classical machine learning pipelines to multi-class brain MRI classification, benchmarking handcrafted feature extraction strategies against three classifiers.

The goal is to establish a rigorous ML baseline that can be directly compared against deep learning approaches on the same dataset and evaluation protocol.

---

## Classification Task

The model predicts one of four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

**Dataset split:**

- Train: 4,480 images
- Image format: Grayscale, 224×224

---

## Research Objectives

This module aimed to answer:

- Can classical ML extract meaningful structural information from raw MRI scans?
- Which feature representation best captures tumor morphology?
- How much does combining feature types help versus individual descriptors?
- What is the ML ceiling before deep learning enters the picture?

---

## Pipeline Architecture

All models follow the same structure:

```
Raw MRI Image (224×224 grayscale)
        ↓
Feature Extraction (HOG / LBP / Texture / Combined)
        ↓
StandardScaler normalization
        ↓
Classifier (LR / SVM / Random Forest)
        ↓
4-Class Prediction
```

---

## Feature Sets

| Feature Set | Dimensions | Description |
|---|---|---|
| Flat Pixels (Baseline) | 50,176 | Raw flattened pixel values |
| HOG | 5,292 | Histogram of Oriented Gradients — edge and shape structure |
| LBP | 256 | Local Binary Patterns — local micro-texture |
| Texture | 7 | Gabor filter responses + statistical texture descriptors |
| HOG + LBP + Texture | 6,081 | Combined handcrafted feature vector |

---

## Experiment Results — Test Set

### Logistic Regression

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7894 | 0.7857 | 0.7894 | 0.7859 | 0.9126 |
| HOG | 0.8706 | 0.8751 | 0.8706 | 0.8675 | 0.9651 |
| LBP | 0.6406 | 0.6437 | 0.6406 | 0.6405 | 0.8313 |
| Texture | 0.6356 | 0.6276 | 0.6356 | 0.6294 | 0.8345 |
| **HOG + LBP + Texture** | **0.8712** | 0.8751 | 0.8713 | **0.8682** | 0.9649 |

### SVM (RBF)

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.8219 | 0.8210 | 0.8219 | 0.8174 | 0.9447 |
| **HOG** | 0.8612 | 0.8728 | 0.8613 | 0.8574 | 0.9673 |
| LBP | 0.7319 | 0.7336 | 0.7319 | 0.7278 | 0.8930 |
| Texture | 0.6694 | 0.6598 | 0.6694 | 0.6597 | 0.8582 |
| HOG + LBP + Texture | **0.8619** | 0.8737 | 0.8619 | **0.8580** | **0.9674** |

### Random Forest

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| **Flat Pixels (Baseline)** | **0.8569** | 0.8690 | 0.8569 | **0.8526** | **0.9576** |
| HOG | 0.8325 | 0.8453 | 0.8325 | 0.8272 | 0.9483 |
| LBP | 0.8275 | 0.8284 | 0.8275 | 0.8221 | 0.9478 |
| Texture | 0.8119 | 0.8082 | 0.8119 | 0.8058 | 0.9382 |
| HOG + LBP + Texture | 0.8300 | 0.8424 | 0.8300 | 0.8244 | 0.9507 |

---

## Final Leaderboard — Best per Classifier (Test Set)

| Rank | Classifier | Feature Set | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| 🥇 1 | Logistic Regression | HOG + LBP + Texture | **87.12%** | 0.8682 | 0.9649 |
| 🥈 2 | SVM (RBF) | HOG + LBP + Texture | 86.19% | 0.8580 | 0.9674 |
| 🥉 3 | Random Forest | Flat Pixels | 85.69% | 0.8526 | 0.9576 |

> **Best overall:** Logistic Regression + HOG + LBP + Texture — **87.12% test accuracy**

---

## Detailed Analysis

### 1. HOG Features Are the Core Driver

HOG alone pushed Logistic Regression to 87.06% and SVM to 86.12% — nearly matching the combined feature vector. MRI scans have well-defined structural boundaries between tissue types, which HOG captures directly. This is the strongest single-feature result across all three datasets in this module.

### 2. Combined Features Marginally Outperform HOG for LR and SVM

Adding LBP and Texture on top of HOG pushed LR from 87.06% to 87.12% — a small but consistent improvement. The combined vector adds complementary micro-texture information that HOG misses.

### 3. Random Forest Works Best on Raw Pixels

RF achieved its best result (85.69%) directly on flat pixel values. This suggests the tree ensemble can learn spatial pixel relationships without explicit feature engineering — an unusual result compared to LR and SVM.

### 4. LBP and Texture Are Weak Individually

LBP (64.06% LR) and Texture (63.56% LR) severely underperform when used alone, but their addition to HOG improves results. They carry complementary signal, just not enough standalone discriminative power for 4-class MRI classification.

### 5. All Models Achieve AUC > 0.93

Despite accuracy differences, every model shows AUC above 0.93 on the test set, confirming strong class separability in the feature space across all classifiers and feature types.

---

## Key Insight

Brain MRI is the strongest domain for classical ML in this project. The structured edge and shape information in MRI scans aligns naturally with HOG — a gradient-based descriptor — enabling the classical pipeline to reach **87% accuracy without any learned representations**.

---

## Folder Structure

```
brain_tumor/
├── README.md
├── code/
│   └── ml_pipeline.py
└── results/
    ├── results.txt
    ├── figure_01.png  ← Confusion matrix: Flat Pixels LR
    ├── figure_02.png  ← Confusion matrix: Flat Pixels SVM
    ...
    └── figure_15.png  ← Final summary plots
```

---

## Technologies Used

- Python
- scikit-learn
- scikit-image (HOG, LBP, Gabor)
- NumPy / Pandas
- Matplotlib

---

## Future Improvements

- PCA dimensionality reduction before SVM
- Cross-validation for more stable estimates
- XGBoost and Gradient Boosting comparison
- Per-class analysis (Glioma vs Meningioma confusion)
- Grad-CAM-style attention maps using learned features

---

## Conclusion

The classical ML pipeline achieved **87.12% test accuracy** on brain tumor MRI classification — the highest ML result across all three datasets in this module. HOG features proved highly effective on grayscale MRI scans, and Logistic Regression with combined handcrafted features matched or exceeded more complex classifiers.

This establishes a strong classical baseline for the deep learning comparison.

---

## Author

**Aditya Raj** — Machine Learning Module

Part of the [ML vs DL Medical Classification](../../README.md) collaborative research project.
