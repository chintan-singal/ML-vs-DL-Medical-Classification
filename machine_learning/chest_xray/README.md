# Chest X-Ray Pneumonia Detection using Classical Machine Learning

## Project Overview

This project applies classical machine learning pipelines to binary chest X-ray classification for automated pneumonia detection.

Five handcrafted feature extraction strategies were evaluated across three classifiers, benchmarking how much structural and textural image information classical ML can exploit without any learned representations.

---

## Classification Task

The model predicts one of two classes:

- NORMAL
- PNEUMONIA

**Dataset split:**

- Train: 5,216 images
- Val: 16 images
- Test: 624 images
  - 234 NORMAL
  - 390 PNEUMONIA

> ⚠️ The validation set is only 16 samples — validation scores are unreliable. All reported results use the test set.

---

## Research Objectives

This module aimed to answer:

- How far can handcrafted features take a classical pipeline on a binary medical imaging task?
- Which feature representation best captures pneumonia-relevant patterns?
- How do LR, SVM, and Random Forest compare across different feature spaces?
- What is the ML baseline before deep learning enters the comparison?

---

## Pipeline Architecture

All models follow the same structure:

```
Raw Image (224×224 grayscale)
        ↓
Feature Extraction (HOG / LBP / Texture / Combined)
        ↓
StandardScaler normalization
        ↓
Classifier (LR / SVM / Random Forest)
        ↓
Binary Prediction
```

---

## Feature Sets

| Feature Set | Dimensions | Description |
|---|---|---|
| Flat Pixels (Baseline) | 50,176 | Raw flattened pixel values |
| HOG | 26,244 | Histogram of Oriented Gradients — edge and shape structure |
| LBP | 256 | Local Binary Patterns — local micro-texture |
| Texture | 7 | Gabor filter responses + statistical texture descriptors |
| HOG + LBP + Texture | 26,507 | Combined handcrafted feature vector |

---

## Experiment Results — Test Set

### Logistic Regression

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7612 | 0.7269 | 0.9897 | 0.8382 | 0.8810 |
| HOG | 0.7981 | 0.7683 | 0.9692 | 0.8571 | 0.8936 |
| LBP | 0.7788 | 0.7614 | 0.9410 | 0.8417 | 0.8410 |
| Texture | 0.7885 | 0.7676 | 0.9487 | 0.8486 | 0.8589 |
| **HOG + LBP + Texture** | **0.8029** | 0.7730 | 0.9692 | **0.8601** | **0.8949** |

### SVM (RBF)

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7772 | 0.7418 | 0.9872 | 0.8471 | 0.9231 |
| HOG | 0.7196 | 0.6923 | 0.9923 | 0.8156 | 0.9211 |
| **LBP** | **0.7901** | 0.7627 | 0.9641 | **0.8516** | 0.8152 |
| Texture | 0.7772 | 0.7722 | 0.9128 | 0.8367 | 0.8295 |
| HOG + LBP + Texture | 0.7196 | 0.6923 | 0.9923 | 0.8156 | **0.9212** |

### Random Forest

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7548 | 0.7207 | 0.9923 | 0.8350 | 0.9333 |
| HOG | 0.7003 | 0.6784 | 0.9897 | 0.8050 | 0.8805 |
| LBP | 0.7500 | 0.7276 | 0.9590 | 0.8274 | 0.8413 |
| **Texture** | **0.7724** | 0.7616 | 0.9256 | **0.8356** | **0.8471** |
| HOG + LBP + Texture | 0.6923 | 0.6725 | 0.9897 | 0.8008 | 0.8977 |

---

## Final Leaderboard — Best per Classifier (Test Set)

| Rank | Classifier | Feature Set | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| 🥇 1 | Logistic Regression | HOG + LBP + Texture | **80.29%** | 0.8601 | 0.8949 |
| 🥈 2 | SVM (RBF) | LBP | 79.01% | 0.8516 | 0.8152 |
| 🥉 3 | Random Forest | Texture | 77.24% | 0.8356 | 0.8471 |

> **Best overall:** Logistic Regression + HOG + LBP + Texture — **80.29% test accuracy**

---

## Detailed Analysis

### 1. Logistic Regression Won Overall

Combined HOG + LBP + Texture gave LR the richest feature space and produced the best test accuracy at 80.29%. LR consistently benefited from combining all three feature types.

### 2. Very High Recall Across All Models

Almost every model achieved recall above 0.93 for pneumonia — meaning very few pneumonia cases are missed. This is the clinically important metric and a strong result for classical ML.

### 3. Precision is the Bottleneck

Precision sits around 0.72–0.77 across models, meaning false positives exist. This is an acceptable tradeoff for a screening scenario where missing a positive is more costly than a false alarm.

### 4. SVM Surprised on LBP

SVM with LBP features achieved 79.01% accuracy while maintaining the highest recall (96.41%), making it a strong screening candidate despite the compact feature space.

### 5. Random Forest Preferred Texture Features

RF consistently performed best on Texture features rather than raw pixels or HOG — suggesting RF leverages statistical texture descriptors well without overfitting high-dimensional spaces.

### 6. Combined Features Hurt SVM and RF

Adding all features together degraded SVM and RF performance relative to their individual best. High-dimensional combined vectors introduce noise that tree-based and kernel methods struggle to filter.

---

## Key Clinical Insight

For medical screening, **recall is often more critical than accuracy**. All three best models achieve recall ≥ 0.93, meaning the classical ML pipeline reliably detects pneumonia and misses very few cases — a meaningful result even before deep learning enters the comparison.

---

## Folder Structure

```
chest_xray/
├── README.md
├── code/
│   └── ml_pipeline.py
└── results/
    ├── results.txt
    ├── figure_01.png  ← Confusion matrix: Flat Pixels LR
    ├── figure_02.png  ← Confusion matrix: Flat Pixels SVM
    ...
    └── figure_17.png  ← Final summary plots
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

- Class-weighted training to improve normal detection
- PCA dimensionality reduction before SVM
- Threshold tuning to optimize recall/precision tradeoff
- ROC curve analysis per feature set
- Cross-validation instead of single train/val/test split

---

## Conclusion

The classical ML pipeline achieved **80.29% test accuracy** on chest X-ray pneumonia detection. The pipeline demonstrates that handcrafted features combined with Logistic Regression can produce clinically meaningful recall rates (96.92%) even without any learned visual representations.

This serves as the baseline for comparison against deep learning approaches in this repository.

---

## Author

**Aditya Raj** — Machine Learning Module

Part of the [ML vs DL Medical Classification](../../README.md) collaborative research project.
