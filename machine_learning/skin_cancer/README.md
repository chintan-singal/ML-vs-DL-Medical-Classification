# Skin Cancer Lesion Classification using Classical Machine Learning

## Project Overview

This project applies classical machine learning pipelines to multi-class dermoscopic skin lesion classification using the HAM10000 dataset.

Five handcrafted feature extraction strategies were evaluated across three classifiers on a severely class-imbalanced 7-class problem — one of the most challenging classification tasks in this project.

---

## Classification Task

The model predicts one of seven dermoscopic lesion classes:

| Label | Description |
|---|---|
| nv | Melanocytic nevi (benign moles) |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |

**Dataset split:**

- Total: 10,015 images
- Train: 7,010 | Val: 1,002 | Test: 2,003
- Image format: RGB, 64×64

**Class distribution (heavily imbalanced):**

| Class | Count | % |
|---|---|---|
| nv | 6,705 | 66.9% |
| mel | 1,113 | 11.1% |
| bkl | 1,099 | 11.0% |
| bcc | 514 | 5.1% |
| akiec | 327 | 3.3% |
| vasc | 142 | 1.4% |
| df | 115 | 1.1% |

> ⚠️ Severe class imbalance — `nv` accounts for ~67% of all samples. Accuracy is inflated by this dominant class. F1 and AUC are the more meaningful metrics here.

---

## Research Objectives

This module aimed to answer:

- Can classical ML distinguish malignant from benign lesions from dermoscopic images?
- Which handcrafted feature best captures color and texture variation in skin lesions?
- How much does class imbalance suppress classical ML performance?
- What is the ML baseline for this difficult 7-class problem?

---

## Pipeline Architecture

All models follow the same structure:

```
Raw Dermoscopic Image (64×64 RGB)
        ↓
Per-channel Feature Extraction (HOG / LBP / Texture / Combined)
        ↓
StandardScaler normalization
        ↓
Classifier (LR / SVM / Random Forest)
        ↓
7-Class Prediction
```

> Note: Features are extracted per RGB channel and concatenated, capturing color-dependent texture information.

---

## Feature Sets

| Feature Set | Dimensions | Description |
|---|---|---|
| Flat Pixels (Baseline) | 12,288 | Raw flattened RGB pixel values (64×64×3) |
| HOG | 5,292 | Histogram of Oriented Gradients |
| LBP | 768 | Local Binary Patterns (per channel) |
| Texture | 21 | Gabor filter responses + statistical texture (per channel) |
| HOG + LBP + Texture | 6,081 | Combined handcrafted feature vector |

---

## Experiment Results — Test Set

### Logistic Regression

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.6595 | 0.4179 | 0.3547 | 0.3788 | 0.8087 |
| HOG | 0.6006 | 0.2446 | 0.2334 | 0.2365 | 0.7348 |
| LBP | 0.6790 | 0.2584 | 0.1956 | 0.2008 | 0.7593 |
| **Texture** | **0.7259** | **0.4966** | **0.3894** | **0.4215** | **0.8859** |
| HOG + LBP + Texture | 0.6485 | 0.3317 | 0.2905 | 0.3014 | 0.7938 |

### SVM (RBF)

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7204 | 0.3959 | 0.2617 | 0.2850 | 0.8907 |
| HOG | 0.6950 | 0.3222 | 0.1850 | 0.1842 | 0.8207 |
| LBP | 0.6890 | 0.2057 | 0.1836 | 0.1802 | 0.7862 |
| **Texture** | **0.7439** | **0.5487** | **0.3735** | **0.4227** | **0.9006** |
| HOG + LBP + Texture | 0.7049 | 0.2989 | 0.1984 | 0.2018 | 0.8459 |

### Random Forest

| Feature Set | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|
| Flat Pixels (Baseline) | 0.7199 | 0.3639 | 0.2633 | 0.2792 | 0.8835 |
| HOG | 0.6745 | 0.3150 | 0.1493 | 0.1273 | 0.7425 |
| LBP | 0.6875 | 0.2680 | 0.1857 | 0.1878 | 0.7826 |
| **Texture** | **0.7379** | **0.5413** | **0.3328** | **0.3798** | **0.8899** |
| HOG + LBP + Texture | 0.6790 | 0.2458 | 0.1562 | 0.1402 | 0.8063 |

---

## Final Leaderboard — Best per Classifier (Test Set)

| Rank | Classifier | Feature Set | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| 🥇 1 | SVM (RBF) | Texture | **74.39%** | 0.4227 | **0.9006** |
| 🥈 2 | Random Forest | Texture | 73.79% | 0.3798 | 0.8899 |
| 🥉 3 | Logistic Regression | Texture | 72.59% | **0.4215** | 0.8859 |

> **Best overall:** SVM (RBF) + Texture features — **74.39% test accuracy, AUC 0.9006**

---

## Detailed Analysis

### 1. Texture Features Dominate Across All Classifiers

This is the clearest finding in this dataset — Texture features outperform every other feature set for every single classifier. Dermoscopic images carry far more diagnostic information in color texture and statistical patterns than in edges or shapes. HOG, which excelled on brain MRI, performs worst here.

### 2. HOG is the Worst Feature Set

HOG consistently ranks last or near last on skin cancer. Dermoscopic lesions are defined by color variation, pigmentation patterns, and subtle surface texture — not hard edges or structural gradients. This directly explains why the feature that works best on MRI fails here.

### 3. Class Imbalance Severely Suppresses F1

Despite accuracy in the 70s, macro F1 scores sit at 0.38–0.42. The `nv` class (67% of data) dominates predictions. Rare but clinically critical classes like `vasc` (1.4%) and `df` (1.1%) are largely missed by all models.

### 4. AUC Stays Strong Despite F1 Drop

AUC reaches 0.90 for SVM + Texture — indicating strong discriminative ability at the class-pair level even when overall multi-class F1 is suppressed by imbalance. The models can separate classes when forced to, but default probability thresholds favor the majority class.

### 5. Combined Features Consistently Hurt

HOG + LBP + Texture underperforms Texture-alone for all three classifiers. On this dataset, the noise introduced by HOG degrades the strong Texture signal rather than complementing it.

---

## Key Clinical Insight

Skin cancer detection from dermoscopy is particularly challenging for classical ML due to class imbalance and the subtle color-based nature of lesion differentiation. The **0.9006 AUC from SVM + Texture** is actually a meaningful result — suggesting the pipeline can rank samples correctly even when absolute predictions are skewed by imbalance. Addressing imbalance (SMOTE, class weighting) is the most important next step for this domain.

---

## Folder Structure

```
skin_cancer/
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
- KaggleHub (HAM10000 dataset)

---

## Future Improvements

- SMOTE / class-weighted training to address severe imbalance
- Per-class ROC curve analysis
- Color histogram features (dermoscopy is color-rich)
- Augmentation-based oversampling for minority classes
- Macro F1 as primary optimization metric

---

## Conclusion

The classical ML pipeline achieved **74.39% test accuracy** on the 7-class HAM10000 skin cancer dataset. While accuracy appears competitive, macro F1 of 0.42 reveals that the pipeline struggles with minority classes. Texture features proved uniquely effective for dermoscopic images — consistently outperforming HOG, LBP, and combined vectors across all classifiers.

This establishes the ML baseline for skin lesion classification and highlights the domain-specific importance of feature selection in medical imaging.

---

## Author

**Aditya Raj** — Machine Learning Module

Part of the [ML vs DL Medical Classification](../../README.md) collaborative research project.
