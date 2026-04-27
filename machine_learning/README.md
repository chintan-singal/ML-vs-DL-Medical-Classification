# Medical Image Classification using Classical Machine Learning

## Project Overview

This module benchmarks classical Machine Learning pipelines on three medical imaging datasets as part of the ML vs DL Medical Classification project.

Five handcrafted feature extraction strategies are evaluated across three classifiers on each dataset, establishing rigorous ML baselines for direct comparison against deep learning approaches.

---

## Module Structure

```
machine_learning/
├── README.md                         ← You are here
├── brain_tumor/
│   ├── README.md
│   ├── code/
│   │   └── ml_pipeline.py
│   └── results/
│       ├── results.txt
│       └── figure_01.png ... figure_15.png
├── chest_xray/
│   ├── README.md
│   ├── code/
│   │   └── ml_pipeline.py
│   └── results/
│       ├── results.txt
│       └── figure_01.png ... figure_17.png
└── skin_cancer/
    ├── README.md
    ├── code/
    │   └── ml_pipeline.py
    └── results/
        ├── results.txt
        └── figure_01.png ... figure_15.png
```

---

## Datasets

| Dataset | Task | Classes | Train Size | Image Format |
|---|---|---|---|---|
| Brain Tumor MRI | Multi-class (4) | Glioma, Meningioma, Pituitary, No Tumor | 4,480 | Grayscale 224×224 |
| Chest X-Ray | Binary | Normal, Pneumonia | 5,216 | Grayscale 224×224 |
| Skin Cancer (HAM10000) | Multi-class (7) | nv, mel, bkl, bcc, akiec, vasc, df | 10,015 | RGB 64×64 |

---

## Feature Extraction Pipeline

All notebooks follow the same extraction protocol:

| Feature Set | Description |
|---|---|
| Flat Pixels (Baseline) | Raw flattened pixel values — no feature engineering |
| HOG | Histogram of Oriented Gradients — captures edge and shape structure |
| LBP | Local Binary Patterns — captures local micro-texture |
| Texture | Gabor filters + statistical texture descriptors |
| HOG + LBP + Texture | Combined handcrafted feature vector |

All pipelines use `StandardScaler` normalization inside a `sklearn.Pipeline`.

---

## Classifiers

- Logistic Regression
- SVM (RBF kernel)
- Random Forest

---

## Module Leaderboard — Best ML Result Per Dataset

| Dataset | Best Model | Feature Set | Test Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| Brain Tumor MRI | Logistic Regression | HOG + LBP + Texture | **87.12%** | 0.8682 | 0.9649 |
| Chest X-Ray | Logistic Regression | HOG + LBP + Texture | **80.29%** | 0.8601 | 0.8949 |
| Skin Cancer | SVM (RBF) | Texture | **74.39%** | 0.4227 | 0.9006 |

---

## Cross-Dataset Findings

**Brain Tumor MRI** is the strongest domain for classical ML — HOG features capture MRI structural boundaries naturally, pushing accuracy to 87% without any learned representations.

**Chest X-Ray** achieves 80% accuracy with exceptionally high recall (>96%), making the pipeline clinically relevant for pneumonia screening even before deep learning enters the picture.

**Skin Cancer** is the hardest task — severe class imbalance and color-based diagnostics limit classical ML. Texture features dominate while HOG, which works well on MRI, completely fails here, confirming that feature selection is domain-specific in medical imaging.

---

## Technologies Used

- Python
- scikit-learn
- scikit-image (HOG, LBP, Gabor)
- NumPy / Pandas
- Matplotlib / Seaborn

---

## Author

**Aditya Raj** — Machine Learning Module

Part of the [ML vs DL Medical Classification](../README.md) collaborative research project.
