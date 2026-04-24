# ML vs DL Medical Classification

## Project Overview

This repository presents a collaborative research project comparing **Machine Learning (ML)** and **Deep Learning (DL)** approaches for medical image classification across multiple healthcare datasets.

The goal is to systematically evaluate how traditional ML pipelines compare against modern deep learning architectures on real-world diagnostic imaging problems.

This project combines contributions from three different domains:

* **Data Engineering** – dataset preparation, cleaning, preprocessing pipelines
* **Machine Learning** – classical ML models and feature-based methods
* **Deep Learning** – CNNs, transfer learning, hybrid AI systems, advanced training strategies

---

# Problem Domains Covered

This repository focuses on three medical imaging tasks:

## 1. Brain Tumor MRI Classification

Multi-class MRI image classification:

* Glioma
* Meningioma
* Pituitary
* No Tumor

## 2. Chest X-Ray Pneumonia Detection

Binary chest radiograph classification:

* NORMAL
* PNEUMONIA

## 3. Skin Cancer Lesion Classification

Multi-class dermoscopic lesion recognition:

* akiec
* bcc
* bkl
* df
* mel
* nv
* vasc

---

# Repository Structure

```text id="jlwm247"
ML-vs-DL-Medical-Classification/
├── README.md
├── deep_learning/
│   ├── brain_tumor/
│   ├── chest_xray/
│   └── skin_cancer/
├── machine_learning/
│   └── (to be added)
└── data_engineering/
    └── (to be added)
```

---

# Team Contributions

## Deep Learning Module

Completed module containing:

* Baseline CNN architectures
* Fine-tuned deep learning models
* Transfer learning pipelines
* Hybrid CNN + ML models
* Hard mining systems
* Detailed experiment reports
* Confusion matrices
* Final leaderboards

Datasets completed:

* Brain Tumor MRI
* Chest X-Ray
* Skin Cancer

## Machine Learning Module *(To Be Added)*

Suggested future contents:

* Logistic Regression
* Random Forest
* SVM
* XGBoost
* PCA pipelines
* Handcrafted feature extraction
* Comparative benchmarking vs DL

## Data Engineering Module *(To Be Added)*

Suggested future contents:

* Data cleaning pipelines
* EDA reports
* Missing value handling
* Augmentation strategies
* Label consistency checks
* Train/validation/test split design
* Metadata management

---

# Deep Learning Results Summary

| Dataset         | Best Model             | Accuracy   |
| --------------- | ---------------------- | ---------- |
| Brain Tumor MRI | PyTorch Hard Mined XGB | **94%**    |
| Chest X-Ray     | TensorFlow CNN         | **86.06%** |
| Skin Cancer     | PyTorch Hard Mined XGB | **86%**    |

---

# Key Research Themes

This project explores:

* ML vs DL performance tradeoffs
* Class imbalance handling
* Medical image augmentation
* Transfer learning effectiveness
* Hard example mining
* Ensemble systems
* Explainability opportunities
* Robustness in healthcare AI

---

# Technologies Used

## Languages & Frameworks

* Python
* TensorFlow / Keras
* PyTorch
* scikit-learn
* XGBoost

## Data & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn
* OpenCV

---

# How to Navigate

## Deep Learning Reports

Open:

```text id="jlwm248"
deep_learning/brain_tumor/README.md
deep_learning/chest_xray/README.md
deep_learning/skin_cancer/README.md
```

for complete experiment histories, model evolution, and detailed metrics.

## Upcoming Sections

Machine Learning and Data Engineering contributors can populate their folders independently.

---

# Future Roadmap

* Add ML benchmark suite across all datasets
* Add preprocessing & feature engineering reports
* Add Grad-CAM explainability dashboards
* Add Streamlit / Flask deployment demos
* Add cross-validation benchmark tables
* Add final comparative research report

---

# Why This Project Matters

Medical AI systems must balance:

* Accuracy
* Interpretability
* Recall for critical diseases
* Robustness
* Scalability

This repository demonstrates practical experimentation across multiple datasets and modeling paradigms.

---

# Authors

## Deep Learning Module

**Chintan Kumar Singal**

## Machine Learning Module

*Aditya Raj*

## Data Engineering Module

*Harsh Mittal*

---

# Final Note

This project is designed not just as a code repository, but as a structured comparison of modern AI methods in medical imaging.

It reflects experimentation, iteration, engineering discipline, and real-world model evaluation.
