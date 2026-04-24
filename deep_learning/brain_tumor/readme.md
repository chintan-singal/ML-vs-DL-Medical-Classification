# Brain Tumor Classification using Deep Learning & Hybrid AI Models

## Project Overview

This project focuses on **multi-class brain tumor MRI image classification** using deep learning, transfer learning, hybrid machine learning pipelines, ensemble learning, and advanced preprocessing strategies.

The classification task predicts one of four classes:

* Glioma
* Meningioma
* No Tumor
* Pituitary

A wide range of experiments were conducted to systematically improve performance, starting from baseline CNN models and progressing toward transfer learning, feature extraction pipelines, ensemble systems, and advanced hard-mining strategies.

---

# Dataset Summary

The evaluation set contains **1600 MRI images**, equally distributed:

* 400 Glioma
* 400 Meningioma
* 400 No Tumor
* 400 Pituitary

This balanced evaluation setup allows fair comparison across models.

---

# Research Objective

The aim was not to train a single model, but to explore the complete model development lifecycle:

* Baseline CNN architectures
* CNN optimization
* Regularization techniques
* Transfer learning with pretrained networks
* CNN feature extraction + classical ML
* Ensemble methods
* MRI enhancement preprocessing
* Hard example mining
* Cross-framework experimentation (TensorFlow + PyTorch)

---

# Experiment Evolution Timeline

## Phase 1 — Baseline CNN Models

Initial custom CNN architectures were built from scratch to establish a performance baseline.

### Models

* Basic CNN
* CNN V3
* CNN Fine Tuned

### Purpose

To understand:

* how standard CNNs perform
* overfitting behavior
* class confusion trends
* effect of pooling / dropout / dense layers

---

## Phase 2 — Architecture Improvements

The baseline CNN was iteratively improved using:

* GlobalAveragePooling
* Dropout tuning
* Better dense layer sizing
* Batch normalization
* Optimizer tuning

These changes improved generalization significantly.

---

## Phase 3 — Transfer Learning

Modern pretrained CNN backbones were introduced:

* DenseNet121
* MobileNetV2

These models leveraged pretrained ImageNet features and were fine-tuned on MRI data.

---

## Phase 4 — Hybrid Deep Learning + Machine Learning

CNNs were used as feature extractors and paired with classical classifiers:

* CNN + SVM
* CNN + Random Forest
* CNN + XGBoost
* DenseNet + SVM

This allowed comparison between neural softmax heads vs classical decision boundaries.

---

## Phase 5 — Ensemble Learning

Predictions from multiple strong models were combined using weighted voting / stacking ideas.

---

## Phase 6 — Advanced Research Experiments

More sophisticated techniques were explored:

* CLAHE contrast enhancement
* Sobel edge enhancement
* MRI sharpening
* EfficientNetB3 backbone
* Hard example mining
* PyTorch feature extraction
* XGBoost second-stage classifier

---

# Final Benchmark Results

## Ranked by Accuracy

| Rank | Model                  | Accuracy |
| ---- | ---------------------- | -------- |
| 1    | PyTorch Hard Mined XGB | 94%      |
| 2    | CNN + Ensemble         | 90%      |
| 3    | CNN + RF               | 90%      |
| 4    | CNN + SVM              | 90%      |
| 5    | CNN + XGB              | 90%      |
| 6    | DenseNet + SVM         | 87%      |
| 7    | Basic CNN              | 83%      |
| 8    | MobileNet FT           | 81%      |
| 9    | EfficientNetB3 Enhance | 81%      |
| 10   | CNN V3                 | 62%      |
| 11   | CNN Class Tuned        | 56%      |
| 12   | DenseNet121            | 57%      |
| 13   | CNN Fine               | 51%      |

---

# Detailed Experiment Analysis

---

## 1. Basic CNN

A standard convolutional network trained from scratch.

### Performance

* Accuracy: **83%**
* Macro F1: **0.83**

### Class-wise Summary

* Strong pituitary classification
* No tumor detected reliably
* Moderate confusion between glioma and meningioma



---

## 2. CNN Fine Tuned

Improved baseline CNN with additional regularization and optimization.

### Performance

* Accuracy: **51%**

Model overfit certain classes and collapsed predictions toward dominant outputs.



---

## 3. CNN V3

Modified baseline with architecture refinements.

### Performance

* Accuracy: **62%**

Better than failed fine-tuned model, but still unstable on meningioma recall.



---

## 4. CNN + Ensemble

Combined multiple strong learners.

### Performance

* Accuracy: **90%**
* Macro F1: **0.90**

Very balanced across all classes.



---

## 5. CNN + Random Forest

CNN embeddings passed into Random Forest classifier.

### Performance

* Accuracy: **90%**

Strong overall performance and robust class separation.



---

## 6. CNN + SVM

CNN feature extraction followed by Support Vector Machine.

### Performance

* Accuracy: **90%**

Excellent boundary-based classification performance.



---

## 7. CNN + XGBoost

CNN embeddings classified using XGBoost.

### Performance

* Accuracy: **90%**

Balanced and highly competitive.



---

## 8. CNN Class Tuned

Used class weighting / imbalance-aware training.

### Performance

* Accuracy: **56%**

Useful experiment showing weighting alone cannot solve all representation issues.



---

## 9. DenseNet + SVM

DenseNet used as pretrained feature extractor with SVM head.

### Performance

* Accuracy: **87%**

One of the strongest transfer + classical ML combinations.



---

## 10. DenseNet121

Pure transfer learning classifier.

### Performance

* Accuracy: **81%**

Good baseline transfer model.



---

## 11. EfficientNetB3 Enhance

MRI enhancement preprocessing + EfficientNetB3.

### Performance

* Accuracy: **57%**

Interesting research experiment; preprocessing may require further tuning.



---

## 12. MobileNet Fine Tuned

Lightweight transfer learning model.

### Performance

* Accuracy: **81%**

Efficient and deployment-friendly.



---

## 13. PyTorch Hard Mined XGB

Most advanced pipeline.

### Components

* PyTorch feature extractor
* Hard example mining
* XGBoost second-stage classifier

### Performance

* Accuracy: **94%**
* Macro F1: **0.94**

Best model in the project.



---

# Key Insights

## Best Overall Accuracy

**PyTorch Hard Mined XGB (94%)**

## Best Classical Hybrid Models

* CNN + SVM
* CNN + RF
* CNN + XGB

(all ~90%)

## Best Pure Deep Learning Model

Basic CNN performed surprisingly strongly at 83%, showing good baseline architecture quality.

## Best Efficient Lightweight Model

MobileNet FT (81%)

---

# Lessons Learned

## 1. Hybrid Models Were Extremely Effective

CNN feature extractors paired with classical ML models consistently outperformed many standalone deep networks.

## 2. Hard Mining Improved Robustness

Focusing on difficult samples helped maximize final performance.

## 3. Transfer Learning Was Useful but Not Always Dominant

Pretrained models were strong but not always superior to tuned hybrid pipelines.

## 4. MRI-Specific Enhancement Needs Careful Calibration

Image enhancement pipelines are promising but sensitive.

---

# Folder Structure

```text
brain_tumor/
├── experiments/
├── data/
├── models/
├── results/
└── README.md
```

---

# Model Weights

Due to GitHub size limitations, trained models are hosted externally.

Please check the `models/` folder for Google Drive links.

---

# Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* scikit-learn
* XGBoost
* OpenCV
* NumPy
* Pandas
* Matplotlib

---

# Future Improvements

* Grad-CAM explainability
* Cross-validation benchmarking
* Web deployment
* Hyperparameter optimization
* Model compression for edge devices
* Clinical interpretability pipeline

---

# Final Conclusion

This project evolved from simple CNN baselines into a sophisticated medical imaging research pipeline involving deep learning, transfer learning, ensemble learning, feature engineering, and hard mining.

The final system achieved **94% classification accuracy** on a balanced 4-class MRI brain tumor benchmark, demonstrating the effectiveness of combining representation learning with advanced ensemble and boosting techniques.
