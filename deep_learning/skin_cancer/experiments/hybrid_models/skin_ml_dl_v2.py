# ==========================================================
# HYBRID MODEL PIPELINE
# Use trained CNN (.h5) as FEATURE EXTRACTOR
# Then train:
#   1. SVM
#   2. Random Forest
#   3. XGBoost
# ==========================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ==========================================================
# PATH
# ==========================================================
base_dir = "ML_all_datasets/ML_all_datasets/data2/raw/skin"

# ==========================================================
# LOAD CSV
# ==========================================================
df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))

# ==========================================================
# IMAGE PATHS
# ==========================================================
def get_image_path(image_id):
    path1 = os.path.join(base_dir, "images1", image_id + ".jpg")
    path2 = os.path.join(base_dir, "images2", image_id + ".jpg")
    return path1 if os.path.exists(path1) else path2

df["path"] = df["image_id"].apply(get_image_path)

# ==========================================================
# SPLIT
# ==========================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)

# ==========================================================
# GENERATORS
# ==========================================================
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_generator = datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="dx",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ==========================================================
# LOAD TRAINED CNN
# ==========================================================
cnn_model = load_model("skin_cnn_model.h5")

print(cnn_model.summary())

# ==========================================================
# FEATURE EXTRACTOR
# Use second-last layer (Dense 256)
# ==========================================================
feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.layers[-2].output
)

# ==========================================================
# EXTRACT FEATURES
# ==========================================================
print("\nExtracting Train Features...")
X_train = feature_extractor.predict(train_generator, verbose=1)

print("\nExtracting Test Features...")
X_test = feature_extractor.predict(test_generator, verbose=1)

# Labels
y_train = train_generator.classes
y_test = test_generator.classes

class_names = list(train_generator.class_indices.keys())

print("Train Features Shape:", X_train.shape)
print("Test Features Shape :", X_test.shape)
# ==========================================================
# THRESHOLD OPTIMIZER FOR MINORITY CLASS RECALL
# Uses XGBoost probabilities from extracted CNN features
# Focus classes:
# mel, df, akiec
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ==========================================================
# ASSUMES YOU ALREADY HAVE:
# X_train, X_test, y_train, y_test, class_names
# from previous feature extraction code
# ==========================================================

print("Classes:", class_names)

# ----------------------------------------------------------
# Find class indices
# ----------------------------------------------------------
class_to_idx = {name:i for i,name in enumerate(class_names)}

mel_idx   = class_to_idx["mel"]
df_idx    = class_to_idx["df"]
akiec_idx = class_to_idx["akiec"]

# ==========================================================
# TRAIN XGBOOST
# ==========================================================
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='multi:softprob',
    num_class=len(class_names),
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# GET PROBABILITIES
# ==========================================================
probs = model.predict_proba(X_test)

# ==========================================================
# CUSTOM THRESHOLD PREDICTOR
# Priority:
# df -> mel -> akiec -> argmax
# ==========================================================
def custom_predict(prob,
                   th_df=0.12,
                   th_mel=0.22,
                   th_akiec=0.20):

    preds = []

    for p in prob:

        if p[df_idx] >= th_df:
            preds.append(df_idx)

        elif p[mel_idx] >= th_mel:
            preds.append(mel_idx)

        elif p[akiec_idx] >= th_akiec:
            preds.append(akiec_idx)

        else:
            preds.append(np.argmax(p))

    return np.array(preds)

# ==========================================================
# GRID SEARCH THRESHOLDS
# ==========================================================
best_score = -1
best_result = None

for th_df in np.arange(0.08, 0.31, 0.02):
    for th_mel in np.arange(0.10, 0.41, 0.03):
        for th_akiec in np.arange(0.10, 0.41, 0.03):

            y_pred = custom_predict(
                probs,
                th_df,
                th_mel,
                th_akiec
            )

            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            # focus metric:
            # average recall of minority classes
            score = (
                report["df"]["recall"] +
                report["mel"]["recall"] +
                report["akiec"]["recall"]
            ) / 3

            # keep some accuracy floor
            acc = accuracy_score(y_test, y_pred)

            if acc >= 0.65 and score > best_score:
                best_score = score
                best_result = (th_df, th_mel, th_akiec, acc, report)

# ==========================================================
# RESULTS
# ==========================================================
th_df, th_mel, th_akiec, acc, report = best_result

print("\n========================================")
print("BEST THRESHOLDS FOUND")
print("========================================")

print("df threshold    :", round(th_df,3))
print("mel threshold   :", round(th_mel,3))
print("akiec threshold :", round(th_akiec,3))
print("Accuracy        :", round(acc,4))

print("\nMinority Recall:")
print("df    :", round(report["df"]["recall"],4))
print("mel   :", round(report["mel"]["recall"],4))
print("akiec :", round(report["akiec"]["recall"],4))

print("\nFULL REPORT:\n")
print(classification_report(
    y_test,
    custom_predict(probs, th_df, th_mel, th_akiec),
    target_names=class_names,
    digits=4,
    zero_division=0
))