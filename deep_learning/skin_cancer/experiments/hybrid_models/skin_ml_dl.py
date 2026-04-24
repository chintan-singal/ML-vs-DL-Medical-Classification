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
# FUNCTION TO EVALUATE MODELS
# ==========================================================
def evaluate_model(name, model):
    
    print("\n===================================")
    print("MODEL:", name)
    print("===================================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(acc,4))
    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================================================
# 1. SVM
# ==========================================================
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

evaluate_model("SVM", svm_model)

# ==========================================================
# 2. RANDOM FOREST
# ==========================================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

evaluate_model("Random Forest", rf_model)

# ==========================================================
# 3. XGBOOST
# ==========================================================
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    random_state=42
)

evaluate_model("XGBoost", xgb_model)