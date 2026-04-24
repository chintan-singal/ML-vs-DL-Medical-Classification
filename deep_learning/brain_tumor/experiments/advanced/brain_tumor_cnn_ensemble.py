import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

# ===== LOAD CNN MODEL =====
cnn_model = load_model("brain_tumor_cnn.h5")

# ===== FEATURE EXTRACTOR =====
feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.layers[-3].output   # Dense(256)
)

# ===== DATA GENERATORS =====
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ===== FEATURE EXTRACTION =====
def extract_features(generator, model):
    features, labels = [], []

    for i in range(len(generator)):
        x, y = generator[i]
        f = model.predict(x, verbose=0)

        features.append(f)
        labels.append(y)

    return np.vstack(features), np.vstack(labels)

X_train, y_train = extract_features(train_generator, feature_extractor)
X_val, y_val = extract_features(val_generator, feature_extractor)
X_test, y_test = extract_features(test_generator, feature_extractor)

# ===== LABELS =====
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)

# ===== SCALING =====
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ===== SVM =====
svm = SVC(kernel='rbf', C=10, probability=True)
svm.fit(X_train, y_train)

# ===== XGBOOST =====
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)

xgb.fit(X_train, y_train)

# ===== CNN PROBABILITIES =====
cnn_probs = cnn_model.predict(test_generator)

# ===== SVM PROBABILITIES =====
svm_probs = svm.predict_proba(X_test)

# ===== XGB PROBABILITIES =====
xgb_probs = xgb.predict_proba(X_test)

# ===== 🔥 WEIGHTED ENSEMBLE =====
final_probs = (
    0.3 * cnn_probs +
    0.4 * svm_probs +
    0.3 * xgb_probs
)

# ===== FINAL PREDICTIONS =====
y_pred = np.argmax(final_probs, axis=1)

# ===== RESULTS =====
print("\n🔥 Ensemble Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))