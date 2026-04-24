import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ===== PATHS =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

# ===== LOAD TRAINED CNN =====
model = load_model("brain_tumor_cnn.h5")

# ===== FEATURE EXTRACTOR (Dense 256 layer) =====
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-3].output
)

# ===== DATA GENERATORS (NO AUGMENTATION, NO SHUFFLE) =====
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

# ===== FEATURE EXTRACTION FUNCTION =====
def extract_features(generator, model):
    features = []
    labels = []

    for i in range(len(generator)):
        print(f"Processing batch {i+1}/{len(generator)}")
        x, y = generator[i]
        f = model.predict(x, verbose=0)

        features.append(f)
        labels.append(y)

    return np.vstack(features), np.vstack(labels)

# ===== EXTRACT FEATURES =====
X_train, y_train = extract_features(train_generator, feature_extractor)
X_val, y_val = extract_features(val_generator, feature_extractor)
X_test, y_test = extract_features(test_generator, feature_extractor)

# ===== CONVERT LABELS =====
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)
y_test = np.argmax(y_test, axis=1)

print("\nShapes:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# ===== RANDOM FOREST MODEL =====
rf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ===== ACCURACY =====
print("\nValidation Accuracy:", rf.score(X_val, y_val))
print("Test Accuracy:", rf.score(X_test, y_test))

# ===== PREDICTIONS =====
y_pred = rf.predict(X_test)

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ===== CLASS NAMES =====
class_names = list(test_generator.class_indices.keys())

# ===== CLASSIFICATION REPORT =====
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))