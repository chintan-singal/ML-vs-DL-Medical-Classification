import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")  # 🔥 THIS was missing

# ===== DATA GENERATORS (SAME AS TRAINING) =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 🔥 IMPORTANT: shuffle=False for feature extraction
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

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False   # 🔥 VERY IMPORTANT
)

# ===== LOAD TRAINED MODEL =====
model = load_model("brain_tumor_cnn.h5")

# ===== CREATE FEATURE EXTRACTOR =====
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-3].output   # Dense(256)
)

# ===== FEATURE EXTRACTION FUNCTION =====
def extract_features(generator, model):
    features = []
    labels = []

    for i in range(len(generator)):
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
# ===== DEBUG OUTPUT =====
print("Train features:", X_train.shape)
print("Train labels:", y_train.shape)
print("Val features:", X_val.shape)
print("Val labels:", y_val.shape)
print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10, class_weight='balanced')

svm.fit(X_train, y_train)

val_acc = svm.score(X_val, y_val)
test_acc = svm.score(X_test, y_test)

print("SVM Test Accuracy:", test_acc)

print("SVM Validation Accuracy:", val_acc)


from sklearn.metrics import confusion_matrix, classification_report

y_pred = svm.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

