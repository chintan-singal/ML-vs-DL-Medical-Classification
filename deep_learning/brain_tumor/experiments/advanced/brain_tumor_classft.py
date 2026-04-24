import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

# ===== DATA (STABLE AUGMENTATION) =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160,160),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160,160),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(160,160),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ===== CLASS WEIGHTS (CONTROLLED) =====
labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

# Mild boosting (not aggressive)
class_weights[0] *= 1.3   # glioma
class_weights[1] *= 1.2   # meningioma

print("Class Weights:", class_weights)

# ===== MODEL =====
model = models.Sequential([

    layers.Conv2D(32, (3,3), padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(160,160,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), padding='same', activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

# ===== COMPILE (FIXED) =====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== CALLBACKS =====
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ===== TRAIN =====
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ===== SAVE =====
model.save("cnn_class_tuned_fixed.keras")

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)
print("\n🔥 Test Accuracy:", test_acc)

# ===== EVALUATION =====
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = list(test_generator.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nPer-Class Accuracy:")
for i, name in enumerate(class_names):
    total = np.sum(cm[i])
    correct = cm[i][i]
    print(f"{name}: {correct/total:.4f}")