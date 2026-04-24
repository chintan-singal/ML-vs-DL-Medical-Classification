import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os

# ===== PATH =====
base_dir = r"C:\Users\chint\OneDrive\Documents\ML Project\ML_all_datasets\ML_all_datasets\data1\raw\brain_tumor"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

# ===== EDGE PREPROCESSING (FINAL FIXED VERSION) =====
def add_sobel_edges(img):
    img = (img * 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(255 * sobel / (np.max(sobel) + 1e-8))

    # convert to 3-channel
    sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    # overlay edges on image
    img = cv2.addWeighted(img, 0.8, sobel, 0.2, 0)

    return img / 255.0

# ===== DATA =====
train_datagen = ImageDataGenerator(
    preprocessing_function=add_sobel_edges,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.25,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=add_sobel_edges
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ===== MODEL =====
model = models.Sequential([

    layers.Input(shape=(150,150,3)),  # 🔥 back to 3 channels

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    # Block 4
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ===== TRAIN =====
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_lr=1e-6
        )
    ]
)

# ===== SAVE =====
model.save("brain_tumor_edge_model.keras")

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)
print("\n🔥 Test Accuracy:", test_acc)