# ==========================================================
# CUSTOM CNN V3 FOR SKIN LESION CLASSIFICATION
# Focus:
# - No pretrained models
# - Deeper custom CNN
# - BatchNorm
# - GlobalAveragePooling
# - Focal Loss
# - Targeted Class Rebalancing
# - Better Recall for minority classes
# - Saves .keras model
# ==========================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

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
    p1 = os.path.join(base_dir, "images1", image_id + ".jpg")
    p2 = os.path.join(base_dir, "images2", image_id + ".jpg")
    return p1 if os.path.exists(p1) else p2

df["path"] = df["image_id"].apply(get_image_path)

print(df["dx"].value_counts())

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
# IMAGE GENERATORS
# ==========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.85,1.15]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(180,180),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="dx",
    target_size=(180,180),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ==========================================================
# CLASS INFO
# ==========================================================
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print("Classes:", class_names)

# ==========================================================
# TARGETED CLASS WEIGHTS
# ==========================================================
# Based on poor recall classes:
# df, mel, akiec boosted intentionally

base_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(base_weights))

# manual boosts
for k, v in train_generator.class_indices.items():
    if k == "df":
        class_weights[v] *= 2.5
    elif k == "mel":
        class_weights[v] *= 1.8
    elif k == "akiec":
        class_weights[v] *= 1.6

print("Class Weights:", class_weights)

# ==========================================================
# FOCAL LOSS
# ==========================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)

        focal = weight * cross_entropy
        return tf.reduce_sum(focal, axis=1)

    return loss

# ==========================================================
# CNN BLOCK
# ==========================================================
def conv_block(x, filters):

    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.20)(x)

    return x

# ==========================================================
# BUILD MODEL
# ==========================================================
inputs = layers.Input(shape=(180,180,3))

x = conv_block(inputs, 32)
x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.40)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ==========================================================
# COMPILE
# ==========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=categorical_focal_loss(gamma=2.0, alpha=0.25),
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================
callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),

    ModelCheckpoint(
        "best_skin_custom.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ==========================================================
# TRAIN
# ==========================================================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=35,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================================
# EVALUATE
# ==========================================================
loss, acc = model.evaluate(test_generator)

print("\nFinal Accuracy:", acc)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================
model.save("skin_custom_final.keras")

print("\nSaved:")
print("best_skin_custom.keras")
print("skin_custom_final.keras")