import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data2/raw/skin"

# ===== LOAD CSV =====
df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))

# ===== CREATE IMAGE PATH =====
def get_image_path(image_id):
    path1 = os.path.join(base_dir, "images1", image_id + ".jpg")
    path2 = os.path.join(base_dir, "images2", image_id + ".jpg")
    return path1 if os.path.exists(path1) else path2

df["path"] = df["image_id"].apply(get_image_path)

# ===== TRAIN-TEST SPLIT =====
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)

# ===== CALCULATE CLASS WEIGHTS =====
# This forces the model to care more about rare classes like 'df' and 'mel'
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
# Map weights to their corresponding generator class indices
class_weights_dict = dict(enumerate(class_weights_array))
print("\nClass Weights:", class_weights_dict)

# ===== ADVANCED DATA GENERATORS =====
# Added rotation, shifts, flips, and zoom for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Test generator ONLY rescales. Never augment test data!
test_datagen = ImageDataGenerator(rescale=1./255)

# BUMPED RESOLUTION TO 224x224
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ===== SMART DEEPER MODEL =====
num_classes = df["dx"].nunique()

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), input_shape=(224, 224, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),
    
    # Block 4 (New Depth)
    layers.Conv2D(256, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    # Block 5 (New Depth)
    layers.Conv2D(512, (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(2,2),

    # Replaced Flatten with GAP2D
    layers.GlobalAveragePooling2D(),

    # Classifier Head
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# ===== COMPILE =====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== TRAIN WITH WEIGHTS =====
history = model.fit(
    train_generator,
    epochs=25, # Bumped epochs since augmentation slows down overfitting
    validation_data=test_generator,
    class_weight=class_weights_dict # Applying the imbalance fix
)

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)
print("\nTest Accuracy:", test_acc)

model.save("skin_cnn_deep_model.h5")