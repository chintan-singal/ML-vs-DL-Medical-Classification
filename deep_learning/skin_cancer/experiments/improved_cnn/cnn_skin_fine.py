import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data2/raw/skin"

# ===== LOAD CSV =====
df = pd.read_csv(os.path.join(base_dir, "metadata.csv"))

# ===== IMAGE PATH =====
def get_image_path(image_id):
    path1 = os.path.join(base_dir, "images1", image_id + ".jpg")
    path2 = os.path.join(base_dir, "images2", image_id + ".jpg")
    return path1 if os.path.exists(path1) else path2

df["path"] = df["image_id"].apply(get_image_path)

print("Classes:\n", df["dx"].value_counts())

# ===== SPLIT =====
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["dx"],
    random_state=42
)

# ===== DATA GENERATORS =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical"
)

val_generator = test_datagen.flow_from_dataframe(
    val_df,
    x_col="path",
    y_col="dx",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="dx",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

# ===== CLASS WEIGHTS =====
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

# ===== FOCAL LOSS =====
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        return alpha * (1-pt)**gamma * ce
    return loss

# ===== MODEL =====
num_classes = df["dx"].nunique()

model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005),
                  input_shape=(224,224,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.MaxPooling2D(2,2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

# ===== COMPILE =====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=focal_loss(),
    metrics=['accuracy']
)

model.summary()

# ===== CALLBACKS =====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2
)

# ===== TRAIN =====
history = model.fit(
    train_generator,
    epochs=18,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)

print("\nTest Accuracy:", test_acc)

# ===== SAVE =====
model.save("skin_final_cnn.keras")