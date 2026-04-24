import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

# ===== DATA =====
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(192,192),   # slightly bigger for DenseNet
    batch_size=16,           # reduced for RAM
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(192,192),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(192,192),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# ===== BASE MODEL =====
base_model = DenseNet121(
    input_shape=(192,192,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze

# ===== HEAD =====
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ===== COMPILE =====
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== TRAIN =====
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# ===== SAVE =====
model.save("brain_tumor_densenet.keras")

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)

print("\nTest Accuracy:", test_acc)