import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
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
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
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

# ===== BASE MODEL =====
base_model = MobileNetV2(
    input_shape=(160,160,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Phase 1

# ===== HEAD =====
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ===== COMPILE =====
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== TRAIN (PHASE 1) =====
print("\n🔹 Phase 1: Training top layers")

history = model.fit(
    train_generator,
    epochs=8,
    validation_data=val_generator,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# ===== FINE-TUNING =====
print("\n🔹 Phase 2: Fine-tuning top layers of MobileNet")

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with LOW LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train again
history_ft = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# ===== SAVE =====
model.save("brain_tumor_mobilenet_ft.keras")

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)

print("\nFinal Test Accuracy:", test_acc)