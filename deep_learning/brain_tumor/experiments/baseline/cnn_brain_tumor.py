import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# ===== PATH =====
base_dir = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")  # if exists

# ===== DATA =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 🔥 create validation from training
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
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

# ===== MODEL =====
model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')  # 🔥 4 classes
])

# ===== COMPILE =====
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== TRAIN =====
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# ===== SAVE =====
model.save("brain_tumor_cnn.h5")


# ===== TEST =====
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False   # 🔥 important for evaluation
)

test_loss, test_acc = model.evaluate(test_generator)

print("\nTest Accuracy:", test_acc)