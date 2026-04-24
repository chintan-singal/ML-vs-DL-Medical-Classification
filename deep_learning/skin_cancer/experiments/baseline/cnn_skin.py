import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
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

# ===== CHECK DATA =====
print("Classes:\n", df["dx"].value_counts())

# ===== TRAIN-TEST SPLIT =====
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)

# ===== DATA GENERATORS =====
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="dx",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical"
)

test_generator = datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="dx",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ===== MODEL =====
num_classes = df["dx"].nunique()

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

    layers.Dense(num_classes, activation='softmax')
])

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
    epochs=15,
    validation_data=test_generator   # using test as validation for now
)

# ===== TEST =====
test_loss, test_acc = model.evaluate(test_generator)

print("\nTest Accuracy:", test_acc)

# ===== SAVE MODEL =====
model.save("skin_cnn_model.h5")