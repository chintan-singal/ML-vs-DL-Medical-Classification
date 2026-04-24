import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
import cv2
import os

# ===== PATHS =====
base_dir = r"C:\Users\chint\OneDrive\Documents\ML Project\ML_all_datasets\ML_all_datasets\data1\raw\brain_tumor"
train_dir = os.path.join(base_dir, "Training")
test_dir  = os.path.join(base_dir, "Testing")

IMG_SIZE    = 300   # B3 optimal size
BATCH_SIZE  = 32
NUM_CLASSES = 4

# ===================================================
# KEY INSIGHT: Keep output as valid RGB (3ch float)
# so EfficientNet pretrained weights still work.
# We enhance the image spatially — not replace channels.
# ===================================================

def mri_enhance(img):
    """
    Edge-aware enhancement that stays in RGB space.
    Pipeline:
      1. CLAHE on L channel (LAB space) — contrast only, no hue shift
      2. Unsharp mask using multi-scale Laplacian — sharpens tumor borders
      3. Anomaly-weighted blend — brightens suspicious regions slightly
    Output: valid RGB float32 in [0, 1]
    """
    img_uint8 = (img * 255).astype(np.uint8)

    # ---- Step 1: CLAHE in LAB space (only L channel) ----
    lab   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # ---- Step 2: Unsharp mask (sharpens edges without channel swap) ----
    # Multi-scale: fine detail (sigma=1) + coarser structure (sigma=3)
    blur_fine   = cv2.GaussianBlur(enhanced, (0, 0), 1)
    blur_coarse = cv2.GaussianBlur(enhanced, (0, 0), 3)

    # Laplacian-of-Gaussian edge map (combined scale)
    log_fine   = cv2.subtract(enhanced, blur_fine)
    log_coarse = cv2.subtract(enhanced, blur_coarse)
    log_combined = cv2.addWeighted(log_fine, 0.6, log_coarse, 0.4, 0)

    # Add sharpening back into enhanced image
    sharpened = cv2.addWeighted(enhanced, 1.0, log_combined, 0.7, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # ---- Step 3: Anomaly region brightening ----
    gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)

    # Otsu threshold to find bright regions (potential tumor mass)
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up mask — remove skull, keep interior blobs only
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Erode to strip outer skull ring
    skull_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    inner   = cv2.erode(mask, skull_k, iterations=2)

    # Soft mask: gaussian blur the binary mask → smooth attention region
    soft_mask = cv2.GaussianBlur(inner.astype(np.float32), (31, 31), 10)
    soft_mask = soft_mask / (soft_mask.max() + 1e-8)  # normalize to [0,1]
    soft_mask_3ch = np.stack([soft_mask]*3, axis=-1)

    # Brighten anomaly regions by 15% — subtle, stays in RGB
    brightened = sharpened.astype(np.float32) * (1.0 + 0.15 * soft_mask_3ch)
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)

    return brightened.astype(np.float32) / 255.0


# ===== DATA GENERATORS =====
train_datagen = ImageDataGenerator(
    preprocessing_function=mri_enhance,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=mri_enhance
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ===== MODEL =====
# Using B3 — better capacity than B0, still fast enough
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)

# Lightweight spatial attention — won't destabilize gradients
# (simpler than CBAM, more stable)
gap = layers.GlobalAveragePooling2D()(x)
gmp = layers.GlobalMaxPooling2D()(x)
x   = layers.Add()([gap, gmp])          # fuse avg + max pool signals
x   = layers.BatchNormalization()(x)
x   = layers.Dense(512, activation='relu')(x)
x   = layers.Dropout(0.5)(x)
x   = layers.Dense(128, activation='relu')(x)
x   = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',   # removed label smoothing for now
    metrics=['accuracy']
)

model.summary()

# ===== PHASE 1: Head only =====
print("\n=== Phase 1: Head only (10 epochs) ===")
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True,
                      monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                         patience=2, min_lr=1e-6),
        ModelCheckpoint("best_phase1.keras",
                        save_best_only=True, monitor='val_accuracy')
    ]
)

# ===== PHASE 2: Fine-tune last 30 layers =====
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== Phase 2: Fine-tuning (20 epochs) ===")
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=2, min_lr=1e-7),
        ModelCheckpoint("best_phase2.keras",
                        save_best_only=True, monitor='val_accuracy')
    ]
)

# ===== EVALUATE =====
model.save("brain_tumor_final.keras")
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")


# ===== SANITY CHECK — run this before training =====
def sanity_check(image_path):
    """
    Visually verify the preprocessing looks right.
    Tumor region should appear slightly brighter/sharper.
    Run on 1 tumor image AND 1 no_tumor image.
    """
    import matplotlib.pyplot as plt

    raw = cv2.imread(image_path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    raw = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
    raw_norm = raw / 255.0

    out = mri_enhance(raw_norm)

    diff = np.abs(out - raw_norm)   # show what changed

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw)
    axes[0].set_title("Original")
    axes[1].imshow(out)
    axes[1].set_title("Enhanced (should be sharper at tumor)")
    axes[2].imshow(diff / diff.max(), cmap='hot')
    axes[2].set_title("Change map (hot = where we modified)")
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150)
    plt.show()

# ---- Uncomment and run BEFORE training to verify ----
sanity_check(r"C:\Users\chint\OneDrive\Documents\ML Project\ML_all_datasets\ML_all_datasets\data1\raw\brain_tumor\Training\glioma\Tr-gl_21.jpg")
sanity_check(r"C:\Users\chint\OneDrive\Documents\ML Project\ML_all_datasets\ML_all_datasets\data1\raw\brain_tumor\Training\notumor\Tr-no_8.jpg")