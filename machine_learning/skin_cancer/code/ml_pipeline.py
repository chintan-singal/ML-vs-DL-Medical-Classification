# ── Cell 0 ──────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from skimage.color import rgb2gray

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
IMG_SIZE    = (64, 64)   # resize all images to this
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1        # fraction of training set used for validation
RANDOM_SEED = 42

CLASS_LABELS = ["BKL", "NV", "DF", "MEL", "VASC", "BCC", "AKIEC"]
LABEL_MAP    = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}

# ──────────────────────────────────────────
# 1. DOWNLOAD & LOCATE FILES
# ──────────────────────────────────────────
print("Downloading dataset...")
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(f"Dataset path: {path}")

# Find CSV and image folders
csv_path = None
img_dirs  = []

for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.csv') and 'metadata' in f.lower():
            csv_path = os.path.join(root, f)
        if f.lower().endswith('.jpg'):
            img_dirs.append(root)
            break  # one entry per folder is enough

img_dirs = list(set(img_dirs))  # deduplicate
print(f"CSV found     : {csv_path}")
print(f"Image folders : {img_dirs}")

# ──────────────────────────────────────────
# 2. LOAD CSV & BUILD IMAGE PATH LOOKUP
# ──────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"\nCSV shape     : {df.shape}")
print(f"CSV columns   : {df.columns.tolist()}")
print(df['dx'].value_counts())

# Build a lookup: image_id → full file path
image_lookup = {}
for folder in img_dirs:
    for fname in os.listdir(folder):
        if fname.lower().endswith('.jpg'):
            image_id = os.path.splitext(fname)[0]
            image_lookup[image_id] = os.path.join(folder, fname)

print(f"\nTotal images found: {len(image_lookup)}")

# ──────────────────────────────────────────
# 3. LOAD & RESIZE IMAGES
# ──────────────────────────────────────────
def load_images(dataframe, lookup, img_size):
    """Load images as (N, H, W, 3) float32 in [0,1]."""
    images = []
    labels = []
    skipped = 0

    for _, row in dataframe.iterrows():
        img_id = row['image_id']
        label  = LABEL_MAP.get(row['dx'].lower(), -1)

        if label == -1 or img_id not in lookup:
            skipped += 1
            continue

        img = Image.open(lookup[img_id]).convert('RGB')
        img = img.resize(img_size, Image.LANCZOS)
        images.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(label)

    print(f"Loaded: {len(images)} images | Skipped: {skipped}")
    return np.array(images), np.array(labels)

print("\nLoading images...")
X, y = load_images(df, image_lookup, IMG_SIZE)
print(f"X shape: {X.shape}")   # (N, 64, 64, 3)
print(f"y shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ──────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=RANDOM_SEED, stratify=y_train_full
)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape[0]}")
print(f"  Val   : {X_val.shape[0]}")
print(f"  Test  : {X_test.shape[0]}")

# ──────────────────────────────────────────
# 5. FEATURE EXTRACTION (RGB-aware)
# ──────────────────────────────────────────

def to_gray(images):
    """(N,H,W,3) → (N,H,W) grayscale for HOG/LBP/Texture."""
    return np.array([rgb2gray(img) for img in images])

def extract_flat_rgb(images):
    """Flatten all 3 channels: (N,H,W,3) → (N, H*W*3)."""
    return images.reshape(images.shape[0], -1)

def extract_hog_rgb(images):
    """
    HOG per channel → concatenate.
    Captures edge structure in each colour channel separately.
    """
    feats = []
    for img in images:
        ch_feats = []
        for c in range(3):
            f = hog(img[:, :, c], orientations=9,
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', feature_vector=True)
            ch_feats.append(f)
        feats.append(np.concatenate(ch_feats))
    return np.array(feats, dtype=np.float32)

def extract_lbp_rgb(images, P=8, R=1, n_bins=256):
    """
    LBP per channel → concatenate histograms.
    Skin lesion textures differ across R/G/B channels.
    """
    feats = []
    for img in images:
        ch_feats = []
        for c in range(3):
            lbp  = local_binary_pattern(img[:, :, c], P=P, R=R, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            ch_feats.append(hist)
        feats.append(np.concatenate(ch_feats))
    return np.array(feats, dtype=np.float32)

def extract_texture_rgb(images):
    """
    Statistical + Sobel features per channel (7 x 3 = 21 features).
    Colour statistics are very informative for skin lesion classification.
    """
    feats = []
    for img in images:
        ch_feats = []
        for c in range(3):
            ch   = img[:, :, c].ravel()
            mean = np.mean(ch)
            std  = np.std(ch)
            skew = np.mean(((ch - mean) / (std + 1e-8)) ** 3)
            energy  = np.mean(ch ** 2)
            hist, _ = np.histogram(ch, bins=64, range=(0, 1), density=True)
            entropy = -np.sum((hist + 1e-10) * np.log(hist + 1e-10))
            edges   = sobel(img[:, :, c])
            ch_feats.extend([mean, std, skew, energy, entropy,
                              np.mean(edges), np.std(edges)])
        feats.append(ch_feats)
    return np.array(feats, dtype=np.float32)

# ── Extract all ──
print("\nExtracting features (this may take a few minutes)...")

flat_tr = extract_flat_rgb(X_train); flat_v = extract_flat_rgb(X_val); flat_te = extract_flat_rgb(X_test)
print("  ✓ Flat pixels (RGB)")

hog_tr  = extract_hog_rgb(X_train);  hog_v  = extract_hog_rgb(X_val);  hog_te  = extract_hog_rgb(X_test)
print("  ✓ HOG (per channel)")

lbp_tr  = extract_lbp_rgb(X_train);  lbp_v  = extract_lbp_rgb(X_val);  lbp_te  = extract_lbp_rgb(X_test)
print("  ✓ LBP (per channel)")

tex_tr  = extract_texture_rgb(X_train); tex_v = extract_texture_rgb(X_val); tex_te = extract_texture_rgb(X_test)
print("  ✓ Texture (per channel)")

combo_tr = np.concatenate([hog_tr, lbp_tr, tex_tr], axis=1)
combo_v  = np.concatenate([hog_v,  lbp_v,  tex_v],  axis=1)
combo_te = np.concatenate([hog_te, lbp_te, tex_te],  axis=1)
print("  ✓ Combined (HOG + LBP + Texture)")

print(f"\nFeature dimensions:")
print(f"  Flat    : {flat_tr.shape[1]}")
print(f"  HOG     : {hog_tr.shape[1]}")
print(f"  LBP     : {lbp_tr.shape[1]}")
print(f"  Texture : {tex_tr.shape[1]}")
print(f"  Combined: {combo_tr.shape[1]}")

feature_sets = {
    "Flat Pixels (Baseline)": (flat_tr,  flat_v,  flat_te),
    "HOG":                    (hog_tr,   hog_v,   hog_te),
    "LBP":                    (lbp_tr,   lbp_v,   lbp_te),
    "Texture":                (tex_tr,   tex_v,   tex_te),
    "HOG + LBP + Texture":    (combo_tr, combo_v, combo_te),
}

# ──────────────────────────────────────────
# 6. MODELS
# ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, multi_class='multinomial'
    ),
    "SVM (RBF)": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42,
        decision_function_shape='ovr'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

# ──────────────────────────────────────────
# 7. EVALUATION
# ──────────────────────────────────────────
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    auc    = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')
    return {
        'Accuracy' : accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='macro', zero_division=0),
        'Recall'   : recall_score(y, y_pred, average='macro', zero_division=0),
        'F1'       : f1_score(y, y_pred, average='macro', zero_division=0),
        'ROC-AUC'  : auc,
        'y_pred'   : y_pred,
    }

def print_results(res, split):
    print(f"  {split:<12} | Acc: {res['Accuracy']:.4f}  Prec: {res['Precision']:.4f}"
          f"  Rec: {res['Recall']:.4f}  F1: {res['F1']:.4f}  AUC: {res['ROC-AUC']:.4f}")

def plot_confusion(y_true, y_pred, title):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────
# 8. TRAIN EVERYTHING
# ──────────────────────────────────────────
all_results = {m: {} for m in models}

for feat_name, (Xtr, Xv, Xte) in feature_sets.items():
    print(f"\n{'='*60}")
    print(f"  Feature Set: {feat_name}")
    print(f"{'='*60}")

    for model_name, clf in models.items():
        print(f"\n  ▸ {model_name}")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    clf)
        ])
        pipe.fit(Xtr, y_train)

        val_res  = evaluate(pipe, Xv,  y_val)
        test_res = evaluate(pipe, Xte, y_test)

        print_results(val_res,  "Validation")
        print_results(test_res, "Test")

        plot_confusion(y_test, test_res['y_pred'],
                       f"{model_name} | {feat_name}")

        all_results[model_name][feat_name] = {'val': val_res, 'test': test_res}

# ──────────────────────────────────────────
# 9. SUMMARY TABLE
# ──────────────────────────────────────────
print("\n\n" + "="*80)
print("FINAL SUMMARY — Test Set (SKIN CANCER)")
print("="*80)

for model_name in models:
    print(f"\n  {model_name}")
    print(f"  {'Feature Set':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-"*60)
    for feat_name in feature_sets:
        r = all_results[model_name][feat_name]['test']
        print(f"  {feat_name:<25} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} "
              f"{r['Recall']:>7.4f} {r['F1']:>7.4f} {r['ROC-AUC']:>7.4f}")

print("\nDone!")
