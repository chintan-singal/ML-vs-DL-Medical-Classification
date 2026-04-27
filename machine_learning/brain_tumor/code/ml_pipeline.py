# ── Cell 0 ──────────────────────────────────────────────────
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from skimage.color import rgb2gray

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────
# CONFIG — just change this block per dataset
# ──────────────────────────────────────────
DATASET = "brain"   # "chest" | "brain" | "skin"

if DATASET == "chest":
    DATA_PATHS = {
        'X_train': 'data/processed/X_train.pkl',
        'X_val'  : 'data/processed/X_val.pkl',
        'X_test' : 'data/processed/X_test.pkl',
        'y_train': 'data/processed/y_train.pkl',
        'y_val'  : 'data/processed/y_val.pkl',
        'y_test' : 'data/processed/y_test.pkl',
    }
    CLASS_LABELS = ["Normal", "Pneumonia"]
    IS_RGB       = False

elif DATASET == "brain":
    DATA_PATHS = {
        'X_train': 'data/processed/brain_X_train.pkl',
        'X_val'  : 'data/processed/brain_X_val.pkl',
        'X_test' : 'data/processed/brain_X_test.pkl',
        'y_train': 'data/processed/brain_y_train.pkl',
        'y_val'  : 'data/processed/brain_y_val.pkl',
        'y_test' : 'data/processed/brain_y_test.pkl',
    }
    CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    IS_RGB       = False   # brain MRI scans are typically grayscale

elif DATASET == "skin":
    DATA_PATHS = {
        'X_train': 'data/processed/skin_X_train.pkl',
        'X_val'  : 'data/processed/skin_X_val.pkl',
        'X_test' : 'data/processed/skin_X_test.pkl',
        'y_train': 'data/processed/skin_y_train.pkl',
        'y_val'  : 'data/processed/skin_y_val.pkl',
        'y_test' : 'data/processed/skin_y_test.pkl',
    }
    CLASS_LABELS = ["BKL", "NV", "DF", "MEL", "VASC", "BCC", "AKIEC"]
    IS_RGB       = True    # skin lesion images are RGB

# ──────────────────────────────────────────
# Auto-detect binary vs multi-class
# ──────────────────────────────────────────
IS_BINARY    = len(CLASS_LABELS) == 2
N_CLASSES    = len(CLASS_LABELS)

print(f"\nDataset  : {DATASET.upper()}")
print(f"Classes  : {CLASS_LABELS}")
print(f"RGB      : {IS_RGB}")
print(f"Binary   : {IS_BINARY}")

# ──────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

X_train = load_pickle(DATA_PATHS['X_train'])
X_val   = load_pickle(DATA_PATHS['X_val'])
X_test  = load_pickle(DATA_PATHS['X_test'])
y_train = load_pickle(DATA_PATHS['y_train'])
y_val   = load_pickle(DATA_PATHS['y_val'])
y_test  = load_pickle(DATA_PATHS['y_test'])

print(f"\nX_train shape: {X_train.shape}")

# ──────────────────────────────────────────
# 2. PREPARE IMAGES → (N, H, W) float32
# ──────────────────────────────────────────
def prepare(images, is_rgb):
    imgs = images.astype(np.float32)
    if imgs.max() > 1.0:
        imgs /= 255.0

    if is_rgb:
        # (N, H, W, 3) → (N, H, W) grayscale
        if imgs.ndim == 4 and imgs.shape[-1] == 3:
            imgs = np.array([rgb2gray(img) for img in imgs])
        elif imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs.squeeze(-1)
    else:
        # Already grayscale — just squeeze channel dim if present
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs.squeeze(-1)

    return imgs  # (N, H, W)

X_train = prepare(X_train, IS_RGB)
X_val   = prepare(X_val,   IS_RGB)
X_test  = prepare(X_test,  IS_RGB)

print(f"After prepare — X_train shape: {X_train.shape}")

# ──────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ──────────────────────────────────────────
def extract_flat(images):
    return images.reshape(images.shape[0], -1)

def extract_hog(images):
    feats = []
    for img in images:
        f = hog(img, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        feats.append(f)
    return np.array(feats, dtype=np.float32)

def extract_lbp(images, P=8, R=1, n_bins=256):
    feats = []
    for img in images:
        lbp  = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats.append(hist)
    return np.array(feats, dtype=np.float32)

def extract_texture(images):
    feats = []
    for img in images:
        flat    = img.ravel()
        mean    = np.mean(flat)
        std     = np.std(flat)
        skew    = np.mean(((flat - mean) / (std + 1e-8)) ** 3)
        energy  = np.mean(flat ** 2)
        hist, _ = np.histogram(flat, bins=64, range=(0, 1), density=True)
        entropy = -np.sum((hist + 1e-10) * np.log(hist + 1e-10))
        edges   = sobel(img)
        feats.append([mean, std, skew, energy, entropy, np.mean(edges), np.std(edges)])
    return np.array(feats, dtype=np.float32)

print("\nExtracting features...")

flat_tr = extract_flat(X_train);    flat_v = extract_flat(X_val);    flat_te = extract_flat(X_test)
print("  ✓ Flat pixels")
hog_tr  = extract_hog(X_train);     hog_v  = extract_hog(X_val);     hog_te  = extract_hog(X_test)
print("  ✓ HOG")
lbp_tr  = extract_lbp(X_train);     lbp_v  = extract_lbp(X_val);     lbp_te  = extract_lbp(X_test)
print("  ✓ LBP")
tex_tr  = extract_texture(X_train); tex_v  = extract_texture(X_val); tex_te  = extract_texture(X_test)
print("  ✓ Texture")

combo_tr = np.concatenate([hog_tr, lbp_tr, tex_tr], axis=1)
combo_v  = np.concatenate([hog_v,  lbp_v,  tex_v],  axis=1)
combo_te = np.concatenate([hog_te, lbp_te, tex_te],  axis=1)
print("  ✓ Combined")

feature_sets = {
    "Flat Pixels (Baseline)": (flat_tr,  flat_v,  flat_te),
    "HOG":                    (hog_tr,   hog_v,   hog_te),
    "LBP":                    (lbp_tr,   lbp_v,   lbp_te),
    "Texture":                (tex_tr,   tex_v,   tex_te),
    "HOG + LBP + Texture":    (combo_tr, combo_v, combo_te),
}

# ──────────────────────────────────────────
# 4. MODELS
# ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0,
        multi_class='multinomial' if not IS_BINARY else 'auto'
    ),
    "SVM (RBF)": SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=42,
        decision_function_shape='ovr'       # one-vs-rest for multi-class
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

# ──────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────
def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # ROC-AUC: binary vs multi-class
    if IS_BINARY:
        auc = roc_auc_score(y, y_prob[:, 1])
    else:
        auc = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')

    # Precision / Recall / F1: use macro average for multi-class
    avg = 'binary' if IS_BINARY else 'macro'

    return {
        'Accuracy' : accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average=avg, zero_division=0),
        'Recall'   : recall_score(y, y_pred, average=avg, zero_division=0),
        'F1'       : f1_score(y, y_pred, average=avg, zero_division=0),
        'ROC-AUC'  : auc,
        'y_pred'   : y_pred,
    }

def print_results(res, split):
    print(f"  {split:<12} | Acc: {res['Accuracy']:.4f}  Prec: {res['Precision']:.4f}"
          f"  Rec: {res['Recall']:.4f}  F1: {res['F1']:.4f}  AUC: {res['ROC-AUC']:.4f}")

def plot_confusion(y_true, y_pred, title):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────
# 6. TRAIN EVERYTHING
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
# 7. SUMMARY TABLE
# ──────────────────────────────────────────
print("\n\n" + "="*80)
print(f"FINAL SUMMARY — Test Set ({DATASET.upper()})")
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
