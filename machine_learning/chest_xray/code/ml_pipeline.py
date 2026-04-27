# ── Cell 0 ──────────────────────────────────────────────────
import pickle

# ── Cell 1 ──────────────────────────────────────────────────
with open('data/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/X_val.pkl', 'rb') as f:
    X_val = pickle.load(f)
with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('data/y_val.pkl', 'rb') as f:
    y_val = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
print("X_test shape: ", X_test.shape)

# ── Cell 2 ──────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ── Cell 3 ──────────────────────────────────────────────────
print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:  ", y_val.shape)
print("y_test shape: ", y_test.shape)

# ── Cell 4 ──────────────────────────────────────────────────
# Flatten the 4D image data (samples, height, width, channels) into 2D data (samples, features)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Original X_train shape: {X_train.shape}")
print(f"Flattened X_train shape: {X_train_flat.shape}")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_flat, y_train)

# ── Cell 5 ──────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# ── Cell 6 ──────────────────────────────────────────────────
import numpy as np

print("Train class distribution:", np.bincount(y_train))
print("Val class distribution:  ", np.bincount(y_val))
print("Test class distribution: ", np.bincount(y_test))

# ── Cell 7 ──────────────────────────────────────────────────
def evaluate(model, X, y, split_name="Validation"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n── {split_name} Results ──")
    print(f"  Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y, y_pred):.4f}")
    print(f"  F1-Score : {f1_score(y, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix — {split_name}")
    plt.show()

evaluate(lr, X_val_flat, y_val, "Validation")
evaluate(lr, X_test_flat, y_test, "Test")

# ── Cell 9 ──────────────────────────────────────────────────
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel

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
# 1. LOAD DATA
# ──────────────────────────────────────────
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

X_train = load_pickle('data/X_train.pkl')
X_val   = load_pickle('data/X_val.pkl')
X_test  = load_pickle('data/X_test.pkl')
y_train = load_pickle('data/y_train.pkl')
y_val   = load_pickle('data/y_val.pkl')
y_test  = load_pickle('data/y_test.pkl')

print(f"X_train shape: {X_train.shape}")

# ──────────────────────────────────────────
# 2. NORMALIZE TO [0, 1] FLOAT
#    (images already grayscale — just squeeze
#     channel dim if present and normalize)
# ──────────────────────────────────────────
def prepare(images):
    """Handle (N,H,W,1), (N,H,W) → (N,H,W) float32 in [0,1]."""
    imgs = images.astype(np.float32)
    if imgs.max() > 1.0:
        imgs /= 255.0
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)   # (N,H,W,1) → (N,H,W)
    return imgs

X_train = prepare(X_train)
X_val   = prepare(X_val)
X_test  = prepare(X_test)

print(f"After prepare — X_train shape: {X_train.shape}")  # should be (N, H, W)

# ──────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ──────────────────────────────────────────

def extract_flat(images):
    """Baseline: flatten raw pixels."""
    return images.reshape(images.shape[0], -1)

def extract_hog(images):
    """
    HOG: edge/gradient structure.
    Good at detecting lung opacity/consolidation patterns.
    """
    feats = []
    for img in images:
        f = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        feats.append(f)
    return np.array(feats, dtype=np.float32)

def extract_lbp(images, P=8, R=1, n_bins=256):
    """
    LBP: local texture patterns.
    Captures tissue texture differences between normal/pneumonia.
    """
    feats = []
    for img in images:
        lbp  = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats.append(hist)
    return np.array(feats, dtype=np.float32)

def extract_texture(images):
    """
    Statistical + Sobel edge features (7 values per image).
    Pneumonia lungs are hazier/brighter — global stats capture this.
    """
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

# ── Extract all ──
print("\nExtracting features (this may take a minute)...")

flat_tr = extract_flat(X_train);  flat_v = extract_flat(X_val);  flat_te = extract_flat(X_test)
print("  ✓ Flat pixels")

hog_tr  = extract_hog(X_train);   hog_v  = extract_hog(X_val);   hog_te  = extract_hog(X_test)
print("  ✓ HOG")

lbp_tr  = extract_lbp(X_train);   lbp_v  = extract_lbp(X_val);   lbp_te  = extract_lbp(X_test)
print("  ✓ LBP")

tex_tr  = extract_texture(X_train); tex_v = extract_texture(X_val); tex_te = extract_texture(X_test)
print("  ✓ Texture")

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
# 4. MODELS
# ──────────────────────────────────────────
# NOTE on Random Forest:
#   RF doesn't need scaling, but we include StandardScaler in the
#   pipeline anyway for consistency — it's a no-op for tree models.

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "SVM (RBF)":           SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=None,
                                                   random_state=42, n_jobs=-1),
}

# ──────────────────────────────────────────
# 5. EVALUATION HELPERS
# ──────────────────────────────────────────
def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'label'    : label,
        'Accuracy' : accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall'   : recall_score(y, y_pred),
        'F1'       : f1_score(y, y_pred),
        'ROC-AUC'  : roc_auc_score(y, y_prob),
        'y_pred'   : y_pred,
    }

def print_results(res, split):
    print(f"  {split:<12} | Acc: {res['Accuracy']:.4f}  Prec: {res['Precision']:.4f}"
          f"  Rec: {res['Recall']:.4f}  F1: {res['F1']:.4f}  AUC: {res['ROC-AUC']:.4f}")

def plot_confusion(y_true, y_pred, title):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────
# 6. TRAIN EVERYTHING & COLLECT RESULTS
# ──────────────────────────────────────────
# all_results[model_name][feature_set_name] = {val: ..., test: ...}
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

        val_res  = evaluate(pipe, Xv,  y_val,  "Validation")
        test_res = evaluate(pipe, Xte, y_test, "Test")

        print_results(val_res,  "Validation")
        print_results(test_res, "Test")

        plot_confusion(y_test, test_res['y_pred'],
                       f"{model_name} | {feat_name}")

        all_results[model_name][feat_name] = {'val': val_res, 'test': test_res}

# ──────────────────────────────────────────
# 7. FINAL SUMMARY TABLE  (Test Set)
# ──────────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

print("\n\n" + "="*80)
print("FINAL SUMMARY — Test Set")
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
