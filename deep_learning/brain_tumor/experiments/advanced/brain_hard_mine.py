# ============================================================
# Brain Tumor Classification
# PyTorch CNN + Hard Mining + Feature Extraction + XGBoost
# GPU Enabled
# ============================================================

# ================= INSTALL FIRST =================
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install timm xgboost scikit-learn numpy pandas matplotlib tqdm pillow

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "ML_all_datasets/ML_all_datasets/data1/raw/brain_tumor"

TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR  = os.path.join(BASE_DIR, "Testing")

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ============================================================
# DATASET
# ============================================================

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = ImageFolder(TEST_DIR, transform=test_transform)

num_classes = len(train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_dataset.classes)

# ============================================================
# MODEL (EfficientNet)
# ============================================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================
# EVAL FUNCTION
# ============================================================

def evaluate(loader):
    model.eval()

    preds = []
    labels = []
    probs_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)

            out = model(x)
            prob = torch.softmax(out, dim=1)

            pred = torch.argmax(prob, dim=1).cpu().numpy()

            preds.extend(pred)
            labels.extend(y.numpy())
            probs_all.extend(prob.cpu().numpy())

    return np.array(labels), np.array(preds), np.array(probs_all)

# ============================================================
# STAGE 1 NORMAL TRAINING
# ============================================================

print("\n===== Stage 1 Training =====")

for epoch in range(EPOCHS_STAGE1):
    loss = train_one_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}")

# ============================================================
# HARD MINING FOR CNN
# ============================================================

print("\n===== Hard Mining CNN =====")

labels, preds, probs = evaluate(train_loader)

hard_indices = []

for i in range(len(labels)):
    conf = probs[i].max()

    if preds[i] != labels[i] or conf < 0.65:
        hard_indices.append(i)

print("Hard Samples Found:", len(hard_indices))

weights = np.ones(len(train_dataset))

for idx in hard_indices:
    weights[idx] = 4.0

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

hard_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

# ============================================================
# STAGE 2 HARD MINING TRAINING
# ============================================================

print("\n===== Stage 2 Hard Mining Training =====")

for epoch in range(EPOCHS_STAGE2):
    loss = train_one_epoch(hard_loader)
    print(f"Hard Epoch {epoch+1}: Loss={loss:.4f}")

# ============================================================
# SAVE CNN
# ============================================================

torch.save(model.state_dict(), "brain_tumor_hardmined_cnn.pth")

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=0,
    global_pool='avg'
)

feature_model.load_state_dict(
    torch.load("brain_tumor_hardmined_cnn.pth"),
    strict=False
)

feature_model = feature_model.to(DEVICE)
feature_model.eval()

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(loader):
    feats = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)

            f = feature_model(x)

            feats.append(f.cpu().numpy())
            labels.extend(y.numpy())

    return np.vstack(feats), np.array(labels)

print("\nExtracting Train Features...")
X_train, y_train = extract_features(train_loader)

print("\nExtracting Test Features...")
X_test, y_test = extract_features(test_loader)

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# ============================================================
# XGBOOST ROUND 1
# ============================================================

xgb = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1
)

print("\n===== XGBoost Stage 1 =====")
xgb.fit(X_train, y_train)

pred1 = xgb.predict(X_train)

# ============================================================
# HARD MINING XGBOOST
# ============================================================

weights = np.ones(len(y_train))

for i in range(len(y_train)):
    if pred1[i] != y_train[i]:
        weights[i] = 3.0

print("Retraining XGBoost with Hard Mining...")

xgb.fit(X_train, y_train, sample_weight=weights)

joblib.dump(xgb, "brain_tumor_xgb.pkl")

# ============================================================
# FINAL TEST
# ============================================================

y_pred = xgb.predict(X_test)

print("\n===== FINAL RESULTS =====")

print("Accuracy:", (y_pred == y_test).mean())

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=train_dataset.classes
    )
)