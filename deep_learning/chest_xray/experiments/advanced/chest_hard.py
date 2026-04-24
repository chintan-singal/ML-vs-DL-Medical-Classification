# ============================================================
# CHEST X-RAY PNEUMONIA DETECTION
# PyTorch EfficientNet + Hard Mining + XGBoost
# GPU Enabled
# ============================================================

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install timm xgboost scikit-learn joblib tqdm numpy

import os
import joblib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import timm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "ML_all_datasets/ML_all_datasets/data/raw/chest_xray"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32

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
# DATASETS
# ============================================================

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = ImageFolder(VAL_DIR, transform=test_transform)
test_dataset  = ImageFolder(TEST_DIR, transform=test_transform)

print("Classes:", train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# MODEL (EfficientNet-B0)
# ============================================================

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TRAIN
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
# EVAL
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

print("\n===== Stage 1 CNN Training =====")

for epoch in range(EPOCHS_STAGE1):
    loss = train_one_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}")

# ============================================================
# HARD MINING CNN
# ============================================================

print("\n===== Hard Mining CNN =====")

labels, preds, probs = evaluate(train_loader)

hard_indices = []

for i in range(len(labels)):

    conf = probs[i].max()

    if preds[i] != labels[i] or conf < 0.75:
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

print("\n===== Stage 2 Hard Mining CNN =====")

for epoch in range(EPOCHS_STAGE2):
    loss = train_one_epoch(hard_loader)
    print(f"Hard Epoch {epoch+1}: Loss={loss:.4f}")

# ============================================================
# SAVE CNN
# ============================================================

torch.save(model.state_dict(), "chest_xray_hardmined_cnn.pth")
print("Saved chest_xray_hardmined_cnn.pth")

# ============================================================
# FEATURE MODEL
# ============================================================

feature_model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=0,
    global_pool='avg'
)

feature_model.load_state_dict(
    torch.load("chest_xray_hardmined_cnn.pth"),
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

print("\nExtracting Validation Features...")
X_val, y_val = extract_features(val_loader)

print("\nExtracting Test Features...")
X_test, y_test = extract_features(test_loader)

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "chest_scaler.pkl")

# ============================================================
# XGBOOST STAGE 1
# ============================================================

xgb = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='hist',
    device='cuda'
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
        weights[i] = 3.5

print("\n===== XGBoost Hard Mining =====")

xgb.fit(
    X_train,
    y_train,
    sample_weight=weights
)

joblib.dump(xgb, "chest_xray_xgb.pkl")

# ============================================================
# FINAL TEST
# ============================================================

y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n===== FINAL RESULTS =====")
print("Accuracy:", acc)

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