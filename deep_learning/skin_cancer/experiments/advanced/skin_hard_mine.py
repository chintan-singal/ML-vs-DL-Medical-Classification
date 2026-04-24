# ===============================================================
# SKIN LESION CLASSIFICATION (UPGRADED)
# PyTorch + GPU + Targeted Hard Mining CNN + XGBoost
# Optimized for df / mel / vasc recall
# Windows Safe
# ===============================================================

# INSTALL:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install timm pandas numpy pillow tqdm scikit-learn xgboost joblib

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ===============================================================
# CONFIG
# ===============================================================

BASE_DIR = "ML_all_datasets/ML_all_datasets/data2/raw/skin"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4

EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", DEVICE)

# ===============================================================
# LOAD CSV
# ===============================================================

df = pd.read_csv(CSV_PATH)

def get_image_path(image_id):
    p1 = os.path.join(BASE_DIR, "images1", image_id + ".jpg")
    p2 = os.path.join(BASE_DIR, "images2", image_id + ".jpg")
    return p1 if os.path.exists(p1) else p2

df["path"] = df["image_id"].apply(get_image_path)

# ===============================================================
# LABEL ENCODING
# ===============================================================

classes = sorted(df["dx"].unique())
class_to_idx = {c:i for i, c in enumerate(classes)}
idx_to_class = {i:c for c, i in class_to_idx.items()}

df["label"] = df["dx"].map(class_to_idx)

print("Classes:", classes)
print(df["dx"].value_counts())

# ===============================================================
# TRAIN TEST SPLIT
# ===============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# ===============================================================
# DATASET
# ===============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

train_dataset = SkinDataset(train_df, train_transform)
test_dataset = SkinDataset(test_df, test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ===============================================================
# MODEL
# ===============================================================

num_classes = len(classes)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================================================
# TRAIN FUNCTION
# ===============================================================

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

# ===============================================================
# EVALUATE FUNCTION
# ===============================================================

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

# ===============================================================
# STAGE 1 NORMAL TRAINING
# ===============================================================

print("\n===== Stage 1 Training =====")

for epoch in range(EPOCHS_STAGE1):
    loss = train_one_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}")

# ===============================================================
# TARGETED HARD MINING CNN
# ===============================================================

print("\n===== Targeted Hard Mining CNN =====")

labels, preds, probs = evaluate(train_loader)

priority_classes = {
    class_to_idx["df"],
    class_to_idx["mel"],
    class_to_idx["vasc"]
}

weights = np.ones(len(train_dataset))

for i in range(len(labels)):

    true_cls = labels[i]
    pred_cls = preds[i]
    conf = probs[i].max()

    if true_cls in priority_classes and pred_cls != true_cls:
        weights[i] = 8.0

    elif true_cls in priority_classes and conf < 0.75:
        weights[i] = 5.0

    elif pred_cls != true_cls:
        weights[i] = 3.0

    elif conf < 0.60:
        weights[i] = 2.0

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

hard_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

# ===============================================================
# STAGE 2 HARD TRAINING
# ===============================================================

print("\n===== Stage 2 Hard Mining Training =====")

for epoch in range(EPOCHS_STAGE2):
    loss = train_one_epoch(hard_loader)
    print(f"Hard Epoch {epoch+1}: Loss={loss:.4f}")

# ===============================================================
# SAVE CNN
# ===============================================================

torch.save(model.state_dict(), "skin_hardmined_targeted_cnn.pth")

# ===============================================================
# FEATURE EXTRACTOR
# ===============================================================

feature_model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=0,
    global_pool="avg"
)

feature_model.load_state_dict(
    torch.load("skin_hardmined_targeted_cnn.pth"),
    strict=False
)

feature_model = feature_model.to(DEVICE)
feature_model.eval()

# ===============================================================
# FEATURE EXTRACTION
# ===============================================================

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

# ===============================================================
# SCALE
# ===============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "skin_scaler.pkl")

# ===============================================================
# XGBOOST ROUND 1
# ===============================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    n_jobs=-1
)

print("\n===== XGBoost Stage 1 =====")
xgb.fit(X_train, y_train)

pred_train = xgb.predict(X_train)

# ===============================================================
# TARGETED HARD MINING XGB
# ===============================================================

sample_weights = np.ones(len(y_train))

for i in range(len(y_train)):

    true_cls = y_train[i]
    pred_cls = pred_train[i]

    if true_cls in priority_classes and pred_cls != true_cls:
        sample_weights[i] = 6.0

    elif pred_cls != true_cls:
        sample_weights[i] = 3.0

print("Retraining XGBoost with Targeted Hard Mining...")

xgb.fit(X_train, y_train, sample_weight=sample_weights)

joblib.dump(xgb, "skin_xgb.pkl")

# ===============================================================
# FINAL TEST
# ===============================================================

y_pred = xgb.predict(X_test)

print("\n===== FINAL RESULTS =====")

acc = (y_pred == y_test).mean()
print("Test Accuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=classes
    )
)