import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# =====================================================
# WINDOWS SAFE ENTRY
# =====================================================
def main():

    # =================================================
    # DEVICE
    # =================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =================================================
    # PATHS
    # =================================================
    base_dir = "ML_all_datasets/ML_all_datasets/data/raw/chest_xray"

    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    # =================================================
    # TRANSFORMS
    # =================================================
    train_transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor()
    ])

    # =================================================
    # DATASETS
    # =================================================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Classes:", train_dataset.classes)

    # =================================================
    # CNN MODEL
    # =================================================
    class ChestCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 128, 3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.flatten = nn.Flatten()

            self.feature_layer = nn.Sequential(
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            self.classifier = nn.Linear(512, 1)

        def forward(self, x):
            x = self.features(x)
            x = self.flatten(x)
            feat = self.feature_layer(x)
            out = self.classifier(feat)
            return out, feat

    model = ChestCNN().to(device)

    # =================================================
    # LOSS + OPTIMIZER
    # =================================================
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =================================================
    # TRAIN CNN
    # =================================================
    epochs = 15
    best_val = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs, _ = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # =============================
        # VALIDATION
        # =============================
        model.eval()

        preds = []
        actual = []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)

                outputs, _ = model(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).int().cpu().numpy()

                preds.extend(predicted.flatten())
                actual.extend(labels.numpy())

        val_acc = accuracy_score(actual, preds)

        print(f"Epoch {epoch+1}/{epochs} Loss {running_loss:.4f} Val Acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "chest_xray_cnn.pth")
            print("Saved best CNN model")

    # =================================================
    # LOAD BEST MODEL
    # =================================================
    model.load_state_dict(torch.load("chest_xray_cnn.pth"))
    model.eval()

    # =================================================
    # FEATURE EXTRACTION
    # =================================================
    def get_features(loader):

        X = []
        y = []

        with torch.no_grad():
            for images, labels in loader:

                images = images.to(device)

                _, feat = model(images)

                X.append(feat.cpu().numpy())
                y.append(labels.numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        return X, y

    print("Extracting features...")

    X_train, y_train = get_features(train_loader)
    X_val, y_val     = get_features(val_loader)
    X_test, y_test   = get_features(test_loader)

    print("Feature shape:", X_train.shape)

    # =================================================
    # XGBOOST GPU
    # =================================================
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda"
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # =================================================
    # TEST
    # =================================================
    y_pred = xgb_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nXGBoost Test Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # =================================================
    # SAVE XGB
    # =================================================
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    print("Saved xgboost_model.pkl")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()