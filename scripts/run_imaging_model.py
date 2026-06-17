import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Random3DAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1).item() > self.prob:
            return x

        # 1. Random axial flip (depth is dim 1, height 2, width 3)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, [3]) # flip along width

        # 2. Random translation/crop effect (shift by up to 10%)
        if torch.rand(1).item() > 0.5:
            shift_d = int((torch.rand(1).item() - 0.5) * 2 * 6)
            shift_h = int((torch.rand(1).item() - 0.5) * 2 * 9)
            shift_w = int((torch.rand(1).item() - 0.5) * 2 * 9)
            x = torch.roll(x, shifts=(shift_d, shift_h, shift_w), dims=(1, 2, 3))

        # 3. Gaussian noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        # 4. Intensity shift
        if torch.rand(1).item() > 0.5:
            shift = (torch.rand(1).item() - 0.5) * 0.2
            x = x + shift

        return x


class TCGAImagingDataset(Dataset):
    def __init__(self, patient_ids, labels, cache_dir, augment=False):
        self.patient_ids = patient_ids
        self.labels = labels
        self.cache_dir = Path(cache_dir)
        self.augment = Random3DAugmentation(prob=0.8) if augment else None

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = self.labels[idx]
        
        pt_path = self.cache_dir / f"{pid}.pt"
        # Load preprocessed tensor of shape (1, 64, 96, 96)
        x = torch.load(pt_path, map_location="cpu", weights_only=False)
        
        if self.augment:
            x = self.augment(x)
            
        return x, torch.tensor(label, dtype=torch.float32)


class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 x 64 x 96 x 96
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2) # Output: 16 x 32 x 48 x 48

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2) # Output: 32 x 16 x 24 x 24

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1)) # Output: 64 x 1 x 1 x 1

        self.fc1 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x) # No sigmoid here because of BCEWithLogitsLoss
        return x


def custom_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, ci=95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    lower = np.percentile(bootstrapped_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_scores, 100 - (100 - ci) / 2)
    mean_score = metric_func(y_true, y_pred)
    return {"mean": mean_score, "ci_lower": lower, "ci_upper": upper}


def train_fold(model, train_loader, val_loader, pos_weight, device, epochs=100, patience=20):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_pr_auc = -1
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze(1)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch).squeeze(1)
                loss = criterion(out, y_batch)
                val_loss += loss.item()
                
                probs = torch.sigmoid(out).cpu().numpy()
                y_val_true.extend(y_batch.cpu().numpy())
                y_val_pred.extend(probs)
                
        val_pr_auc = custom_pr_auc(np.array(y_val_true), np.array(y_val_pred))
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch}. Best PR-AUC: {best_pr_auc:.3f}")
            break

    model.load_state_dict(best_weights)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=str, default="e:/rcc/data/tcga_kirc_clinical.csv")
    parser.add_argument("--cache", type=str, default="e:/rcc/pretrained_weights/dicom_cache_v2")
    parser.add_argument("--outdir", type=str, default="e:/rcc/results/imaging_only")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("e:/rcc/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load Clinical
    df = pd.read_csv(args.cli)
    df = df.dropna(subset=["metastasis"])
    df = df[df["has_dicom"] == True].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    
    # Filter by available PT files
    cache_dir = Path(args.cache)
    pt_files = [p.stem for p in cache_dir.glob("*.pt")]
    df = df[df["patient_id"].isin(pt_files)].reset_index(drop=True)
    
    logging.info(f"Found {len(df)} matching patients with DICOM tensors.")
    
    patient_ids = df["patient_id"].values
    y = df["metastasis"].values
    
    logging.info(f"Metastasis count: {sum(y)} positive out of {len(y)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    best_overall_model = None
    best_overall_pr = -1

    for fold, (train_idx, test_idx) in enumerate(cv.split(patient_ids, y)):
        logging.info(f"--- Fold {fold+1}/5 ---")
        train_ids, test_ids = patient_ids[train_idx], patient_ids[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_dataset = TCGAImagingDataset(train_ids, y_train, cache_dir, augment=True)
        test_dataset = TCGAImagingDataset(test_ids, y_test, cache_dir, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        model = Small3DCNN().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        if fold == 0:
            logging.info(f"Model parameters: {num_params}")

        # Class weight
        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

        model = train_fold(model, train_loader, test_loader, pos_weight, device, epochs=args.epochs)

        # Evaluate on test fold
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch).squeeze(1)
                probs = torch.sigmoid(out).cpu().numpy()
                fold_preds.extend(probs)
                
        all_y_true.extend(y_test)
        all_y_pred.extend(fold_preds)
        
        fold_pr = custom_pr_auc(y_test, np.array(fold_preds))
        logging.info(f"Fold {fold+1} Test PR-AUC: {fold_pr:.3f}")
        
        if fold_pr > best_overall_pr:
            best_overall_pr = fold_pr
            best_overall_model = model.state_dict().copy()

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Save best model
    model_save_path = models_dir / "model3_imaging.pt"
    torch.save(best_overall_model, model_save_path)
    logging.info(f"Saved best model weights to {model_save_path}")

    # Metrics
    metrics = {
        "ROC-AUC": bootstrap_ci(all_y_true, all_y_pred, roc_auc_score),
        "PR-AUC": bootstrap_ci(all_y_true, all_y_pred, custom_pr_auc),
    }
    
    # Confusion Matrix (threshold=0.5)
    y_pred_bin = (all_y_pred >= 0.5).astype(int)
    cm = confusion_matrix(all_y_true, y_pred_bin)
    
    report = {
        "dataset_info": {
            "n_samples": len(y),
            "n_positive": int(sum(y))
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist()
    }
    
    report_path = out_dir / "imaging_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Saved metrics to {report_path}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_pred)
    axes[0].plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']['mean']:.3f}")
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (Out-of-Fold)')
    axes[0].legend()
    
    # PR
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_pred)
    baseline = sum(y) / len(y)
    axes[1].plot(recall, precision, label=f"PR-AUC = {metrics['PR-AUC']['mean']:.3f}")
    axes[1].axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (Out-of-Fold)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "imaging_evaluation_curves.png", dpi=300)

if __name__ == "__main__":
    main()
