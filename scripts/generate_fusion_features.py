import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))
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
        x = self.fc2(x)
        return x

class TCGAImagingDataset(Dataset):
    def __init__(self, patient_ids, cache_dir):
        self.patient_ids = patient_ids
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pt_path = self.cache_dir / f"{pid}.pt"
        x = torch.load(pt_path, map_location="cpu", weights_only=False)
        return x, pid

def main():
    cli_path = Path("e:/rcc/data/tcga_kirc_clinical.csv")
    cache_dir = Path("e:/rcc/pretrained_weights/dicom_cache_v2")
    model3_path = Path("e:/rcc/models/model3_imaging.pt")
    out_path = Path("e:/rcc/data/fusion_features.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Clinical Data
    df = pd.read_csv(cli_path)
    df = df.dropna(subset=["metastasis"])
    df = df[df["has_dicom"] == True].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    
    # Restrict to available DICOMs
    pt_files = [p.stem for p in cache_dir.glob("*.pt")]
    df = df[df["patient_id"].isin(pt_files)].reset_index(drop=True)
    
    logging.info(f"Loaded {len(df)} patients for fusion feature extraction.")

    # 2. Extract p_imaging
    logging.info("Extracting p_imaging from Model 3...")
    model3 = Small3DCNN().to(device)
    model3.load_state_dict(torch.load(model3_path, map_location=device))
    model3.eval()

    dataset = TCGAImagingDataset(df["patient_id"].values, cache_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    p_imaging_dict = {}
    with torch.no_grad():
        for X_batch, pids in loader:
            X_batch = X_batch.to(device)
            out = model3(X_batch).squeeze(1)
            probs = torch.sigmoid(out).cpu().numpy()
            for pid, prob in zip(pids, probs):
                p_imaging_dict[pid] = prob

    df["p_imaging"] = df["patient_id"].map(p_imaging_dict)

    # 3. Extract p_clinical (Honest out-of-fold predictions)
    logging.info("Generating p_clinical via 5-Fold CV on TCGA features...")
    # Features: age, sex, t_stage_raw, n_stage_raw
    X_cli = df[["age", "sex", "t_stage_raw", "n_stage_raw"]]
    y = df["metastasis"].values

    # Preprocessing pipeline
    numeric_features = ["age"]
    categorical_features = ["sex", "t_stage_raw", "n_stage_raw"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

    cli_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_clinical = cross_val_predict(cli_model, X_cli, y, cv=cv, method="predict_proba")[:, 1]
    
    df["p_clinical"] = p_clinical

    # 4. Add missing modalities
    df["p_genomic"] = 0.5
    df["has_genomic"] = 0
    df["has_imaging"] = 1

    # Save features
    fusion_cols = ["patient_id", "metastasis", "p_clinical", "p_genomic", "p_imaging", "has_genomic", "has_imaging"]
    df_fusion = df[fusion_cols]
    df_fusion.to_csv(out_path, index=False)
    logging.info(f"Saved fusion features to {out_path}")

if __name__ == "__main__":
    main()
