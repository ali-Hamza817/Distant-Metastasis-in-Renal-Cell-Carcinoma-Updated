"""MMRCCNet training configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # Data
    "seer_csv": str(ROOT / "seer_rcc_2010_2018_clean.csv"),
    "clinical_features": [
        "age", "sex", "t_stage", "n_stage", "grade",
        "histology_enc", "tumor_size_cm",
    ],
    "label_col": "metastasis",
    "site_cols": ["lung_met", "bone_met", "liver_met", "brain_met"],
    "survival_col": "survival_months",
    "event_col": "vital_status",

    # Class imbalance (computed from SEER: 34534 neg / 2204 pos)
    "pos_weight": 15.67,

    # Training
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dim": 128,
    "num_heads": 4,
    "dropout": 0.3,
    "patience": 15,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "n_folds": 5,

    # Modalities
    "radiomics_top_k": 40,
    "genomics_dim": 500,
    "imaging_size": 64,

    # Paths
    "runs_dir": str(ROOT / "runs"),
    "data_dir": str(ROOT / "data"),
    "dicom_dir": str(ROOT / "data" / "dicom"),
    "genomics_dir": str(ROOT / "data" / "genomics"),
    "radiomics_dir": str(ROOT / "data" / "radiomics"),
    "manifest_dir": str(ROOT / "data" / "manifests"),
}
