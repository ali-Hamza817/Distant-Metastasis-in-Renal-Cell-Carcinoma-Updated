"""PHASE 8 — Standalone DeLong test + quick CV summary from saved checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG
from data.datasets import SEERDataset, load_seer_splits
from models.mmrccnet import MMRCCNet
from scripts.train_lightgbm_baseline import train_and_evaluate
from utils.stats_validation import cv_summary, delong_test, save_validation_report
from utils.training import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(CONFIG["runs_dir"]) / "clinical_only"
    ckpt_path = out_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print("No clinical checkpoint found. Run Phase 2 first.")
        return

    splits = load_seer_splits(CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"])
    df = splits["df"]
    common = dict(
        df=df, clinical_features=CONFIG["clinical_features"], label_col=CONFIG["label_col"],
        site_cols=CONFIG["site_cols"], survival_col=CONFIG["survival_col"],
        event_col=CONFIG["event_col"], scaler=splits["scaler"],
    )
    test_ds = SEERDataset(indices=splits["test_idx"], **common)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    model = MMRCCNet(clinical_dim=len(CONFIG["clinical_features"]), use_clinical=True,
                     use_radiomics=False, use_imaging=False, use_genomics=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    mmrcc_probs, y_true = [], []
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model(clinical=batch["clinical"])
            mmrcc_probs.append(torch.sigmoid(out["metastasis_logit"]).cpu().numpy())
            y_true.append(batch["metastasis"].cpu().numpy())
    mmrcc_probs = np.concatenate(mmrcc_probs)
    y_true = np.concatenate(y_true)

    lgb = train_and_evaluate(df, splits["train_idx"], splits["test_idx"])
    delong = delong_test(y_true, mmrcc_probs, lgb["prob"])

    baseline_cv = json.loads((Path(CONFIG["runs_dir"]) / "baseline" / "baseline_results.json").read_text())

    report = {
        "delong_test": delong,
        "mmrcc_test_auc": float(evaluate(model, test_loader, device, False, False, True, False)["auc"]),
        "lightgbm_holdout_auc": lgb["auc"],
        "lightgbm_cv": cv_summary(baseline_cv["cv_folds"]),
        "significant_at_0.05": delong["p_value"] < 0.05,
        "interpretation": (
            "MMRCCNet significantly outperforms LightGBM"
            if delong["p_value"] < 0.05 and delong["auc_a"] > delong["auc_b"]
            else "No significant improvement at p<0.05"
        ),
    }
    save_validation_report(report, out_dir / "statistical_validation.json")

    print("=" * 60)
    print("PHASE 8 — Statistical Validation")
    print("=" * 60)
    print(f"MMRCCNet test AUC: {report['mmrcc_test_auc']:.4f}")
    print(f"LightGBM test AUC: {lgb['auc']:.4f}")
    print(f"DeLong z={delong['z']:.3f}, p={delong['p_value']:.4e}")
    print(f"Significant (p<0.05): {report['significant_at_0.05']}")
    print(f"Saved: {out_dir / 'statistical_validation.json'}")


if __name__ == "__main__":
    main()
