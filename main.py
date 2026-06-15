"""
MMRCCNet — Multi-Modal Renal Cell Carcinoma Network
Main entry point for training, ablation, and explainability.

Usage:
  python main.py --mode train --manifest seer_rcc_2010_2018_clean.csv --no_imaging
  python main.py --mode train --manifest data/manifests/multimodal_manifest.json --use_genomics
  python main.py --mode ablation --manifest data/manifests/multimodal_manifest.json
  python main.py --mode explain --checkpoint runs/full/best_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG
from data.datasets import MultimodalDataset, SEERDataset, load_seer_splits
from models.mmrccnet import MMRCCNet
from utils.explainability import generate_gradcam_imaging, generate_shap_clinical
from utils.metrics import binary_metrics
from utils.stats_validation import cv_summary, delong_test, save_validation_report, stratified_kfold_indices
from utils.training import evaluate, train_model

ABLATION_CONFIGS = {
    "clinical_only":           dict(use_clinical=True,  use_radiomics=False, use_imaging=False, use_genomics=False),
    "clinical_radiomics":      dict(use_clinical=True,  use_radiomics=True,  use_imaging=False, use_genomics=False),
    "clinical_imaging":        dict(use_clinical=True,  use_radiomics=False, use_imaging=True,  use_genomics=False),
    "clinical_genomics":       dict(use_clinical=True,  use_radiomics=False, use_imaging=False, use_genomics=True),
    "clinical_rad_img":        dict(use_clinical=True,  use_radiomics=True,  use_imaging=True,  use_genomics=False),
    "clinical_rad_gen":        dict(use_clinical=True,  use_radiomics=True,  use_imaging=False, use_genomics=True),
    "clinical_img_gen":        dict(use_clinical=True,  use_radiomics=False, use_imaging=True,  use_genomics=True),
    "full":                    dict(use_clinical=True,  use_radiomics=True,  use_imaging=True,  use_genomics=True),
}


def _is_seer_csv(path: str) -> bool:
    return path.endswith(".csv")


from imblearn.combine import SMOTETomek
import pandas as pd

def _build_seer_loaders(manifest: str, batch_size: int):
    splits = load_seer_splits(
        manifest, CONFIG["clinical_features"], CONFIG["label_col"],
        CONFIG["val_ratio"], CONFIG["test_ratio"], CONFIG["seed"],
    )
    common = dict(
        df=splits["df"],
        clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"],
        site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"],
        event_col=CONFIG["event_col"],
        scaler=splits["scaler"],
    )
    # Original indices
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]
    # Apply SMOTETomek to training set
    train_df = splits["df"].iloc[train_idx].reset_index(drop=True)
    # Features for resampling: clinical + site + survival + event
    resample_features = CONFIG["clinical_features"] + CONFIG["site_cols"] + [CONFIG["survival_col"], CONFIG["event_col"]]
    X = train_df[resample_features].values
    y = train_df[CONFIG["label_col"]].values
    smt = SMOTETomek(random_state=CONFIG["seed"])
    X_bal, y_bal = smt.fit_resample(X, y)
    # Recreate balanced dataframe
    df_bal = pd.DataFrame(X_bal, columns=resample_features)
    df_bal[CONFIG["label_col"]] = y_bal
    if "patient_id" in splits["df"].columns:
        df_bal["patient_id"] = range(len(df_bal))
    # New training dataset using balanced dataframe
    train_ds = SEERDataset(
        df=df_bal,
        indices=np.arange(len(df_bal)),
        clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"],
        site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"],
        event_col=CONFIG["event_col"],
        scaler=splits["scaler"],
    )
    val_ds = SEERDataset(
        df=splits["df"],
        indices=val_idx,
        clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"],
        site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"],
        event_col=CONFIG["event_col"],
        scaler=splits["scaler"],
    )
    test_ds = SEERDataset(
        df=splits["df"],
        indices=test_idx,
        clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"],
        site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"],
        event_col=CONFIG["event_col"],
        scaler=splits["scaler"],
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size),
        "test": DataLoader(test_ds, batch_size=batch_size),
    }
    return loaders, splits


def _build_multimodal_loaders(manifest: str, batch_size: int, clinical_scaler):
    radiomics_features = []
    rad_json = Path(CONFIG["radiomics_dir"]) / "lasso_selected.json"
    if rad_json.exists():
        radiomics_features = json.loads(rad_json.read_text())["selected_features"]
    if not radiomics_features:
        radiomics_features = [f"radiomics_{i}" for i in range(CONFIG["radiomics_top_k"])]

    full_ds = MultimodalDataset(
        manifest, clinical_scaler, CONFIG["clinical_features"],
        radiomics_features, CONFIG["genomics_dim"], CONFIG["imaging_size"], ROOT,
    )
    n = len(full_ds)
    idx = np.arange(n)
    rng = np.random.default_rng(CONFIG["seed"])
    rng.shuffle(idx)
    n_test = int(n * CONFIG["test_ratio"])
    n_val = int(n * CONFIG["val_ratio"])
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    loaders = {
        "train": DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True),
        "val": DataLoader(Subset(full_ds, val_idx), batch_size=batch_size),
        "test": DataLoader(Subset(full_ds, test_idx), batch_size=batch_size),
    }
    return loaders, {"radiomics_dim": len(radiomics_features)}


def _make_model(flags: dict, radiomics_dim: int = 40) -> MMRCCNet:
    return MMRCCNet(
        clinical_dim=len(CONFIG["clinical_features"]),
        radiomics_dim=radiomics_dim,
        genomics_dim=CONFIG["genomics_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["dropout"],
        **flags,
    )


def mode_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if _is_seer_csv(args.manifest):
        print("=" * 60)
        print("PHASE 2 — MMRCCNet Clinical-Only Training")
        print("=" * 60)
        loaders, _ = _build_seer_loaders(args.manifest, CONFIG["batch_size"])
        flags = ABLATION_CONFIGS["clinical_only"]
        radiomics_dim = CONFIG["radiomics_top_k"]
        out_dir = Path(CONFIG["runs_dir"]) / "clinical_only"
    else:
        print("=" * 60)
        print("PHASE 5 — Full MMRCCNet Training")
        print("=" * 60)
        seer_splits = load_seer_splits(
            CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"],
            CONFIG["val_ratio"], CONFIG["test_ratio"], CONFIG["seed"],
        )
        loaders, meta = _build_multimodal_loaders(args.manifest, CONFIG["batch_size"], seer_splits["scaler"])
        if args.no_imaging and not args.use_genomics:
            flags = ABLATION_CONFIGS["clinical_only"]
        elif args.use_genomics and not args.no_imaging:
            flags = ABLATION_CONFIGS["full"]
        elif args.use_genomics:
            flags = ABLATION_CONFIGS["clinical_genomics"]
        elif args.no_imaging:
            flags = ABLATION_CONFIGS["clinical_radiomics"]
        else:
            flags = ABLATION_CONFIGS["full"]
        radiomics_dim = meta["radiomics_dim"]
        out_dir = Path(CONFIG["runs_dir"]) / ("clinical_only" if flags == ABLATION_CONFIGS["clinical_only"] else "full")

    model = _make_model(flags, radiomics_dim)
    print(f"Active branches: {[k for k, v in flags.items() if v and k.startswith('use_')]}")

    result = train_model(
        model, loaders["train"], loaders["val"], CONFIG, device, out_dir,
        flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"],
    )

    # Load best and evaluate on test
    ckpt = torch.load(out_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(
        model, loaders["test"], device,
        flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"],
    )
    print("\nTest set results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report = {"best_val_auc": result["best_auc"], "test_metrics": test_metrics, "flags": flags}
    save_validation_report(report, out_dir / "final_results.json")

    # PHASE 8 — compare with baseline via DeLong if available
    baseline_path = Path(CONFIG["runs_dir"]) / "baseline" / "baseline_results.json"
    if baseline_path.exists() and _is_seer_csv(args.manifest):
        _run_statistical_validation(model, loaders, device, flags, out_dir)

    return report


def _run_statistical_validation(model, loaders, device, flags, out_dir):
    """PHASE 8 — DeLong test + 5-fold CV for clinical SEER."""
    from scripts.train_lightgbm_baseline import train_and_evaluate
    from data.datasets import load_seer_dataframe

    print("\n" + "=" * 60)
    print("PHASE 8 — Statistical Validation")
    print("=" * 60)

    df = load_seer_dataframe(CONFIG["seer_csv"])
    y = df[CONFIG["label_col"]].values

    # 5-fold CV for MMRCCNet
    fold_aucs = []
    for fold, (tr, te) in enumerate(stratified_kfold_indices(y, CONFIG["n_folds"], CONFIG["seed"])):
        splits = load_seer_splits(CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"])
        common = dict(
            df=df, clinical_features=CONFIG["clinical_features"], label_col=CONFIG["label_col"],
            site_cols=CONFIG["site_cols"], survival_col=CONFIG["survival_col"],
            event_col=CONFIG["event_col"], scaler=splits["scaler"],
        )
        train_ds = SEERDataset(indices=tr, **common)
        val_ds = SEERDataset(indices=te, **common)
        m = _make_model(flags)
        mini_cfg = {**CONFIG, "epochs": 30, "patience": 8}
        train_model(
            m, DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True),
            DataLoader(val_ds, batch_size=CONFIG["batch_size"]),
            mini_cfg, device, out_dir / f"cv_fold_{fold}",
            flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"],
        )
        ckpt = torch.load(out_dir / f"cv_fold_{fold}" / "best_checkpoint.pt", map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model"])
        met = evaluate(m, DataLoader(val_ds, batch_size=CONFIG["batch_size"]), device,
                       flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"])
        fold_aucs.append(met["auc"])
        print(f"  MMRCCNet fold {fold+1}: AUC = {met['auc']:.4f}")

    mmrcc_cv = cv_summary(fold_aucs)
    print(f"MMRCCNet CV: {mmrcc_cv['mean']:.4f} ± {mmrcc_cv['std']:.4f}")

    # DeLong: MMRCCNet vs LightGBM on same test split
    splits = load_seer_splits(CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"])
    test_ds = SEERDataset(
        df=df, indices=splits["test_idx"], clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"], site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"], event_col=CONFIG["event_col"], scaler=splits["scaler"],
    )
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])
    ckpt = torch.load(out_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model = _make_model(flags)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    mmrcc_probs, y_true = [], []
    for batch in test_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model(clinical=batch["clinical"])
            mmrcc_probs.append(torch.sigmoid(out["metastasis_logit"]).cpu().numpy())
            y_true.append(batch["metastasis"].cpu().numpy())
    mmrcc_probs = np.concatenate(mmrcc_probs)
    y_true = np.concatenate(y_true)

    lgb_result = train_and_evaluate(df, splits["train_idx"], splits["test_idx"])
    delong = delong_test(y_true, mmrcc_probs, lgb_result["prob"])

    validation = {
        "mmrcc_cv": mmrcc_cv,
        "lightgbm_cv": json.loads((Path(CONFIG["runs_dir"]) / "baseline" / "baseline_results.json").read_text()),
        "delong_test": delong,
        "significant_at_0.05": delong["p_value"] < 0.05,
    }
    save_validation_report(validation, out_dir / "statistical_validation.json")
    print(f"DeLong test: AUC MMRCCNet={delong['auc_a']:.4f} vs LightGBM={delong['auc_b']:.4f}, p={delong['p_value']:.4e}")
    print(f"Statistically significant (p<0.05): {validation['significant_at_0.05']}")


def mode_ablation(args):
    """PHASE 6 — 8 ablation combinations."""
    print("=" * 60)
    print("PHASE 6 — Ablation Study (8 combinations)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seer_splits = load_seer_splits(
        CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"],
    )
    loaders, meta = _build_multimodal_loaders(args.manifest, CONFIG["batch_size"], seer_splits["scaler"])
    out_dir = Path(args.output or Path(CONFIG["runs_dir"]) / "ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    mini_cfg = {**CONFIG, "epochs": 40, "patience": 10}
    results = []

    for name, flags in ABLATION_CONFIGS.items():
        print(f"\n--- Ablation: {name} ---")
        model = _make_model(flags, meta["radiomics_dim"])
        run_dir = out_dir / name
        train_model(
            model, loaders["train"], loaders["val"], mini_cfg, device, run_dir,
            flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"],
        )
        ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_m = evaluate(
            model, loaders["test"], device,
            flags["use_imaging"], flags["use_radiomics"], flags["use_clinical"], flags["use_genomics"],
        )
        entry = {"config": name, **flags, **test_m}
        results.append(entry)
        print(f"  Test AUC: {test_m['auc']:.4f} | mean_site_auc: {test_m.get('mean_site_auc', 0):.4f} | c_index: {test_m.get('c_index', 0):.4f}")

    save_validation_report({"ablation": results}, out_dir / "ablation_results.json")

    # LaTeX table
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Configuration & Met AUC & Site AUC & C-index & Modalities \\\\", "\\midrule"]
    for r in results:
        mods = "+".join(m for m, k in [("C", "use_clinical"), ("R", "use_radiomics"),
                                         ("I", "use_imaging"), ("G", "use_genomics")] if r.get(k))
        lines.append(
            f"{r['config']} & {r['auc']:.3f} & {r.get('mean_site_auc', 0):.3f} & "
            f"{r.get('c_index', 0):.3f} & {mods} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAblation results saved to {out_dir}")
    return results


def mode_explain(args):
    """PHASE 7 — SHAP + Grad-CAM."""
    print("=" * 60)
    print("PHASE 7 — Explainability (SHAP + Grad-CAM)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    flags = ckpt.get("config", {}).get("flags", ABLATION_CONFIGS["clinical_only"])
    if not isinstance(flags, dict) or "use_clinical" not in flags:
        flags = ABLATION_CONFIGS["full"]

    model = _make_model(flags)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    out_dir = Path(args.output or Path(CONFIG["runs_dir"]) / "explainability")
    splits = load_seer_splits(CONFIG["seer_csv"], CONFIG["clinical_features"], CONFIG["label_col"])
    common = dict(
        df=splits["df"], clinical_features=CONFIG["clinical_features"],
        label_col=CONFIG["label_col"], site_cols=CONFIG["site_cols"],
        survival_col=CONFIG["survival_col"], event_col=CONFIG["event_col"],
        scaler=splits["scaler"],
    )
    test_ds = SEERDataset(indices=splits["test_idx"], **common)
    X = test_ds.X
    shap_path = generate_shap_clinical(
        model, X[:200], X[200:300], CONFIG["clinical_features"], out_dir, device,
    )
    print(f"SHAP plots saved: {shap_path}")

    if flags.get("use_imaging"):
        sample = test_ds[0]
        if "imaging" in sample:
            gc_path = generate_gradcam_imaging(model, sample["imaging"], out_dir, device)
            print(f"Grad-CAM saved: {gc_path}")
    else:
        # Demo Grad-CAM with imaging-only synthetic input
        model_img = _make_model({
            "use_clinical": False, "use_radiomics": False,
            "use_imaging": True, "use_genomics": False,
        })
        ckpt_full = Path(CONFIG["runs_dir"]) / "full" / "best_checkpoint.pt"
        if ckpt_full.exists():
            c = torch.load(ckpt_full, map_location=device, weights_only=False)
            try:
                model_img.load_state_dict(c["model"], strict=False)
            except Exception:
                pass
        model_img.to(device)
        dummy = torch.randn(64, 64)
        gc_path = generate_gradcam_imaging(model_img, dummy, out_dir, device)
        print(f"Grad-CAM (demo) saved: {gc_path}")

    print(f"Explainability outputs in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="MMRCCNet pipeline")
    parser.add_argument("--mode", choices=["train", "ablation", "explain"], default="train")
    parser.add_argument("--manifest", default=CONFIG["seer_csv"])
    parser.add_argument("--no_imaging", action="store_true")
    parser.add_argument("--use_genomics", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "ablation":
        mode_ablation(args)
    elif args.mode == "explain":
        if not args.checkpoint:
            args.checkpoint = str(Path(CONFIG["runs_dir"]) / "clinical_only" / "best_checkpoint.pt")
        mode_explain(args)


if __name__ == "__main__":
    main()
