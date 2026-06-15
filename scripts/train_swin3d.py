"""
train_swin3d.py
===============
End-to-end training script for the Fused 3D Swin Transformer model.

Architecture:
  CT       -> 3D Swin Transformer  -> 768-dim imaging features
  RNA-seq  -> Transformer Encoder  -> 768-dim genomic features
  Clinical -> Linear Projection    ->  50-dim clinical features
                  v
           Cross-Attention Fusion -> 512-dim
                  v
           Shared Dense Layers   -> 256-dim
                  v
     +-------------+--------------+--------------+
  Metastasis    Survival      Site Mets     Clinical
  (AUC target)  (C-index)     (4 sites)     Decision

Run:
  python scripts/train_swin3d.py
  python scripts/train_swin3d.py --epochs 80 --lr 5e-4 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG
from models.swin3d_fusion import FusedSwin3DNet
from utils.metrics import binary_metrics, concordance_index, multi_site_auc

# -------------------------------------------------------------
# Args
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--n_folds",     type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out_dir",     type=str,   default=str(ROOT / "runs" / "swin3d_fusion"))
    p.add_argument("--no_imaging",  action="store_true",
                   help="Skip CT imaging (use tabular+genomics only; much faster on CPU)")
    return p.parse_args()


# -------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------

def load_seer_data():
    """Load SEER clinical CSV and return cleaned DataFrame."""
    csv = Path(CONFIG["seer_csv"])
    df  = pd.read_csv(csv, low_memory=False)

    # Required columns
    feat_cols  = CONFIG["clinical_features"]
    label_col  = CONFIG["label_col"]
    site_cols  = CONFIG["site_cols"]
    surv_col   = CONFIG["survival_col"]
    event_col  = CONFIG["event_col"]

    all_cols = feat_cols + site_cols + [surv_col, event_col, label_col]
    missing  = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in SEER CSV: {missing}")

    df = df[all_cols].dropna().reset_index(drop=True)
    return df, feat_cols, label_col, site_cols, surv_col, event_col


def load_genomics_data(n_samples: int, genomics_dim: int, seed: int):
    """
    Load real genomics .npy files if available, otherwise generate
    realistic synthetic RNA-seq data (log-normal distribution).
    """
    gen_dir = Path(CONFIG["genomics_dir"])
    npy_files = sorted(gen_dir.glob("*.npy"))

    if len(npy_files) >= n_samples:
        print(f"  Loading {n_samples} real genomics files from {gen_dir}")
        arrays = [np.load(f)[:genomics_dim] for f in npy_files[:n_samples]]
        # Pad if short
        arrays = [
            np.pad(a, (0, max(0, genomics_dim - len(a))), mode="constant")[:genomics_dim]
            for a in arrays
        ]
        return np.stack(arrays).astype(np.float32)
    else:
        print(f"  [INFO] {len(npy_files)} genomics files found (need {n_samples}).")
        print(f"  Generating synthetic RNA-seq features ({genomics_dim}-dim log-normal).")
        rng = np.random.default_rng(seed)
        # Log-normal mimics RNA-seq count distribution
        G = rng.lognormal(mean=0.0, sigma=1.5, size=(n_samples, genomics_dim)).astype(np.float32)
        # Standardize (mimics DESeq2 normalised counts)
        G = (G - G.mean(axis=0, keepdims=True)) / (G.std(axis=0, keepdims=True) + 1e-8)
        return G


def smote_oversample(X: np.ndarray, y: np.ndarray, seed: int):
    """SMOTE oversampling for minority class (training only)."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=seed, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE: {y.sum():.0f} -> {y_res.sum():.0f} positives "
              f"({len(y)} -> {len(y_res)} total samples)")
        return X_res, y_res
    except Exception as e:
        print(f"  SMOTE skipped ({e}). Using class-weighted sampling instead.")
        return X, y


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Weighted random sampler for imbalanced metastasis labels."""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    sample_weights = weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_tensors(X_clin, X_gen, y_met, y_sites, y_surv, y_event):
    """Pack numpy arrays into tensors."""
    return (
        torch.tensor(X_clin,   dtype=torch.float32),
        torch.tensor(X_gen,    dtype=torch.float32),
        torch.tensor(y_met,    dtype=torch.float32),
        torch.tensor(y_sites,  dtype=torch.float32),
        torch.tensor(y_surv,   dtype=torch.float32),
        torch.tensor(y_event,  dtype=torch.float32),
    )


# -------------------------------------------------------------
# Loss
# -------------------------------------------------------------

class FusedLoss(nn.Module):
    def __init__(self, pos_weight: float = 15.67, site_w: float = 0.5,
                 surv_w: float = 0.3):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.met_loss  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.site_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.surv_loss = nn.MSELoss()
        self.site_w    = site_w
        self.surv_w    = surv_w

    def forward(self, out: dict, met: torch.Tensor,
                sites: torch.Tensor, surv: torch.Tensor) -> torch.Tensor:
        l_met  = self.met_loss(out["metastasis_logit"], met)
        l_site = self.site_loss(out["site_logits"], sites)
        l_surv = self.surv_loss(out["survival_risk"], surv / 100.0)
        return l_met + self.site_w * l_site + self.surv_w * l_surv


# -------------------------------------------------------------
# Train / Eval loops
# -------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler_amp, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        clin, gen, met, sites, surv, _ = [b.to(device) for b in batch]
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out  = model(clinical=clin, genomics=gen)
            loss = criterion(out, met, sites, surv)
        if device.type == "cuda":
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    met_probs, met_true   = [], []
    site_probs, site_true = [], []
    risks, survs, events  = [], [], []
    for batch in loader:
        clin, gen, met, sites, surv, evt = [b.to(device) for b in batch]
        out = model(clinical=clin, genomics=gen)
        met_probs.append(torch.sigmoid(out["metastasis_logit"]).cpu().numpy())
        met_true.append(met.cpu().numpy())
        site_probs.append(torch.sigmoid(out["site_logits"]).cpu().numpy())
        site_true.append(sites.cpu().numpy())
        risks.append(out["survival_risk"].cpu().numpy())
        survs.append(surv.cpu().numpy())
        events.append(evt.cpu().numpy())

    met_probs  = np.concatenate(met_probs)
    met_true   = np.concatenate(met_true)
    site_probs = np.concatenate(site_probs)
    site_true  = np.concatenate(site_true)
    risks      = np.concatenate(risks)
    survs      = np.concatenate(survs)
    events     = np.concatenate(events)

    m = binary_metrics(met_true, met_probs)
    m.update(multi_site_auc(site_true, site_probs))
    m["c_index"] = concordance_index(survs, events, risks)
    return m, met_probs, met_true


# -------------------------------------------------------------
# Main training
# -------------------------------------------------------------

def train_fold(fold_idx, train_clin, train_gen, train_met, train_sites,
               train_surv, train_event,
               val_clin, val_gen, val_met, val_sites, val_surv, val_event,
               cfg: dict, device: torch.device, out_dir: Path):
    """Train one fold; return best val metrics and model path."""

    # SMOTE on training split
    X_all = np.concatenate([train_clin, train_gen], axis=1)
    X_bal, y_bal = smote_oversample(X_all, train_met, cfg["seed"])
    c_dim = train_clin.shape[1]
    train_clin_b = X_bal[:, :c_dim].astype(np.float32)
    train_gen_b  = X_bal[:, c_dim:].astype(np.float32)

    # Rebuild sites/surv/event for SMOTE-expanded set (tile original)
    n_orig = len(train_met)
    n_new  = len(y_bal)
    repeat = int(np.ceil(n_new / n_orig))
    sites_b = np.tile(train_sites, (repeat, 1))[:n_new].astype(np.float32)
    surv_b  = np.tile(train_surv,  (repeat,))[:n_new].astype(np.float32)
    event_b = np.tile(train_event, (repeat,))[:n_new].astype(np.float32)

    train_tensors = build_tensors(train_clin_b, train_gen_b, y_bal.astype(np.float32),
                                  sites_b, surv_b, event_b)
    val_tensors   = build_tensors(val_clin, val_gen, val_met, val_sites,
                                  val_surv, val_event)

    sampler      = make_weighted_sampler(y_bal)
    train_ds     = TensorDataset(*train_tensors)
    val_ds       = TensorDataset(*val_tensors)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2)

    model = FusedSwin3DNet(cfg).to(device)
    criterion = FusedLoss(pos_weight=CONFIG["pos_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = 0.0
    patience  = 0
    ckpt_path = out_dir / f"fold{fold_idx}_best.pt"
    history   = []

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, criterion,
                               optimizer, scaler_amp, device)
        val_m, _, _ = evaluate_loader(model, val_loader, device)
        scheduler.step()

        history.append({"epoch": epoch, "loss": loss, **val_m})

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "cfg": cfg, "metrics": val_m}, ckpt_path)
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Fold {fold_idx} Ep {epoch:03d}: "
                  f"loss={loss:.4f}  val_AUC={val_m['auc']:.4f}  "
                  f"val_AP={val_m['ap']:.4f}  "
                  f"best={best_auc:.4f}")

        if patience >= cfg["patience"]:
            print(f"    Early stopping at epoch {epoch}")
            break

    return best_auc, ckpt_path, history


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  Fused 3D Swin Transformer – Multimodal RCC Training")
    print(f"{'='*60}")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Output   : {out_dir}")
    print()

    # -- Load data ----------------------------------------------
    print("[1/5] Loading SEER clinical data …")
    df, feat_cols, label_col, site_cols, surv_col, event_col = load_seer_data()
    print(f"      {len(df):,} patients | positives: {df[label_col].sum():,} "
          f"({df[label_col].mean()*100:.1f}%)")

    scaler = StandardScaler()
    X_clin = scaler.fit_transform(df[feat_cols].values).astype(np.float32)
    y_met  = df[label_col].values.astype(np.float32)
    y_sites= df[site_cols].values.astype(np.float32)
    y_surv = df[surv_col].values.astype(np.float32)
    y_evt  = df[event_col].values.astype(np.float32)

    # -- Genomics ------------------------------------------------
    gen_dim = CONFIG["genomics_dim"]
    print(f"\n[2/5] Loading genomics data ({gen_dim}-dim) …")
    X_gen = load_genomics_data(len(df), gen_dim, args.seed)

    # -- Build model config --------------------------------------
    cfg = {
        "use_imaging":        False,          # CT imaging off for tabular run
        "use_genomics":       True,
        "use_clinical":       True,
        "clinical_input_dim": len(feat_cols),
        "genomics_input_dim": gen_dim,
        "genomics_hidden_dim":128,
        "genomics_heads":     8,
        "genomics_layers":    4,
        "swin_patch_size":    4,
        "swin_embed_dim":     96,
        "swin_depths":        [2, 2, 2],
        "swin_heads":         [3, 6, 12],
        "fused_dim":          512,
        "fusion_heads":       8,
        "shared_hidden":      args.hidden_dim,
        "dropout":            args.dropout,
        "lr":                 args.lr,
        "weight_decay":       args.weight_decay,
        "batch_size":         args.batch_size,
        "epochs":             args.epochs,
        "patience":           args.patience,
        "seed":               args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # -- K-fold CV -----------------------------------------------
    print(f"\n[3/5] {args.n_folds}-Fold Stratified Cross-Validation …")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_aucs = []
    all_histories = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_clin, y_met), 1):
        print(f"\n  -- Fold {fold}/{args.n_folds} --")
        best_auc, ckpt, hist = train_fold(
            fold_idx=fold,
            train_clin=X_clin[tr_idx], train_gen=X_gen[tr_idx],
            train_met=y_met[tr_idx],   train_sites=y_sites[tr_idx],
            train_surv=y_surv[tr_idx], train_event=y_evt[tr_idx],
            val_clin=X_clin[va_idx],   val_gen=X_gen[va_idx],
            val_met=y_met[va_idx],     val_sites=y_sites[va_idx],
            val_surv=y_surv[va_idx],   val_event=y_evt[va_idx],
            cfg=cfg, device=device, out_dir=out_dir,
        )
        fold_aucs.append(best_auc)
        all_histories.append(hist)
        print(f"  Fold {fold} best AUC: {best_auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # -- Final held-out test -------------------------------------
    print(f"\n[4/5] Final test evaluation (held-out 15% split) …")
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(df) * CONFIG["test_ratio"])
    n_val  = int(len(df) * CONFIG["val_ratio"])
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    # Retrain on full train+val with best fold config
    print("  Retraining on full train split for final model …")
    full_train_clin = X_clin[train_idx]
    full_train_gen  = X_gen[train_idx]
    full_train_met  = y_met[train_idx]
    full_train_sites= y_sites[train_idx]
    full_train_surv = y_surv[train_idx]
    full_train_evt  = y_evt[train_idx]

    best_fold_auc, final_ckpt, final_hist = train_fold(
        fold_idx="final",
        train_clin=full_train_clin, train_gen=full_train_gen,
        train_met=full_train_met,   train_sites=full_train_sites,
        train_surv=full_train_surv, train_event=full_train_evt,
        val_clin=X_clin[val_idx],   val_gen=X_gen[val_idx],
        val_met=y_met[val_idx],     val_sites=y_sites[val_idx],
        val_surv=y_surv[val_idx],   val_event=y_evt[val_idx],
        cfg=cfg, device=device, out_dir=out_dir,
    )

    # Load best checkpoint and evaluate on test set
    ckpt_data  = torch.load(final_ckpt, map_location=device, weights_only=False)
    final_model = FusedSwin3DNet(cfg).to(device)
    final_model.load_state_dict(ckpt_data["state_dict"])

    test_tensors = build_tensors(
        X_clin[test_idx], X_gen[test_idx],
        y_met[test_idx],  y_sites[test_idx],
        y_surv[test_idx], y_evt[test_idx],
    )
    test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=256)
    test_m, test_probs, test_true = evaluate_loader(final_model, test_loader, device)

    print(f"\n{'='*60}")
    print("  FINAL TEST RESULTS – Fused Swin3D Multimodal Network")
    print(f"{'='*60}")
    print(f"  AUC (ROC)    : {test_m['auc']:.4f}")
    print(f"  AUPRC        : {test_m['ap']:.4f}")
    print(f"  F1 Score     : {test_m['f1']:.4f}")
    print(f"  Lung AUC     : {test_m.get('lung_auc', 0):.4f}")
    print(f"  Bone AUC     : {test_m.get('bone_auc', 0):.4f}")
    print(f"  Liver AUC    : {test_m.get('liver_auc', 0):.4f}")
    print(f"  Brain AUC    : {test_m.get('brain_auc', 0):.4f}")
    print(f"  Mean Site AUC: {test_m.get('mean_site_auc', 0):.4f}")
    print(f"  C-Index      : {test_m['c_index']:.4f}")
    print(f"  CV AUC       : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'='*60}")

    # -- DeLong comparison vs LightGBM baseline ------------------
    print("\n[5/5] Statistical validation …")
    try:
        from utils.stats_validation import delong_test
        # Load prior LightGBM scores if available
        lgbm_path = Path(CONFIG["runs_dir"]) / "clinical_only" / "lgbm_probs.npy"
        if lgbm_path.exists():
            lgbm_probs = np.load(str(lgbm_path))[:len(test_probs)]
            delong_auc, delong_p = delong_test(test_true, test_probs, lgbm_probs)
            print(f"  DeLong AUC: swin={test_m['auc']:.4f} vs lgbm={delong_auc:.4f}  p={delong_p:.4e}")
        else:
            print("  LightGBM probs not found – skipping DeLong test.")
    except Exception as e:
        print(f"  DeLong test skipped: {e}")

    # -- Save results ---------------------------------------------
    results = {
        "model": "FusedSwin3DNet",
        "cv_auc_mean": mean_auc,
        "cv_auc_std":  std_auc,
        "fold_aucs":   fold_aucs,
        "test_metrics": test_m,
        "best_val_auc": best_fold_auc,
        "config": cfg,
    }
    (out_dir / "final_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history_final.json").write_text(json.dumps(final_hist, indent=2))
    np.save(str(out_dir / "test_probs.npy"), test_probs)
    np.save(str(out_dir / "test_true.npy"),  test_true)

    print(f"\n  Results saved to: {out_dir}")
    print("  Done! ✓")
    return results


if __name__ == "__main__":
    main()
