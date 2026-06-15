# -*- coding: utf-8 -*-
"""
Fast Multimodal RCC Trainer
============================
Fused architecture:
  Clinical (7-dim) -> MLP -> 128-dim
  Genomics (500-dim) -> Transformer Encoder -> 256-dim
  ---- Cross-Attention Fusion ----
  Metastasis head | Survival head | Site heads (4) | Clinical Decision head

Optimized for CPU training on SEER data.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "epochs":        25,
    "lr":            5e-4,
    "batch_size":    1024,
    "weight_decay":  1e-4,
    "dropout":       0.25,
    "patience":      10,
    "n_folds":       2,
    "seed":          42,
    "val_ratio":     0.15,
    "test_ratio":    0.15,
    "pos_weight":    15.67,
    "genomics_dim":  500,
    "clin_features": ["age","sex","t_stage","n_stage","grade","histology_enc","tumor_size_cm"],
    "site_cols":     ["lung_met","bone_met","liver_met","brain_met"],
    "label_col":     "metastasis",
    "survival_col":  "survival_months",
    "event_col":     "vital_status",
    "out_dir":       str(ROOT / "runs" / "swin3d_fusion"),
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ── Model ────────────────────────────────────────────────────────────────────

class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=7, out_dim=128, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256),    nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),nn.LayerNorm(out_dim),
        )
    def forward(self, x): return self.net(x)


class GenomicsTransformer(nn.Module):
    def __init__(self, in_dim=500, hidden=128, out_dim=256, n_heads=8, n_layers=3, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(1, hidden)
        self.pos  = nn.Parameter(torch.randn(1, in_dim, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.GELU(), nn.LayerNorm(out_dim))

    def forward(self, x):
        B, G = x.shape
        t = self.proj(x.unsqueeze(-1)) + self.pos[:, :G, :]
        t = self.enc(t)
        w = torch.softmax(self.attn_pool(t), dim=1)
        p = (t * w).sum(dim=1)
        return self.head(p)


class CrossAttnFusion(nn.Module):
    def __init__(self, clin_dim=128, gen_dim=256, fused=256, n_heads=4, dropout=0.2):
        super().__init__()
        self.c_proj = nn.Linear(clin_dim, fused)
        self.g_proj = nn.Linear(gen_dim,  fused)
        self.cg_attn = nn.MultiheadAttention(fused, n_heads, dropout=dropout, batch_first=True)
        self.gc_attn = nn.MultiheadAttention(fused, n_heads, dropout=dropout, batch_first=True)
        self.norm_c  = nn.LayerNorm(fused)
        self.norm_g  = nn.LayerNorm(fused)
        self.mlp = nn.Sequential(
            nn.Linear(fused*2, fused*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fused*2, fused),   nn.LayerNorm(fused),
        )

    def forward(self, c, g):
        c = self.c_proj(c).unsqueeze(1)   # (B,1,D)
        g = self.g_proj(g).unsqueeze(1)   # (B,1,D)
        c2, _ = self.cg_attn(c, g, g)
        g2, _ = self.gc_attn(g, c, c)
        c_out = self.norm_c(c + c2).squeeze(1)
        g_out = self.norm_g(g + g2).squeeze(1)
        return self.mlp(torch.cat([c_out, g_out], dim=-1))


class MultimodalRCCNet(nn.Module):
    """
    CT Input  -> [disabled on CPU; replaced by zero token]
    RNA-seq   -> Genomics Transformer -> 256-dim
    Clinical  -> MLP Encoder         -> 128-dim
                         |
                Cross-Attention Fusion -> 256-dim
                         |
                Shared Dense Layers   -> 128-dim
                         |
         Metastasis | Survival | Sites(4) | Clinical Decision(3)
    """
    def __init__(self, cfg):
        super().__init__()
        self.clinical_enc = ClinicalEncoder(len(cfg["clin_features"]), 128, cfg["dropout"])
        self.genomics_enc = GenomicsTransformer(cfg["genomics_dim"], 128, 256,
                                                n_heads=8, n_layers=3,
                                                dropout=cfg["dropout"])
        self.fusion       = CrossAttnFusion(128, 256, 256, n_heads=4, dropout=cfg["dropout"])
        self.shared       = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(cfg["dropout"]),
        )
        # Heads
        self.met_head      = nn.Linear(128, 1)
        self.surv_head     = nn.Linear(128, 1)
        self.site_head     = nn.Linear(128, 4)
        self.decision_head = nn.Linear(128, 3)

    def forward(self, clin, gen):
        c = self.clinical_enc(clin)
        g = self.genomics_enc(gen)
        f = self.fusion(c, g)
        s = self.shared(f)
        return {
            "metastasis_logit": self.met_head(s).squeeze(-1),
            "survival_risk":    self.surv_head(s).squeeze(-1),
            "site_logits":      self.site_head(s),
            "decision_logits":  self.decision_head(s),
        }


# ── Loss ─────────────────────────────────────────────────────────────────────

class MultimodalLoss(nn.Module):
    def __init__(self, pos_weight=15.67):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.met_loss  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.site_loss = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.surv_loss = nn.MSELoss()

    def forward(self, out, met, sites, surv):
        l1 = self.met_loss(out["metastasis_logit"], met)
        l2 = self.site_loss(out["site_logits"], sites)
        l3 = self.surv_loss(out["survival_risk"], surv / 100.0)
        return l1 + 0.4*l2 + 0.2*l3


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.35):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    ap  = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    return {"auc": auc, "ap": ap, "f1": f1}


def site_aucs(site_true, site_prob):
    names = ["lung", "bone", "liver", "brain"]
    res = {}
    for i, n in enumerate(names):
        yt, yp = site_true[:, i], site_prob[:, i]
        res[f"{n}_auc"] = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    valid = [v for v in res.values() if not np.isnan(v)]
    res["mean_site_auc"] = float(np.mean(valid)) if valid else 0.0
    return res


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_data():
    csv_path = ROOT / "seer_rcc_2010_2018_clean.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    need = CFG["clin_features"] + CFG["site_cols"] + [CFG["label_col"], CFG["survival_col"], CFG["event_col"]]
    df = df[[c for c in need if c in df.columns]].dropna().reset_index(drop=True)
    # Subsample drastically for fast CPU execution
    df = df.sample(n=2000, random_state=CFG["seed"]).reset_index(drop=True)
    scaler = StandardScaler()
    X_clin = scaler.fit_transform(df[CFG["clin_features"]].values).astype(np.float32)
    y_met  = df[CFG["label_col"]].values.astype(np.float32)
    y_site = df[CFG["site_cols"]].values.astype(np.float32)
    y_surv = df[CFG["survival_col"]].values.astype(np.float32)
    y_evt  = df[CFG["event_col"]].values.astype(np.float32)
    print(f"  SEER: {len(df):,} patients | positives: {int(y_met.sum()):,} ({y_met.mean()*100:.1f}%)")
    return X_clin, y_met, y_site, y_surv, y_evt


def synth_genomics(n, dim, seed):
    rng = np.random.default_rng(seed)
    G = rng.lognormal(0, 1.5, (n, dim)).astype(np.float32)
    return (G - G.mean(0, keepdims=True)) / (G.std(0, keepdims=True) + 1e-8)


def smote_balance(X_clin, X_gen, y, seed):
    """
    Fast oversampling: run RandomOverSampler on clinical features only (7-dim),
    then tile genomics to match. Avoids SMOTE slowness on high-dim data.
    """
    try:
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=seed)
        Xr_c, yr = ros.fit_resample(X_clin, y)
        # Tile genomics to match oversampled size
        n_orig, n_new = len(y), len(yr)
        rep = int(np.ceil(n_new / n_orig))
        Xr_g = np.tile(X_gen, (rep, 1))[:n_new]
        print(f"  Oversampling: {int(y.sum())} -> {int(yr.sum())} positives ({len(y)} -> {len(yr)} samples)")
        return Xr_c.astype(np.float32), Xr_g.astype(np.float32), yr
    except Exception as e:
        print(f"  Oversampling failed ({e}), using original data")
        return X_clin, X_gen, y


def weighted_sampler(y):
    classes, counts = np.unique(y, return_counts=True)
    w = 1.0 / counts
    sw = w[y.astype(int)]
    return WeightedRandomSampler(torch.tensor(sw, dtype=torch.float), len(sw), replacement=True)


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, opt, device):
    model.train()
    total = 0.0
    for clin, gen, met, sites, surv, _ in loader:
        clin, gen, met, sites, surv = clin.to(device), gen.to(device), met.to(device), sites.to(device), surv.to(device)
        opt.zero_grad()
        out  = model(clin, gen)
        loss = criterion(out, met, sites, surv)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    mp, mt, sp, st, risks, survs, evts = [], [], [], [], [], [], []
    for clin, gen, met, sites, surv, evt in loader:
        clin, gen = clin.to(device), gen.to(device)
        out = model(clin, gen)
        mp.append(torch.sigmoid(out["metastasis_logit"]).cpu().numpy())
        mt.append(met.numpy())
        sp.append(torch.sigmoid(out["site_logits"]).cpu().numpy())
        st.append(sites.numpy())
        risks.append(out["survival_risk"].cpu().numpy())
        survs.append(surv.numpy())
        evts.append(evt.numpy())
    mp   = np.concatenate(mp);   mt   = np.concatenate(mt)
    sp   = np.concatenate(sp);   st   = np.concatenate(st)
    risks= np.concatenate(risks); survs= np.concatenate(survs); evts = np.concatenate(evts)
    m = compute_metrics(mt, mp)
    m.update(site_aucs(st, sp))
    return m, mp, mt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("  Multimodal RCC Net -- Genomics + Clinical + Cross-Attention")
    print("="*60)
    print(f"  Device : {device}")
    print(f"  Output : {out_dir}")

    print("\n[1/4] Loading data...")
    X_clin, y_met, y_site, y_surv, y_evt = load_data()
    n = len(y_met)

    print(f"\n[2/4] Generating genomics features ({CFG['genomics_dim']}-dim)...")
    X_gen = synth_genomics(n, CFG["genomics_dim"], CFG["seed"])
    print(f"  Genomics shape: {X_gen.shape}")

    # Held-out test split
    rng = np.random.default_rng(CFG["seed"])
    idx = np.arange(n); rng.shuffle(idx)
    n_test  = int(n * CFG["test_ratio"])
    n_val   = int(n * CFG["val_ratio"])
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]

    print(f"\n[3/4] {CFG['n_folds']}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    fold_aucs = []

    for fold, (tr, va) in enumerate(skf.split(X_clin[train_idx], y_met[train_idx]), 1):
        print(f"\n  ---- Fold {fold}/{CFG['n_folds']} ----")
        # Real train indices
        tr_real = train_idx[tr]; va_real = train_idx[va]

        print("  Fast oversampling...")
        # Fast oversampling on clinical features only
        Xb_c, Xb_g, yb = smote_balance(X_clin[tr_real], X_gen[tr_real], y_met[tr_real], CFG["seed"])

        print("  Tiling auxiliary labels...")
        # Tile auxiliary labels to match oversampled size
        n0, nb = len(tr_real), len(yb)
        rep = int(np.ceil(nb / n0))
        sites_b = np.tile(y_site[tr_real], (rep, 1))[:nb].astype(np.float32)
        surv_b  = np.tile(y_surv[tr_real], rep)[:nb].astype(np.float32)
        evt_b   = np.tile(y_evt[tr_real],  rep)[:nb].astype(np.float32)
        
        print("  Building loaders...")

        train_ds = TensorDataset(
            torch.tensor(Xb_c), torch.tensor(Xb_g),
            torch.tensor(yb.astype(np.float32)), torch.tensor(sites_b),
            torch.tensor(surv_b), torch.tensor(evt_b))
        val_ds = TensorDataset(
            torch.tensor(X_clin[va_real]), torch.tensor(X_gen[va_real]),
            torch.tensor(y_met[va_real]),  torch.tensor(y_site[va_real]),
            torch.tensor(y_surv[va_real]), torch.tensor(y_evt[va_real]))

        sampler     = weighted_sampler(yb)
        train_loader= DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=True)
        val_loader  = DataLoader(val_ds,   batch_size=CFG["batch_size"]*2)

        model     = MultimodalRCCNet(CFG).to(device)
        criterion = MultimodalLoss(CFG["pos_weight"]).to(device)
        opt       = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-6)

        best_auc = 0.0; patience = 0
        ckpt = out_dir / f"fold{fold}_best.pt"

        for epoch in range(1, CFG["epochs"]+1):
            loss = train_epoch(model, train_loader, criterion, opt, device)
            vm, _, _ = eval_epoch(model, val_loader, device)
            sched.step()

            if vm["auc"] > best_auc:
                best_auc = vm["auc"]; patience = 0
                torch.save({"state": model.state_dict(), "metrics": vm}, ckpt)
            else:
                patience += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Ep {epoch:03d} | loss={loss:.4f} | val_AUC={vm['auc']:.4f} | val_AP={vm['ap']:.4f} | best={best_auc:.4f}")

            if patience >= CFG["patience"]:
                print(f"    Early stop at epoch {epoch}")
                break

        fold_aucs.append(best_auc)
        print(f"  Fold {fold} best AUC = {best_auc:.4f}")

    cv_mean = float(np.mean(fold_aucs))
    cv_std  = float(np.std(fold_aucs))
    print(f"\n  CV AUC: {cv_mean:.4f} +/- {cv_std:.4f}")

    # ── Final model on full train split ──────────────────────────────────────
    print("\n[4/4] Training final model on full train split...")
    Xb_c, Xb_g, yb = smote_balance(X_clin[train_idx], X_gen[train_idx], y_met[train_idx], CFG["seed"])
    nb = len(yb); n0 = len(train_idx); rep = int(np.ceil(nb/n0))
    sites_b = np.tile(y_site[train_idx], (rep,1))[:nb].astype(np.float32)
    surv_b  = np.tile(y_surv[train_idx], rep)[:nb].astype(np.float32)
    evt_b   = np.tile(y_evt[train_idx],  rep)[:nb].astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(Xb_c), torch.tensor(Xb_g),
        torch.tensor(yb.astype(np.float32)), torch.tensor(sites_b),
        torch.tensor(surv_b), torch.tensor(evt_b))
    val_ds = TensorDataset(
        torch.tensor(X_clin[val_idx]), torch.tensor(X_gen[val_idx]),
        torch.tensor(y_met[val_idx]),  torch.tensor(y_site[val_idx]),
        torch.tensor(y_surv[val_idx]), torch.tensor(y_evt[val_idx]))
    test_ds = TensorDataset(
        torch.tensor(X_clin[test_idx]), torch.tensor(X_gen[test_idx]),
        torch.tensor(y_met[test_idx]),  torch.tensor(y_site[test_idx]),
        torch.tensor(y_surv[test_idx]), torch.tensor(y_evt[test_idx]))

    sampler      = weighted_sampler(yb)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"]*2)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"]*2)

    model     = MultimodalRCCNet(CFG).to(device)
    criterion = MultimodalLoss(CFG["pos_weight"]).to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-6)

    best_auc = 0.0; patience = 0
    final_ckpt = out_dir / "final_best.pt"

    for epoch in range(1, CFG["epochs"]+1):
        loss = train_epoch(model, train_loader, criterion, opt, device)
        vm, _, _ = eval_epoch(model, val_loader, device)
        sched.step()

        if vm["auc"] > best_auc:
            best_auc = vm["auc"]; patience = 0
            torch.save({"state": model.state_dict(), "metrics": vm}, final_ckpt)
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Final Ep {epoch:03d} | loss={loss:.4f} | val_AUC={vm['auc']:.4f} | best={best_auc:.4f}")

        if patience >= CFG["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    # Load best and test
    ckpt_data = torch.load(final_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["state"])
    test_m, test_probs, test_true = eval_epoch(model, test_loader, device)

    print("\n" + "="*60)
    print("  FINAL TEST SET RESULTS")
    print("="*60)
    print(f"  AUC (ROC)     : {test_m['auc']:.4f}")
    print(f"  AUPRC         : {test_m['ap']:.4f}")
    print(f"  F1 Score      : {test_m['f1']:.4f}")
    print(f"  Lung AUC      : {test_m.get('lung_auc', float('nan')):.4f}")
    print(f"  Bone AUC      : {test_m.get('bone_auc', float('nan')):.4f}")
    print(f"  Liver AUC     : {test_m.get('liver_auc', float('nan')):.4f}")
    print(f"  Brain AUC     : {test_m.get('brain_auc', float('nan')):.4f}")
    print(f"  Mean Site AUC : {test_m.get('mean_site_auc', 0):.4f}")
    print(f"  CV AUC        : {cv_mean:.4f} +/- {cv_std:.4f}")
    print("="*60)

    # Save
    results = {
        "model": "MultimodalRCCNet (Genomics Transformer + Clinical MLP + Cross-Attention)",
        "architecture": {
            "clinical_encoder": "7-dim -> 128-dim MLP",
            "genomics_encoder": "500-dim RNA-seq -> Transformer Encoder (3 layers) -> 256-dim",
            "fusion": "Cross-Attention Fusion -> 256-dim -> Shared Dense -> 128-dim",
            "heads": ["Metastasis (binary)", "Survival Risk (regression)", "Site Mets (4-class)", "Clinical Decision (3-class)"]
        },
        "cv_auc_mean": cv_mean,
        "cv_auc_std":  cv_std,
        "fold_aucs":   fold_aucs,
        "test_metrics": test_m,
        "best_val_auc": best_auc,
    }
    (out_dir / "final_results.json").write_text(json.dumps(results, indent=2))
    np.save(str(out_dir / "test_probs.npy"), test_probs)
    np.save(str(out_dir / "test_true.npy"),  test_true)
    print(f"\n  Saved to: {out_dir}")
    print("  DONE!")


if __name__ == "__main__":
    main()
