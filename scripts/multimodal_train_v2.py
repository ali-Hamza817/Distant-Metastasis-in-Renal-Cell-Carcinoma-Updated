# -*- coding: utf-8 -*-
import sys
"""
FIXED Multimodal RCC Metastasis Prediction Pipeline v2
=======================================================
CRITICAL FIX: No more random image cycling. Every sample has correct modality masks.

Dataset A: 36,738 SEER patients  -> clinical features only (has_img=0)
Dataset B: 255 TCGA-KIRC patients -> real CT scan + clinical (has_img=1, real M-stage label)

Training strategy:
 - Clinical encoder (SEER) learns on 36K real patient records -> should reach AUC ~82-88%
 - SwinUNETR encoder (TCGA) gets 255 real paired CT+label samples -> grounded imaging signal
 - Cross-attention fusion merges both towers

Architecture is IDENTICAL to v1 (pretrained Swin ViT weights re-loaded).
This ensures a fair ablation: same model, fixed data pipeline.
"""

import os, sys, glob, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR

# ===========================================================================
# CONFIG
# ===========================================================================
SEER_CSV        = "e:/rcc/seer_rcc_2010_2018_clean.csv"
TCGA_CLIN_CSV   = "e:/rcc/data/tcga_kirc_clinical.csv"
WEIGHTS_PATH    = "e:/rcc/pretrained_weights/model_swinvit.pt"
BEST_CKPT       = "e:/rcc/pretrained_weights/best_model_v2.pt"
CACHE_DIR       = "e:/rcc/pretrained_weights/dicom_cache_v2"  # patient_id-named .pt files
CHECKPOINT_DIR  = "e:/rcc/pretrained_weights"
META_CSV        = "E:/rcc/TCIA_TCGA-KIRC_09-16-2015 (2)/metadata/metadata.csv"

IMG_SIZE   = (96, 96, 96)
N_GENES    = 500
N_CLINICAL = 8            # added histology_enc and prior_tx -> 8 features
BATCH_SIZE = 4
LR         = 2e-4
EPOCHS     = 60
PATIENCE   = 12

SEER_CLIN_FEATURES = [
    'age', 'sex', 't_stage', 'n_stage',
    'tumor_size_cm', 'grade', 'histology_enc', 'prior_tx'
]

# ===========================================================================
# DATASET
# ===========================================================================

class RCCDataset(Dataset):
    """
    Unified dataset combining SEER (clinical-only) and TCGA-KIRC (CT+clinical).
    
    Each sample returns:
      img        : (1,96,96,96) tensor or zeros
      clin       : (8,) clinical feature vector
      label      : (1,) binary metastasis label
      surv       : (1,) survival months (normalized)
      has_img    : float scalar (1.0 if real CT, 0.0 otherwise)
    """
    def __init__(self, records, cached_volumes, vol_key_map):
        """
        records: list of dicts with keys:
            patient_id, age, sex, t_stage, n_stage, tumor_size_cm,
            grade, histology_enc, prior_tx, metastasis, survival_months,
            cache_key (str or None)
        cached_volumes: dict of path -> tensor
        vol_key_map: dict of patient_id -> cache_path (for TCGA patients)
        """
        self.records = records
        self.volumes = cached_volumes
        self.vol_map = vol_key_map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        clin = torch.tensor([
            r['age'], r['sex'], r['t_stage'], r['n_stage'],
            r['tumor_size_cm'], r['grade'], r['histology_enc'], r['prior_tx']
        ], dtype=torch.float32)

        label = torch.tensor([r['metastasis']], dtype=torch.float32)
        surv  = torch.tensor([min(r['survival_months'], 167.0) / 167.0], dtype=torch.float32)

        pid = r.get('patient_id', '')
        if pid in self.vol_map and self.vol_map[pid] in self.volumes:
            img     = self.volumes[self.vol_map[pid]].clone()
            has_img = torch.tensor(1.0)
        else:
            img     = torch.zeros((1, *IMG_SIZE), dtype=torch.float32)
            has_img = torch.tensor(0.0)

        return img, clin, label, surv, has_img


# ===========================================================================
# MODEL (identical architecture to v1 — fair comparison)
# ===========================================================================

class SwinImagingEncoder(nn.Module):
    def __init__(self, weights_path, feature_size=48):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            in_channels=1, out_channels=14,
            feature_size=feature_size, use_checkpoint=True,
        )
        if os.path.exists(weights_path):
            print(f"  Loading pretrained Swin ViT weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            swin_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.swin_unetr.swinViT.load_state_dict(swin_state, strict=False)
            loaded = len(state_dict) - len(unexpected)
            print(f"  Pretrained: {loaded} tensors loaded, {len(missing)} missing")
        else:
            print(f"  WARNING: weights not found at {weights_path}. Random init.")

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(feature_size * 16, 768),
            nn.LayerNorm(768), nn.GELU(),
        )

    def forward(self, x):
        hidden = self.swin_unetr.swinViT(x, normalize=True)
        feat = self.pool(hidden[-1]).view(hidden[-1].size(0), -1)
        return self.proj(feat)


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=8, out_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 512),       nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, out_dim),   nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityFusion(nn.Module):
    """
    Simpler fusion: clinical always present; imaging only when has_img=1.
    Uses gated attention to weight imaging contribution.
    """
    def __init__(self, img_dim=768, clin_dim=256, fused_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        D = fused_dim
        self.img_proj  = nn.Linear(img_dim,  D)
        self.clin_proj = nn.Linear(clin_dim, D)

        self.cross_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(D)
        self.norm_i = nn.LayerNorm(D)

        # Gating: learns when to trust imaging signal
        self.gate = nn.Sequential(nn.Linear(D * 2, D), nn.Sigmoid())

        self.fusion_mlp = nn.Sequential(
            nn.Linear(D * 2, D), nn.LayerNorm(D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D, D),
        )

    def forward(self, img_feat, clin_feat, has_img):
        i = self.img_proj(img_feat).unsqueeze(1)   # (B, 1, D)
        c = self.clin_proj(clin_feat).unsqueeze(1) # (B, 1, D)

        # Zero out imaging contribution where has_img=0
        mask = has_img.view(-1, 1, 1)
        i_masked = i * mask

        # Clinical attends to imaging (only matters when has_img=1)
        c_attn, _ = self.cross_attn(c, i_masked, i_masked)
        c_out = self.norm_c(c + c_attn)   # residual
        i_out = self.norm_i(i + i_masked) # trivial when masked

        c_sq = c_out.squeeze(1)
        i_sq = i_out.squeeze(1)

        gate = self.gate(torch.cat([c_sq, i_sq], dim=-1))
        fused = torch.cat([c_sq, gate * i_sq], dim=-1)
        return self.fusion_mlp(fused)


class RCCNetV2(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.img_encoder  = SwinImagingEncoder(weights_path)
        self.clin_encoder = ClinicalEncoder(N_CLINICAL, 256, 0.15)
        self.fusion       = ModalityFusion(768, 256, 512, 8, 0.1)

        self.shared = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
        )
        self.met_head  = nn.Linear(256, 1)
        self.surv_head = nn.Linear(256, 1)

    def forward(self, img, clin, has_img):
        i_feat = self.img_encoder(img)
        c_feat = self.clin_encoder(clin)
        fused  = self.fusion(i_feat, c_feat, has_img)
        shared = self.shared(fused)
        return {
            'metastasis_logit': self.met_head(shared),
            'survival':         self.surv_head(shared),
        }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  RCC Multimodal v2 -- FIXED DATA PIPELINE -- {device.type.upper()}")
    print(f"{'='*60}\n")

    # ── 1. Load SEER (clinical only) ─────────────────────────────────────
    print("[ 1 ] Loading SEER clinical data...")
    seer = pd.read_csv(SEER_CSV, low_memory=False)
    seer = seer.dropna(subset=SEER_CLIN_FEATURES + ['metastasis', 'survival_months'])
    seer = seer.reset_index(drop=True)
    print(f"  SEER samples: {len(seer)} | Metastasis rate: {seer['metastasis'].mean()*100:.1f}%")

    # ── 2. Load TCGA-KIRC with real M-stage labels ────────────────────────
    print("\n[ 2 ] Loading TCGA-KIRC clinical labels...")
    tcga = pd.read_csv(TCGA_CLIN_CSV)
    tcga = tcga[tcga['metastasis'].notna()].reset_index(drop=True)
    print(f"  TCGA samples with M-stage: {len(tcga)} | M1: {int((tcga['metastasis']==1).sum())} | M0: {int((tcga['metastasis']==0).sum())}")

    # ── 3. Load cached DICOM volumes ──────────────────────────────────────
    print("\n[ 3 ] Loading cached DICOM volumes from dicom_cache_v2 (patient_id named)...")
    cached_volumes = {}
    pt_files = glob.glob(os.path.join(CACHE_DIR, '*.pt'))
    if not pt_files:
        print(f"  WARNING: No .pt files found in {CACHE_DIR}")
        print(f"  Re-caching is likely still in progress. TCGA patients will use zero tensors.")
    for pt_file in pt_files:
        try:
            tensor = torch.load(pt_file, map_location='cpu', weights_only=False)
            if not isinstance(tensor, torch.Tensor):
                tensor = tensor.as_tensor() if hasattr(tensor, 'as_tensor') else torch.tensor(tensor)
            cached_volumes[pt_file] = tensor.float()
        except Exception as e:
            print(f"  Skip {os.path.basename(pt_file)}: {e}")
    print(f"  Loaded {len(cached_volumes)} cached volumes.")

    # Map TCGA patient IDs -> cache file paths
    # dicom_cache_v2 uses patient_id.pt naming — trivial lookup
    vol_key_map = {}  # patient_id -> cache_path
    for pid in tcga['patient_id'].values:
        candidate = os.path.join(CACHE_DIR, f"{pid}.pt")
        if candidate in cached_volumes:
            vol_key_map[pid] = candidate
    print(f"  TCGA patients matched to cached volumes: {len(vol_key_map)}")

    # ── 4. Build unified clinical feature matrix ──────────────────────────
    print("\n[ 4 ] Building unified training records...")

    # Normalize SEER clinical features
    seer_scaler = StandardScaler()
    seer_clin = seer_scaler.fit_transform(seer[SEER_CLIN_FEATURES].values).astype(np.float32)

    def make_record(row, clin_vec, pid=None):
        return {
            'patient_id':     pid or '',
            'age':            float(clin_vec[0]),
            'sex':            float(clin_vec[1]),
            't_stage':        float(clin_vec[2]),
            'n_stage':        float(clin_vec[3]),
            'tumor_size_cm':  float(clin_vec[4]),
            'grade':          float(clin_vec[5]),
            'histology_enc':  float(clin_vec[6]),
            'prior_tx':       float(clin_vec[7]),
            'metastasis':     float(row['metastasis']),
            'survival_months': float(row['survival_months']) if pd.notna(row.get('survival_months', None)) else 60.0,
        }

    # SEER records (no image)
    seer_records = []
    for i, row in seer.iterrows():
        r = make_record(row, seer_clin[i])
        seer_records.append(r)

    # TCGA records — need to map their clinical features to same scale
    # TCGA has: age, sex, t_stage, n_stage. Fill missing with SEER means.
    seer_means = seer[SEER_CLIN_FEATURES].mean()

    def t_stage_num(s):
        s = str(s).lower()
        if 't4' in s: return 4
        if 't3' in s: return 3
        if 't2' in s: return 2
        if 't1' in s: return 1
        return 0

    def n_stage_num(s):
        s = str(s).lower()
        if 'n1' in s: return 1
        return 0

    tcga_records = []
    for _, row in tcga.iterrows():
        raw_age = row.get('age', seer_means['age'])
        if pd.isna(raw_age): raw_age = seer_means['age']
        raw_sex = row.get('sex', 0.5)
        if pd.isna(raw_sex): raw_sex = 0.5
        t_st = t_stage_num(row.get('t_stage_raw', 'T0'))
        n_st = n_stage_num(row.get('n_stage_raw', 'N0'))

        # Scale using SEER scaler (same normalization space)
        raw_vec = np.array([
            float(raw_age),
            float(raw_sex),
            float(t_st),
            float(n_st),
            float(seer_means['tumor_size_cm']),  # unknown -> mean
            float(seer_means['grade']),
            float(seer_means['histology_enc']),
            float(seer_means['prior_tx']),
        ], dtype=np.float32).reshape(1, -1)
        scaled = seer_scaler.transform(raw_vec)[0]

        surv = row.get('survival_months', 60.0)
        if pd.isna(surv): surv = 60.0

        r = make_record(
            {'metastasis': row['metastasis'], 'survival_months': surv},
            scaled,
            pid=row['patient_id']
        )
        tcga_records.append(r)

    print(f"  SEER records  (no image): {len(seer_records)}")
    print(f"  TCGA records  (with img): {len(tcga_records)}")
    total_records = seer_records + tcga_records
    print(f"  Total records: {len(total_records)}")
    met_rate = np.mean([r['metastasis'] for r in total_records])
    print(f"  Overall metastasis rate: {met_rate*100:.1f}%")

    # ── 5. Stratified split ───────────────────────────────────────────────
    print("\n[ 5 ] Stratified train/val split...")
    labels_all = [r['metastasis'] for r in total_records]
    idx = list(range(len(total_records)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=labels_all, random_state=42)
    print(f"  Train: {len(tr_idx)} | Val: {len(va_idx)}")

    tr_records = [total_records[i] for i in tr_idx]
    va_records = [total_records[i] for i in va_idx]

    train_ds = RCCDataset(tr_records, cached_volumes, vol_key_map)
    val_ds   = RCCDataset(va_records, cached_volumes, vol_key_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── 6. Build Model ────────────────────────────────────────────────────
    print("\n[ 6 ] Building model v2 (fixed fusion)...")
    model = RCCNetV2(WEIGHTS_PATH).to(device)
    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_p:,} | Trainable: {trainable:,}")

    # ── 7. Loss (per-sample, correctly weighted) ──────────────────────────
    pos_count = sum(r['metastasis'] for r in tr_records)
    neg_count = len(tr_records) - pos_count
    pw = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    print(f"\n[ 7 ] pos_weight = {pw.item():.2f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    mse = nn.MSELoss()

    # ── 8. Optimizer: lower LR for Swin encoder ───────────────────────────
    pretrained_params = list(model.img_encoder.swin_unetr.swinViT.parameters())
    pretrained_ids    = {id(p) for p in pretrained_params}
    new_params        = [p for p in model.parameters() if id(p) not in pretrained_ids]
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': LR * 0.05},  # even more conservative for Swin
        {'params': new_params,        'lr': LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── 9. Training Loop ──────────────────────────────────────────────────
    print(f"\n[ 8 ] Training for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 80)

    best_auc   = 0.0
    no_improve = 0
    history    = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for img, clin, met, surv, has_img in train_loader:
            img, clin = img.to(device), clin.to(device)
            met, surv = met.to(device), surv.to(device)
            has_img   = has_img.to(device)

            optimizer.zero_grad()
            out = model(img, clin, has_img)

            l_met  = bce(out['metastasis_logit'], met)
            l_surv = mse(out['survival'], surv)
            loss   = l_met + 0.2 * l_surv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

            if n_batches % 50 == 0:
                print(f"    Epoch {epoch+1} | Batch {n_batches}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")
                sys.stdout.flush()

        scheduler.step()

        # Validate
        model.eval()
        all_true, all_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for img, clin, met, surv, has_img in val_loader:
                img, clin = img.to(device), clin.to(device)
                met       = met.to(device)
                has_img   = has_img.to(device)
                out       = model(img, clin, has_img)
                val_loss += bce(out['metastasis_logit'], met).item()
                probs = torch.sigmoid(out['metastasis_logit']).cpu().numpy().flatten()
                all_pred.extend(probs)
                all_true.extend(met.cpu().numpy().flatten())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        auc = roc_auc_score(all_true, all_pred) if len(np.unique(all_true)) > 1 else 0.5
        acc = ((all_pred > 0.5) == all_true).mean() * 100
        tl  = train_loss / max(n_batches, 1)
        vl  = val_loss / max(len(val_loader), 1)
        dt  = time.time() - t0

        is_best = auc > best_auc
        history.append({'epoch': epoch+1, 'train_loss': tl, 'val_loss': vl,
                        'auc': auc, 'acc': acc, 'time_sec': dt})

        tag = " << BEST" if is_best else ""
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | TLoss: {tl:.4f} | VLoss: {vl:.4f} | "
              f"AUC: {auc:.4f} | Acc: {acc:.1f}%{tag}")

        if is_best:
            best_auc   = auc
            no_improve = 0
            torch.save({
                'epoch':             epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc':               best_auc,
                'history':           history,
            }, BEST_CKPT)
            print(f"  -> Checkpoint saved: {BEST_CKPT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}.")
                break

    # Final
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE (v2)")
    print(f"  Best ROC AUC  : {best_auc:.4f} ({best_auc*100:.2f}%)")
    print(f"  Epochs trained: {len(history)}")
    print(f"  Checkpoint    : {BEST_CKPT}")
    print(f"{'='*60}\n")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(CHECKPOINT_DIR, "training_history_v2.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"  History saved: {hist_path}")


if __name__ == '__main__':
    main()
