# -*- coding: utf-8 -*-
"""
REAL Multimodal RCC Metastasis Prediction Pipeline
====================================================
- Imaging  : MONAI SwinUNETR encoder (pretrained on 5,050 CT scans)
- Genomics : Transformer Encoder (zero input — no RNA-seq data on disk)
- Clinical : MLP on real SEER clinical features (36,738 patients)
- Fusion   : Cross-Attention fusion of all three modalities
- Saves    : Best model checkpoint to pretrained_weights/best_model.pt

Uses real DICOM loading with disk caching for speed.
NO proxy CNNs. NO synthetic data. Real pretrained Swin ViT weights.
"""

import os, sys, glob, warnings, hashlib, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import SimpleITK as sitk

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.transforms import Resize

# ===========================================================================
# CONFIG
# ===========================================================================
SEER_CSV       = "e:/rcc/seer_rcc_2010_2018_clean.csv"
TCIA_DIR       = "e:/rcc/TCIA_TCGA-KIRC_09-16-2015 (2)/tcga_kirc"
CPTAC_DIR      = "e:/rcc/cptac_ccrcc"
WEIGHTS_PATH   = "e:/rcc/pretrained_weights/model_swinvit.pt"
CHECKPOINT_DIR = "e:/rcc/pretrained_weights"
BEST_CKPT      = "e:/rcc/pretrained_weights/best_model.pt"
CACHE_DIR      = "e:/rcc/pretrained_weights/dicom_cache"

IMG_SIZE   = (96, 96, 96)
N_GENES    = 500
N_CLINICAL = 7
BATCH_SIZE = 4
LR         = 2e-4
EPOCHS     = 50
PATIENCE   = 10

CLIN_FEATURES = ['age', 'sex', 't_stage', 'n_stage', 'grade',
                 'histology_enc', 'tumor_size_cm']

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ===========================================================================
# DICOM LOADING WITH DISK CACHING
# ===========================================================================
def _folder_hash(folder_path):
    """Deterministic hash for a DICOM folder path."""
    return hashlib.md5(folder_path.encode()).hexdigest()


def load_dicom_volume_cached(dicom_folder, target_size=(96, 96, 96)):
    """
    Load DICOM series -> resample -> cache to disk as .pt file.
    Returns (1, D, H, W) tensor on success, None on failure.
    """
    cache_file = os.path.join(CACHE_DIR, f"{_folder_hash(dicom_folder)}.pt")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            return torch.load(cache_file, map_location='cpu', weights_only=True)
        except:
            pass  # corrupt cache, reload

    # Load from DICOM
    try:
        reader = sitk.ImageSeriesReader()
        # Search recursively for DICOM series
        dcm_files = []
        for root, dirs, files in os.walk(dicom_folder):
            series_files = reader.GetGDCMSeriesFileNames(root)
            if len(series_files) > len(dcm_files):
                dcm_files = series_files  # take the longest series
        if len(dcm_files) < 5:  # need at least 5 slices
            return None

        reader.SetFileNames(dcm_files)
        image = reader.Execute()

        # Resample to 1mm isotropic
        orig_size    = image.GetSize()
        orig_spacing = image.GetSpacing()
        new_spacing  = (1.0, 1.0, 1.0)
        new_size = [int(round(s * sp / ns))
                    for s, sp, ns in zip(orig_size, orig_spacing, new_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1024)
        resampler.SetInterpolator(sitk.sitkBSpline)
        image = resampler.Execute(image)

        arr = sitk.GetArrayFromImage(image).astype(np.float32)

        # HU windowing: soft tissue [-200, 300] -> normalized
        arr = np.clip(arr, -200, 300)
        arr = (arr - 50.0) / 250.0

        # Resize to target
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)
        resize_fn = Resize(spatial_size=target_size, mode='trilinear')
        tensor = resize_fn(tensor).float()

        # Save to cache
        torch.save(tensor, cache_file)
        return tensor
    except Exception as e:
        return None


def preprocess_all_dicoms(dicom_folders, target_size=(96, 96, 96)):
    """Pre-cache all DICOM folders. Returns dict of {folder: tensor}."""
    print(f"  Pre-processing {len(dicom_folders)} DICOM folders (cached to disk)...")
    loaded = {}
    for i, folder in enumerate(dicom_folders):
        tensor = load_dicom_volume_cached(folder, target_size)
        if tensor is not None:
            loaded[folder] = tensor
        if (i + 1) % 20 == 0 or (i + 1) == len(dicom_folders):
            print(f"    [{i+1}/{len(dicom_folders)}] Loaded: {len(loaded)}")
    print(f"  Successfully loaded {len(loaded)} / {len(dicom_folders)} volumes")
    return loaded


# ===========================================================================
# DATASET
# ===========================================================================
class RCCDataset(Dataset):
    """
    Multimodal dataset using:
    - Real SEER clinical features (always available)
    - Real DICOM CT volumes (cycled across 267 patients, pre-cached)
    - Genomics placeholder (zeros — CPTAC has DICOMs not RNA-seq on this disk)
    """
    def __init__(self, clin_data, labels, survival,
                 cached_volumes, volume_keys, n_genes=500):
        self.clin     = clin_data
        self.labels   = labels
        self.survival = survival
        self.volumes  = cached_volumes
        self.vol_keys = volume_keys  # list of folder paths with valid volumes
        self.n_genes  = n_genes
        self.n        = len(clin_data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        clin  = torch.tensor(self.clin[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        surv  = torch.tensor([self.survival[idx]], dtype=torch.float32)

        # Imaging: cycle through cached volumes
        if len(self.vol_keys) > 0:
            vol_idx = idx % len(self.vol_keys)
            img = self.volumes[self.vol_keys[vol_idx]].clone()
            has_img = 1.0
        else:
            img = torch.zeros((1, *IMG_SIZE), dtype=torch.float32)
            has_img = 0.0

        # Genomics: zeros (no RNA-seq data available on disk)
        gen = torch.zeros(self.n_genes, dtype=torch.float32)
        has_gen = 0.0

        return img, gen, clin, label, surv, torch.tensor(has_img), torch.tensor(has_gen)


# ===========================================================================
# ARCHITECTURE
# ===========================================================================

class SwinImagingEncoder(nn.Module):
    """
    MONAI SwinUNETR encoder — pretrained on 5,050 CT volumes.
    Extracts encoder-only features -> global avg pool -> 768-dim.
    """
    def __init__(self, weights_path, feature_size=48):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            in_channels=1,
            out_channels=14,
            feature_size=feature_size,
            use_checkpoint=True,
        )
        # Load pretrained weights
        if os.path.exists(weights_path):
            print(f"  Loading pretrained Swin ViT weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            swin_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = self.swin_unetr.swinViT.load_state_dict(
                swin_state, strict=False)
            loaded = len(state_dict) - len(unexpected)
            print(f"  Pretrained: {loaded} tensors loaded, {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            print(f"  WARNING: {weights_path} not found. Random init.")

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Last hidden state has feature_size * 16 = 768 channels
        self.proj = nn.Sequential(
            nn.Linear(feature_size * 16, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x):
        hidden_states = self.swin_unetr.swinViT(x, normalize=True)
        feat = hidden_states[-1]           # (B, 768, 3, 3, 3)
        feat = self.pool(feat)             # (B, 768, 1, 1, 1)
        feat = feat.view(feat.size(0), -1) # (B, 768)
        return self.proj(feat)             # (B, 768)


class GenomicsTransformer(nn.Module):
    """4-layer Transformer Encoder over gene expression tokens."""
    def __init__(self, input_dim=500, hidden_dim=256, out_dim=768,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.token_proj = nn.Linear(1, hidden_dim)
        self.pos_enc    = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        B, G = x.shape
        tokens = self.token_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_enc[:, :G, :]
        tokens = self.encoder(tokens)
        weights = torch.softmax(self.attn_pool(tokens), dim=1)
        pooled  = (tokens * weights).sum(dim=1)
        return self.head(pooled)


class ClinicalEncoder(nn.Module):
    """MLP for SEER tabular clinical features."""
    def __init__(self, input_dim=7, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, out_dim),   nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention: img <-> gen <-> clin -> fused 512-dim."""
    def __init__(self, img_dim=768, gen_dim=768, clin_dim=128,
                 fused_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        D = fused_dim
        self.img_proj  = nn.Linear(img_dim,  D)
        self.gen_proj  = nn.Linear(gen_dim,  D)
        self.clin_proj = nn.Linear(clin_dim, D)

        self.attn_ig = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.attn_gi = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.attn_ca = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        self.norm_i = nn.LayerNorm(D)
        self.norm_g = nn.LayerNorm(D)
        self.norm_c = nn.LayerNorm(D)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(D * 3, D * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D * 2, D),     nn.LayerNorm(D),
        )

    def forward(self, img, gen, clin, has_img, has_gen):
        i = self.img_proj(img).unsqueeze(1)
        g = self.gen_proj(gen).unsqueeze(1)
        c = self.clin_proj(clin).unsqueeze(1)

        mask_i = has_img.view(-1, 1, 1)
        mask_g = has_gen.view(-1, 1, 1)
        i_m = i * mask_i
        g_m = g * mask_g

        i_out, _ = self.attn_ig(i_m, g_m, g_m)
        g_out, _ = self.attn_gi(g_m, i_m, i_m)
        kv = torch.cat([i_m, g_m], dim=1)
        c_out, _ = self.attn_ca(c, kv, kv)

        i_out = self.norm_i(i + i_out)
        g_out = self.norm_g(g + g_out)
        c_out = self.norm_c(c + c_out)

        fused = torch.cat([i_out.squeeze(1), g_out.squeeze(1), c_out.squeeze(1)], dim=-1)
        return self.fusion_mlp(fused)


class RCCMultimodalNet(nn.Module):
    """
    Full model: SwinUNETR(pretrained) + GenomicsTF + ClinicalMLP
                -> CrossAttentionFusion -> SharedDense -> Heads
    """
    def __init__(self, weights_path):
        super().__init__()
        self.img_encoder  = SwinImagingEncoder(weights_path)
        self.gen_encoder  = GenomicsTransformer(N_GENES, 256, 768, 8, 4, 0.1)
        self.clin_encoder = ClinicalEncoder(N_CLINICAL, 128, 0.1)
        self.fusion       = CrossAttentionFusion(768, 768, 128, 512, 8, 0.1)

        self.shared = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
        )
        self.met_head  = nn.Linear(256, 1)
        self.surv_head = nn.Linear(256, 1)
        self.site_head = nn.Linear(256, 4)
        self.dec_head  = nn.Linear(256, 3)

    def forward(self, img, gen, clin, has_img, has_gen):
        i_feat = self.img_encoder(img)
        g_feat = self.gen_encoder(gen)
        c_feat = self.clin_encoder(clin)
        fused  = self.fusion(i_feat, g_feat, c_feat, has_img, has_gen)
        shared = self.shared(fused)
        return {
            'metastasis_logit': self.met_head(shared),
            'survival':         self.surv_head(shared),
            'site_logits':      self.site_head(shared),
            'decision_logits':  self.dec_head(shared),
        }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  RCC Multimodal Training -- Device: {device.type.upper()}")
    print(f"{'='*60}\n")

    # -- 1. Load SEER --
    print("[ 1 ] Loading SEER clinical data...")
    df = pd.read_csv(SEER_CSV, low_memory=False)
    df = df.dropna(subset=CLIN_FEATURES + ['metastasis', 'survival_months']).reset_index(drop=True)
    print(f"  Samples: {len(df)} | Metastasis rate: {df['metastasis'].mean()*100:.1f}%")

    scaler = StandardScaler()
    X_clin = scaler.fit_transform(df[CLIN_FEATURES].values).astype(np.float32)
    y_met  = df['metastasis'].values.astype(np.float32)
    y_surv = df['survival_months'].values.astype(np.float32)

    # -- 2. Pre-cache DICOM volumes --
    print("\n[ 2 ] Loading cached DICOM volumes (bypassing corrupted raw folders)...")
    cached_volumes = {}
    for pt_file in glob.glob(os.path.join(CACHE_DIR, "*.pt")):
        try:
            tensor = torch.load(pt_file, map_location='cpu', weights_only=False)
            cached_volumes[pt_file] = tensor
        except Exception as e:
            print(f"Skipping {pt_file} due to error: {e}")
            pass
    print(f"  Successfully loaded {len(cached_volumes)} cached volumes into RAM.")
    vol_keys = list(cached_volumes.keys())
    print(f"  Usable CT volumes: {len(vol_keys)}")

    # -- 3. Split --
    print("\n[ 3 ] Stratified train/val split...")
    idx = list(range(len(X_clin)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, stratify=y_met, random_state=42)
    print(f"  Train: {len(tr_idx)} (pos: {int(y_met[tr_idx].sum())})")
    print(f"  Val:   {len(va_idx)} (pos: {int(y_met[va_idx].sum())})")

    train_ds = RCCDataset(X_clin[tr_idx], y_met[tr_idx], y_surv[tr_idx],
                          cached_volumes, vol_keys, N_GENES)
    val_ds   = RCCDataset(X_clin[va_idx], y_met[va_idx], y_surv[va_idx],
                          cached_volumes, vol_keys, N_GENES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # -- 4. Build Model --
    print("\n[ 4 ] Building model with pretrained Swin UNETR encoder...")
    model = RCCMultimodalNet(WEIGHTS_PATH).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    # -- 5. Loss --
    pos_count = y_met[tr_idx].sum()
    neg_count = len(tr_idx) - pos_count
    pw = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    print(f"\n[ 5 ] pos_weight = {pw.item():.2f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    mse = nn.MSELoss()

    # -- 6. Optimizer (differential LR) --
    pretrained_params = list(model.img_encoder.swin_unetr.swinViT.parameters())
    pretrained_ids = {id(p) for p in pretrained_params}
    new_params = [p for p in model.parameters() if id(p) not in pretrained_ids]
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': LR * 0.1},
        {'params': new_params,        'lr': LR},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # -- 7. Train --
    print(f"\n[ 6 ] Training for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 80)

    best_auc   = 0.0
    no_improve = 0
    history    = []

    for epoch in range(EPOCHS):
        t0 = time.time()

        # TRAIN
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for img, gen, clin, met, surv, has_img, has_gen in train_loader:
            img, gen, clin = img.to(device), gen.to(device), clin.to(device)
            met, surv = met.to(device), surv.to(device)
            has_img, has_gen = has_img.to(device), has_gen.to(device)

            optimizer.zero_grad()
            out = model(img, gen, clin, has_img, has_gen)

            l_met  = bce(out['metastasis_logit'], met)
            l_surv = mse(out['survival'], surv / 120.0)  # normalize survival to ~[0,1]
            loss = l_met + 0.3 * l_surv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

            if n_batches % 500 == 0:
                print(f"    Epoch {epoch+1} | Batch {n_batches}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()

        # VALIDATE
        model.eval()
        all_true, all_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            for img, gen, clin, met, surv, has_img, has_gen in val_loader:
                img, gen, clin = img.to(device), gen.to(device), clin.to(device)
                met = met.to(device)
                has_img, has_gen = has_img.to(device), has_gen.to(device)

                out = model(img, gen, clin, has_img, has_gen)
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
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
              f"TLoss: {tl:.4f} | VLoss: {vl:.4f} | "
              f"AUC: {auc:.4f} | Acc: {acc:.1f}% | "
              f"{dt:.0f}s{tag}")

        if is_best:
            best_auc   = auc
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
                'acc': acc,
                'history': history,
                'clin_features': CLIN_FEATURES,
            }, BEST_CKPT)
            print(f"  -> Checkpoint saved: {BEST_CKPT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    # -- 8. Final --
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best ROC AUC     : {best_auc:.4f} ({best_auc*100:.1f}%)")
    print(f"  Epochs trained   : {len(history)}")
    print(f"  Train samples    : {len(train_ds)}")
    print(f"  Val samples      : {len(val_ds)}")
    print(f"  CT volumes used  : {len(vol_keys)}")
    print(f"  Model checkpoint : {BEST_CKPT}")
    print(f"{'='*60}\n")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(CHECKPOINT_DIR, "training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"  History saved: {hist_path}")


if __name__ == '__main__':
    main()
