"""
Fused 3D Swin Transformer Multimodal Network
=============================================
Architecture:
  CT Volume  -> 3D Swin Transformer  -> 768-dim imaging features
  RNA-seq    -> Transformer Encoder  -> 768-dim genomic features
  Clinical   -> Linear projection    ->  50-dim clinical features
                    |
              Cross-Attention Fusion
                    |
              Shared Dense Layers
                    |
       ┌────────────┼────────────┐
   Metastasis   Survival     Clinical
   (binary)     (risk score)  Decision
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Volumetric patch embedding (P, H, W) -> tokens."""

    def __init__(self, in_channels: int = 1, patch_size: int = 4,
                 embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)                         # (B, embed_dim, d, h, w)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, D*H*W, C)
        x = self.norm(x)
        return x


class WindowAttention3D(nn.Module):
    """Multi-head self-attention with relative position bias (simplified)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock3D(nn.Module):
    """Single Swin-Transformer block (no window shifting for simplicity)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    """Downsample token count by 2x in each spatial dim."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple channel-doubling merge (works on 1D token sequence)
        B, N, C = x.shape
        # Pad to even
        if N % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            N = N + 1
        x = x.reshape(B, N // 2, 2, C)
        x = x.reshape(B, N // 2, 2 * C)
        # Pad to multiple of 4 for second merge
        N2 = x.shape[1]
        if N2 % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            N2 = N2 + 1
        x = x.reshape(B, N2 // 2, 2 * 2 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------------------------------------------------------
# 3-D Swin Transformer Encoder
# ---------------------------------------------------------------------------

class SwinTransformer3D(nn.Module):
    """
    Lightweight 3-D Swin Transformer.

    Stages:
      Stage 0: patch embed  → embed_dim    tokens  (2 blocks)
      Stage 1: merge        → 2*embed_dim  tokens  (2 blocks)
      Stage 2: merge        → 4*embed_dim  tokens  (2 blocks)
      Global avg pool → project to out_dim
    """

    def __init__(self, in_channels: int = 1, patch_size: int = 4,
                 embed_dim: int = 96, depths: tuple = (2, 2, 2),
                 num_heads: tuple = (3, 6, 12), out_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            blocks = nn.Sequential(*[
                SwinBlock3D(dim, num_heads=heads, dropout=dropout)
                for _ in range(depth)
            ])
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.merges.append(PatchMerging3D(dim))
                dim = dim * 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.patch_embed(x)   # (B, N, embed_dim)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.merges):
                x = self.merges[i](x)
        x = self.norm(x)
        x = x.mean(dim=1)         # global average pool → (B, dim)
        x = self.head(x)          # (B, out_dim)
        return x


# ---------------------------------------------------------------------------
# Genomics Transformer Encoder
# ---------------------------------------------------------------------------

class GenomicsTransformer(nn.Module):
    """
    1-D Transformer Encoder for RNA-seq vectors.
    Treats each gene as one token in the sequence.
    """

    def __init__(self, input_dim: int = 500, hidden_dim: int = 256,
                 out_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        # Project each feature scalar → hidden_dim token
        self.token_proj = nn.Linear(1, hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, input_dim, hidden_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Linear(hidden_dim, 1)   # attention pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, G)  G = num genes/features
        B, G = x.shape
        tokens = self.token_proj(x.unsqueeze(-1))     # (B, G, hidden_dim)
        tokens = tokens + self.pos_encoding[:, :G, :]  # positional encoding
        tokens = self.encoder(tokens)                  # (B, G, hidden_dim)
        # Attention-weighted pooling
        weights = torch.softmax(self.pool(tokens), dim=1)  # (B, G, 1)
        pooled = (tokens * weights).sum(dim=1)             # (B, hidden_dim)
        out = self.head(pooled)                            # (B, out_dim)
        return out


# ---------------------------------------------------------------------------
# Clinical Feature Encoder
# ---------------------------------------------------------------------------

class ClinicalEncoder(nn.Module):
    """Simple MLP encoder for tabular clinical features."""

    def __init__(self, input_dim: int = 7, out_dim: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between imaging, genomics, and clinical tokens.
    Each modality attends to the others, then outputs are concatenated
    and projected.
    """

    def __init__(self, img_dim: int = 768, gen_dim: int = 768,
                 clin_dim: int = 50, fused_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project all to common key/value dim
        common_dim = fused_dim
        self.img_proj  = nn.Linear(img_dim,  common_dim)
        self.gen_proj  = nn.Linear(gen_dim,  common_dim)
        self.clin_proj = nn.Linear(clin_dim, common_dim)

        self.cross_img_gen   = nn.MultiheadAttention(common_dim, num_heads,
                                                      dropout=dropout,
                                                      batch_first=True)
        self.cross_gen_img   = nn.MultiheadAttention(common_dim, num_heads,
                                                      dropout=dropout,
                                                      batch_first=True)
        self.cross_clin_all  = nn.MultiheadAttention(common_dim, num_heads,
                                                      dropout=dropout,
                                                      batch_first=True)

        self.norm1 = nn.LayerNorm(common_dim)
        self.norm2 = nn.LayerNorm(common_dim)
        self.norm3 = nn.LayerNorm(common_dim)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(common_dim * 3, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, img: torch.Tensor, gen: torch.Tensor,
                clin: torch.Tensor) -> torch.Tensor:
        # Unsqueeze to add sequence dim (length 1 per modality)
        img_t  = self.img_proj(img).unsqueeze(1)    # (B, 1, D)
        gen_t  = self.gen_proj(gen).unsqueeze(1)    # (B, 1, D)
        clin_t = self.clin_proj(clin).unsqueeze(1)  # (B, 1, D)

        # Cross-attention: each modality attends to others
        img_out,  _ = self.cross_img_gen(img_t,  gen_t,  gen_t)
        gen_out,  _ = self.cross_gen_img(gen_t,  img_t,  img_t)
        # Clinical attends to both imaging and genomics
        kv = torch.cat([img_t, gen_t], dim=1)       # (B, 2, D)
        clin_out, _ = self.cross_clin_all(clin_t, kv, kv)

        img_out  = self.norm1(img_t  + img_out)
        gen_out  = self.norm2(gen_t  + gen_out)
        clin_out = self.norm3(clin_t + clin_out)

        fused = torch.cat([img_out.squeeze(1),
                           gen_out.squeeze(1),
                           clin_out.squeeze(1)], dim=-1)   # (B, 3*D)
        return self.fusion_mlp(fused)                       # (B, fused_dim)


# ---------------------------------------------------------------------------
# Shared Dense Layers + Prediction Heads
# ---------------------------------------------------------------------------

class SharedDenseLayers(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256,
                 dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class FusedSwin3DNet(nn.Module):
    """
    Full multimodal network:
      CT  → 3D-Swin → 768-dim
      RNA → Genomics Transformer → 768-dim
      Clin → MLP → 50-dim
         ↓
      Cross-Attention Fusion → 512-dim
         ↓
      Shared Dense Layers → 256-dim
         ↓
      ┌──────────┬────────────┬───────────┐
   Metastasis  Survival     Sites       Clinical Decision
   (1 logit)  (1 scalar)  (4 logits)  (optional)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # Imaging branch
        self.use_imaging  = cfg.get("use_imaging",  True)
        self.use_genomics = cfg.get("use_genomics", True)
        self.use_clinical = cfg.get("use_clinical", True)

        img_out = 768
        gen_out = 768
        clin_out = 50

        if self.use_imaging:
            self.imaging_encoder = SwinTransformer3D(
                in_channels=1,
                patch_size=cfg.get("swin_patch_size", 4),
                embed_dim=cfg.get("swin_embed_dim", 96),
                depths=tuple(cfg.get("swin_depths", [2, 2, 2])),
                num_heads=tuple(cfg.get("swin_heads", [3, 6, 12])),
                out_dim=img_out,
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            self.imaging_dummy = nn.Parameter(torch.zeros(1, img_out),
                                              requires_grad=False)

        if self.use_genomics:
            self.genomics_encoder = GenomicsTransformer(
                input_dim=cfg.get("genomics_input_dim", 500),
                hidden_dim=cfg.get("genomics_hidden_dim", 256),
                out_dim=gen_out,
                num_heads=cfg.get("genomics_heads", 8),
                num_layers=cfg.get("genomics_layers", 4),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            self.genomics_dummy = nn.Parameter(torch.zeros(1, gen_out),
                                               requires_grad=False)

        if self.use_clinical:
            self.clinical_encoder = ClinicalEncoder(
                input_dim=cfg.get("clinical_input_dim", 7),
                out_dim=clin_out,
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            self.clinical_dummy = nn.Parameter(torch.zeros(1, clin_out),
                                               requires_grad=False)

        fused_dim = cfg.get("fused_dim", 512)
        self.fusion = CrossAttentionFusion(
            img_dim=img_out, gen_dim=gen_out, clin_dim=clin_out,
            fused_dim=fused_dim,
            num_heads=cfg.get("fusion_heads", 8),
            dropout=cfg.get("dropout", 0.1),
        )

        shared_hidden = cfg.get("shared_hidden", 256)
        self.shared = SharedDenseLayers(
            in_dim=fused_dim,
            hidden_dim=shared_hidden,
            dropout=cfg.get("dropout", 0.15),
        )

        # Output heads
        H = self.shared.out_dim
        self.met_head      = nn.Linear(H, 1)      # Metastasis
        self.surv_head     = nn.Linear(H, 1)      # Survival risk
        self.site_head     = nn.Linear(H, 4)      # lung/bone/liver/brain
        self.decision_head = nn.Linear(H, 3)      # clinical decision classes

    def forward(self, clinical: Optional[torch.Tensor] = None,
                genomics: Optional[torch.Tensor]  = None,
                imaging:  Optional[torch.Tensor]  = None,
                **kwargs) -> dict:
        B = 1
        if clinical is not None:
            B = clinical.shape[0]
        elif genomics is not None:
            B = genomics.shape[0]
        elif imaging is not None:
            B = imaging.shape[0]

        device = next(self.parameters()).device

        # Encode each modality
        if self.use_imaging and imaging is not None:
            img_feat = self.imaging_encoder(imaging)
        else:
            img_feat = self.imaging_dummy.expand(B, -1).to(device)

        if self.use_genomics and genomics is not None:
            gen_feat = self.genomics_encoder(genomics)
        else:
            gen_feat = self.genomics_dummy.expand(B, -1).to(device)

        if self.use_clinical and clinical is not None:
            clin_feat = self.clinical_encoder(clinical)
        else:
            clin_feat = self.clinical_dummy.expand(B, -1).to(device)

        fused   = self.fusion(img_feat, gen_feat, clin_feat)
        shared  = self.shared(fused)

        return {
            "metastasis_logit": self.met_head(shared).squeeze(-1),
            "survival_risk":    self.surv_head(shared).squeeze(-1),
            "site_logits":      self.site_head(shared),
            "decision_logits":  self.decision_head(shared),
        }
