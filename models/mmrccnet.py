"""MMRCCNet: Multi-Modal Renal Cell Carcinoma Network with cross-attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImagingBranch(nn.Module):
    """Lightweight 2D CNN for single-slice or projected CT."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv(x).flatten(1)
        return self.fc(h)


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention across modality token embeddings."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, M, D); mask: (B, M) True = ignore
        key_padding = mask if mask is not None else None
        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding)
        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class MMRCCNet(nn.Module):
    """
    Four-branch multimodal network:
      1. Imaging (CT)
      2. Radiomics
      3. Clinical (SEER tabular)
      4. Genomics (RNA-seq)
    """

    def __init__(
        self,
        clinical_dim: int = 7,
        radiomics_dim: int = 40,
        genomics_dim: int = 500,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_imaging: bool = True,
        use_radiomics: bool = True,
        use_clinical: bool = True,
        use_genomics: bool = True,
    ):
        super().__init__()
        self.use_imaging = use_imaging
        self.use_radiomics = use_radiomics
        self.use_clinical = use_clinical
        self.use_genomics = use_genomics
        self.hidden_dim = hidden_dim

        if use_imaging:
            self.imaging_branch = ImagingBranch(hidden_dim)
        if use_radiomics:
            self.radiomics_branch = ModalityEncoder(radiomics_dim, hidden_dim, dropout)
        if use_clinical:
            self.clinical_branch = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        if use_genomics:
            self.genomics_branch = ModalityEncoder(genomics_dim, hidden_dim, dropout)

        self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.metastasis_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _active_branches(self) -> list[str]:
        branches = []
        if self.use_imaging:
            branches.append("imaging")
        if self.use_radiomics:
            branches.append("radiomics")
        if self.use_clinical:
            branches.append("clinical")
        if self.use_genomics:
            branches.append("genomics")
        return branches

    def encode_modalities(
        self,
        clinical: torch.Tensor | None = None,
        radiomics: torch.Tensor | None = None,
        imaging: torch.Tensor | None = None,
        genomics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for name in self._active_branches():
            if name == "clinical":
                if clinical is None:
                    raise ValueError("clinical tensor required when clinical branch is active")
                tokens.append(self.clinical_branch(clinical))
            elif name == "radiomics":
                if radiomics is None:
                    raise ValueError("radiomics tensor required when radiomics branch is active")
                tokens.append(self.radiomics_branch(radiomics))
            elif name == "imaging":
                if imaging is None:
                    raise ValueError("imaging tensor required when imaging branch is active")
                tokens.append(self.imaging_branch(imaging))
            elif name == "genomics":
                if genomics is None:
                    raise ValueError("genomics tensor required when genomics branch is active")
                tokens.append(self.genomics_branch(genomics))
        if not tokens:
            raise ValueError("At least one modality branch must be active")
        stacked = torch.stack(tokens, dim=1)  # (B, M, D)
        fused = self.fusion(stacked)
        pooled = fused.mean(dim=1)
        return fused, pooled

    def forward(
        self,
        clinical: torch.Tensor | None = None,
        radiomics: torch.Tensor | None = None,
        imaging: torch.Tensor | None = None,
        genomics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, pooled = self.encode_modalities(clinical, radiomics, imaging, genomics)
        return {
            "metastasis_logit": self.metastasis_head(pooled).squeeze(-1),
            "site_logits": self.site_head(pooled),
            "survival_risk": self.survival_head(pooled).squeeze(-1),
            "embedding": pooled,
        }

    def get_imaging_gradcam_target(self) -> nn.Module:
        return self.imaging_branch.conv[-2]  # last conv before pool
