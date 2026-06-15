"""Training loop and loss functions for MMRCCNet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import binary_metrics, concordance_index, multi_site_auc


class MMRCCNetLoss(nn.Module):
    def __init__(self, pos_weight: float = 15.67, site_weight: float = 1.0, survival_weight: float = 0.5):
        super().__init__()
        self.met_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.site_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.survival_loss = nn.MSELoss()
        self.site_weight = site_weight
        self.survival_weight = survival_weight

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        met = self.met_loss(outputs["metastasis_logit"], batch["metastasis"])
        site = self.site_loss(outputs["site_logits"], batch["sites"])
        surv = self.survival_loss(outputs["survival_risk"], batch["survival"] / 100.0)
        total = met + self.site_weight * site + self.survival_weight * surv
        return total, {"met": met.item(), "site": site.item(), "surv": surv.item()}


def _forward_batch(model, batch, use_imaging, use_radiomics, use_clinical, use_genomics):
    kwargs = {}
    if use_clinical:
        kwargs["clinical"] = batch["clinical"]
    if use_radiomics and "radiomics" in batch:
        kwargs["radiomics"] = batch["radiomics"]
    if use_imaging and "imaging" in batch:
        kwargs["imaging"] = batch["imaging"]
    if use_genomics and "genomics" in batch:
        kwargs["genomics"] = batch["genomics"]
    return model(**kwargs)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    use_imaging: bool,
    use_radiomics: bool,
    use_clinical: bool,
    use_genomics: bool,
) -> dict:
    model.eval()
    met_probs, met_true = [], []
    site_probs, site_true = [], []
    surv_risk, surv_time, surv_event = [], [], []

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = _forward_batch(model, batch, use_imaging, use_radiomics, use_clinical, use_genomics)
        met_probs.append(torch.sigmoid(out["metastasis_logit"]).cpu().numpy())
        met_true.append(batch["metastasis"].cpu().numpy())
        site_probs.append(torch.sigmoid(out["site_logits"]).cpu().numpy())
        site_true.append(batch["sites"].cpu().numpy())
        surv_risk.append(out["survival_risk"].cpu().numpy())
        surv_time.append(batch["survival"].cpu().numpy())
        surv_event.append(batch["event"].cpu().numpy())

    met_probs = np.concatenate(met_probs)
    met_true = np.concatenate(met_true)
    site_probs = np.concatenate(site_probs)
    site_true = np.concatenate(site_true)
    surv_risk = np.concatenate(surv_risk)
    surv_time = np.concatenate(surv_time)
    surv_event = np.concatenate(surv_event)

    metrics = binary_metrics(met_true, met_probs)
    metrics.update(multi_site_auc(site_true, site_probs))
    metrics["c_index"] = concordance_index(surv_time, surv_event, surv_risk)
    return metrics


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
    use_imaging: bool = False,
    use_radiomics: bool = False,
    use_clinical: bool = True,
    use_genomics: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = MMRCCNetLoss(pos_weight=config["pos_weight"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            out = _forward_batch(model, batch, use_imaging, use_radiomics, use_clinical, use_genomics)
            loss, _ = criterion(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(
            model, val_loader, device, use_imaging, use_radiomics, use_clinical, use_genomics
        )
        scheduler.step(1 - val_metrics["auc"])
        history.append({"epoch": epoch, "loss": epoch_loss / len(train_loader), **val_metrics})

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "config": config, "metrics": val_metrics},
                output_dir / "best_checkpoint.pt",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={epoch_loss/len(train_loader):.4f} val_auc={val_metrics['auc']:.4f}")

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"best_auc": best_auc, "history": history}
