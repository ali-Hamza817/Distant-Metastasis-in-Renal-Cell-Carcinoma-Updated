"""Evaluation metrics for MMRCCNet."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.3) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "ap": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return out


def concordance_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index for survival."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk = np.asarray(risk, dtype=float)
    concordant = discordant = tied = 0
    n = len(times)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied += 1
            elif times[i] == times[j] and events[j] == 1:
                tied += 1
    total = concordant + discordant + tied
    return float(concordant / total) if total > 0 else 0.5


def multi_site_auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Per-site AUC for lung/bone/liver/brain metastasis."""
    site_names = ["lung", "bone", "liver", "brain"]
    results = {}
    for i, name in enumerate(site_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) > 1:
            results[f"{name}_auc"] = float(roc_auc_score(yt, yp))
        else:
            results[f"{name}_auc"] = float("nan")
    valid = [v for v in results.values() if not np.isnan(v)]
    results["mean_site_auc"] = float(np.mean(valid)) if valid else 0.0
    return results
