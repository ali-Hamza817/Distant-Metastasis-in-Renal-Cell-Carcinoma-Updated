"""Statistical validation: DeLong test, cross-validation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """
    DeLong's test for comparing two correlated ROC AUCs.
    Implementation based on Sun & Xu (2014) / fastDeLong approach.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)

    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        return {"auc_a": 0.5, "auc_b": 0.5, "z": 0.0, "p_value": 1.0}

    def _structural_components(probs):
        v10 = np.array([np.mean(probs[pos] > p) for p in probs[neg]])
        v01 = np.array([np.mean(probs[neg] < p) for p in probs[pos]])
        return v10, v01

    v10_a, v01_a = _structural_components(prob_a)
    v10_b, v01_b = _structural_components(prob_b)

    auc_a = float(np.mean(v01_a))
    auc_b = float(np.mean(v01_b))
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / m + s01 / n
    diff = auc_a - auc_b
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": 0.0, "p_value": 1.0}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": float(z), "p_value": p}


def cv_summary(fold_scores: list[float]) -> dict:
    arr = np.asarray(fold_scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci_95_low": float(arr.mean() - 1.96 * arr.std() / np.sqrt(len(arr))),
        "ci_95_high": float(arr.mean() + 1.96 * arr.std() / np.sqrt(len(arr))),
        "folds": [float(x) for x in arr],
        "n_folds": len(arr),
    }


def stratified_kfold_indices(y: np.ndarray, n_folds: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def save_validation_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
