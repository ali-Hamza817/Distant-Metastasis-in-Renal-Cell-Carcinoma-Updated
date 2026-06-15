"""PHASE 1 — LightGBM baseline for metastasis prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG
from utils.stats_validation import cv_summary, delong_test, save_validation_report


FEATURES = CONFIG["clinical_features"]
LABEL = CONFIG["label_col"]


def train_and_evaluate(df: pd.DataFrame, train_idx, test_idx, seed: int = 42) -> dict:
    X_train = df.iloc[train_idx][FEATURES]
    y_train = df.iloc[train_idx][LABEL]
    X_test = df.iloc[test_idx][FEATURES]
    y_test = df.iloc[test_idx][LABEL]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    return {"model": model, "y_test": y_test.values, "prob": prob, "auc": float(auc)}


def run_cv(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> dict:
    X = df[FEATURES]
    y = df[LABEL]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        result = train_and_evaluate(df, tr, te, seed=seed + fold)
        fold_aucs.append(result["auc"])
        print(f"  Fold {fold + 1}: AUC = {result['auc']:.4f}")
    return cv_summary(fold_aucs)


def main():
    out_dir = Path(CONFIG["runs_dir"]) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CONFIG["seer_csv"])
    print("=" * 60)
    print("PHASE 1 — LightGBM Baseline")
    print("=" * 60)
    print(f"Samples: {len(df):,} | Features: {FEATURES} | Label: {LABEL}")
    print(f"Positive rate: {df[LABEL].mean()*100:.2f}%")

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=CONFIG["test_ratio"], stratify=df[LABEL], random_state=CONFIG["seed"]
    )

    print("\nTraining hold-out model...")
    result = train_and_evaluate(df, train_idx, test_idx, seed=CONFIG["seed"])
    print(f"Hold-out Test AUC: {result['auc']:.4f}")

    print(f"\n{CONFIG['n_folds']}-fold cross-validation:")
    cv_result = run_cv(df, n_folds=CONFIG["n_folds"], seed=CONFIG["seed"])
    print(f"CV AUC: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": result["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    report = {
        "model": "LightGBM",
        "phase": 1,
        "holdout_auc": result["auc"],
        "cv_auc_mean": cv_result["mean"],
        "cv_auc_std": cv_result["std"],
        "cv_folds": cv_result["folds"],
        "features": FEATURES,
        "label": LABEL,
        "n_samples": len(df),
        "pos_weight": CONFIG["pos_weight"],
        "feature_importance": imp.to_dict(orient="records"),
        "table1_entry": {
            "Method": "LightGBM (SEER clinical)",
            "AUC": f"{result['auc']:.3f}",
            "CV AUC": f"{cv_result['mean']:.3f} ± {cv_result['std']:.3f}",
        },
    }

    save_validation_report(report, out_dir / "baseline_results.json")
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    result["model"].booster_.save_model(str(out_dir / "lightgbm_model.txt"))

    # LaTeX row for Table 1
    latex = (
        f"LightGBM (clinical) & {result['auc']:.3f} & "
        f"{cv_result['mean']:.3f}$\\pm${cv_result['std']:.3f} \\\\\n"
    )
    (out_dir / "table1_row.tex").write_text(latex, encoding="utf-8")

    print(f"\nResults saved to {out_dir}")
    print(f"Table 1 entry: AUC = {result['auc']:.4f}")
    return report


if __name__ == "__main__":
    main()
