"""Benchmark classical + MLP models on SEER RCC metastasis prediction."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "seer_rcc_2010_2018_clean.csv"
OUT = ROOT / "runs" / "model_benchmark.json"

FEATURES = ["age", "sex", "t_stage", "n_stage", "grade", "histology_enc", "tumor_size_cm"]
LABEL = "metastasis"
SEED = 42


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = df[LABEL].values
    pos_weight = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, random_state=SEED, verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=SEED,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=200,
                early_stopping=True, random_state=SEED,
            )),
        ]),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            random_state=SEED,
        ),
    }

    results = []
    for name, model in models.items():
        if name == "LightGBM":
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        results.append({"model": name, "auc": float(auc)})
        print(f"{name:25s}  AUC = {auc:.4f}")

    results.sort(key=lambda r: r["auc"], reverse=True)
    best = results[0]

    report = {
        "dataset": str(DATA),
        "n_samples": len(df),
        "features": FEATURES,
        "label": LABEL,
        "test_size": 0.15,
        "results_ranked": results,
        "best_model": best["model"],
        "best_auc": best["auc"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest model: {best['model']}  (AUC = {best['auc']:.4f})")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
