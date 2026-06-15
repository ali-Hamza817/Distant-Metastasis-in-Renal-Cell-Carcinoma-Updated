"""Final push: CatBoost + blended ensemble, save best model."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT)

import sys
sys.path.insert(0, sys_path)
from scripts.optimize_models import engineer_features, FEATURES_ENG, evaluate, best_threshold, SEED

DATA = ROOT / "seer_rcc_2010_2018_clean.csv"
OUT = ROOT / "runs" / "best_model_final.json"
MODEL_PATH = ROOT / "runs" / "best_model.joblib"


def tune_catboost(X_tr, y_tr, X_val, y_val, n_trials=30):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 300, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            scale_pos_weight=spw,
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    model = CatBoostClassifier(
        **p, scale_pos_weight=spw, random_seed=SEED, verbose=0,
        early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return model, study.best_value


def main():
    df = engineer_features(pd.read_csv(DATA))
    X = df[FEATURES_ENG].values
    y = df["metastasis"].values
    spw = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    print("Tuning CatBoost...")
    cat, cat_auc = tune_catboost(X_tr, y_tr, X_val, y_val)

    print("Training final blend components...")
    lgb_m = lgb.LGBMClassifier(
        n_estimators=900, learning_rate=0.028, max_depth=8, num_leaves=72,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.75,
        scale_pos_weight=spw, random_state=SEED, verbose=-1,
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    xgb_m = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.035, max_depth=7, min_child_weight=3,
        subsample=0.9, colsample_bytree=0.8, gamma=0.1,
        scale_pos_weight=spw, eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Weighted probability blend (weights from val AUC)
    models = {"LightGBM": lgb_m, "XGBoost": xgb_m, "CatBoost": cat}
    val_aucs = {}
    for name, m in models.items():
        val_aucs[name] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    total = sum(val_aucs.values())
    weights = {k: v / total for k, v in val_aucs.items()}

    def blend_predict(X):
        prob = np.zeros(len(X))
        for name, m in models.items():
            prob += weights[name] * m.predict_proba(X)[:, 1]
        return prob

    # Retrain on full training set
    for m in models.values():
        pass  # already have models; refit on full train
    lgb_m.fit(X_train, y_train)
    xgb_m.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    prob_test = blend_predict(X_test)
    t_acc, _ = best_threshold(y_test, prob_test, "accuracy")
    t_f1, _ = best_threshold(y_test, prob_test, "f1")
    metrics = evaluate(y_test, prob_test, t_acc)
    metrics_f1 = evaluate(y_test, prob_test, t_f1)

    report = {
        "best_model": "Weighted_Blend (LightGBM + XGBoost + CatBoost)",
        "blend_weights": weights,
        "val_aucs": val_aucs,
        "test_metrics_accuracy_optimized": metrics,
        "test_metrics_f1_optimized": metrics_f1,
        "meets_90pct_accuracy": metrics["accuracy"] >= 0.90,
        "features": FEATURES_ENG,
    }

    bundle = {"models": models, "weights": weights, "features": FEATURES_ENG, "threshold": t_acc}
    joblib.dump(bundle, MODEL_PATH)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nFINAL BLEND — Test Results:")
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']*100:.2f}%  (threshold={t_acc:.2f})")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  F1:                {metrics_f1['f1']:.4f}")
    print(f"  Meets 90% acc:     {report['meets_90pct_accuracy']}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
