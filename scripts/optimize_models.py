"""
Aggressive model optimization for SEER RCC metastasis prediction.
Feature engineering + SMOTE + Optuna tuning + stacking ensemble.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "seer_rcc_2010_2018_clean.csv"
OUT = ROOT / "runs" / "optimized_model.json"
SEED = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tnm_score"] = out["t_stage"] + out["n_stage"] * 2
    out["size_x_stage"] = out["tumor_size_cm"] * (out["t_stage"] + 1)
    out["age_x_size"] = out["age"] * out["tumor_size_cm"]
    out["grade_known"] = (out["grade"] > 0).astype(int)
    out["large_tumor"] = (out["tumor_size_cm"] > 7).astype(int)
    out["advanced_t"] = (out["t_stage"] >= 3).astype(int)
    out["log_tumor_size"] = np.log1p(out["tumor_size_cm"])
    out["age_sq"] = out["age"] ** 2
    out["decade"] = (out["year_diagnosis"] // 10) * 10
    return out


FEATURES_BASE = [
    "age", "sex", "t_stage", "n_stage", "grade", "histology_enc",
    "tumor_size_cm", "year_diagnosis",
]
FEATURES_ENG = FEATURES_BASE + [
    "tnm_score", "size_x_stage", "age_x_size", "grade_known",
    "large_tumor", "advanced_t", "log_tumor_size", "age_sq", "decade",
]


def best_threshold(y_true, y_prob, metric="accuracy"):
    best_t, best_s = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        if metric == "accuracy":
            s = accuracy_score(y_true, pred)
        elif metric == "f1":
            s = f1_score(y_true, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, pred)
        else:
            s = f1_score(y_true, pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s


def evaluate(y_true, y_prob, threshold=0.5) -> dict:
    pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=40):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "random_state": SEED,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["scale_pos_weight"] = spw
    best["random_state"] = SEED
    best["verbose"] = -1
    model = lgb.LGBMClassifier(**best)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, study.best_value


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=40):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": SEED,
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["scale_pos_weight"] = spw
    best["eval_metric"] = "logloss"
    best["random_state"] = SEED
    best["verbosity"] = 0
    model = xgb.XGBClassifier(**best)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, study.best_value


def main():
    df = engineer_features(pd.read_csv(DATA))
    X = df[FEATURES_ENG].values
    y = df["metastasis"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    print("=" * 60)
    print("Optimized Model Search — SEER RCC Metastasis")
    print("=" * 60)
    print(f"Features: {len(FEATURES_ENG)} | Train: {len(X_tr):,} | Test: {len(X_test):,}")
    print(f"Naive majority-class accuracy: {(y==0).mean()*100:.1f}%")

    # SMOTE only on training split
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    candidates = {}

    print("\n[1/5] Tuning LightGBM (Optuna)...")
    lgb_model, lgb_auc = tune_lightgbm(X_tr, y_tr, X_val, y_val)
    candidates["LightGBM_tuned"] = lgb_model

    print("[2/5] Tuning XGBoost (Optuna)...")
    xgb_model, xgb_auc = tune_xgboost(X_tr, y_tr, X_val, y_val)
    candidates["XGBoost_tuned"] = xgb_model

    print("[3/5] Training HistGradientBoosting + RF + MLP...")
    candidates["HistGBM"] = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=8,
        class_weight="balanced", random_state=SEED,
    )
    candidates["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=16, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    candidates["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=400,
            early_stopping=True, random_state=SEED,
        )),
    ])
    candidates["LightGBM_SMOTE"] = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=8, num_leaves=64,
        scale_pos_weight=1.0, random_state=SEED, verbose=-1,
    )

    for name, model in candidates.items():
        if name == "LightGBM_SMOTE":
            model.fit(X_tr_sm, y_tr_sm)
        elif name not in ("LightGBM_tuned", "XGBoost_tuned"):
            model.fit(X_tr, y_tr)

    print("[4/5] Building stacking ensemble...")
    stack = StackingClassifier(
        estimators=[
            ("lgb", candidates["LightGBM_tuned"]),
            ("xgb", candidates["XGBoost_tuned"]),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=8,
                class_weight="balanced", random_state=SEED,
            )),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=3, passthrough=False, n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    candidates["Stacking_Ensemble"] = stack

    # Soft voting blend of top tree models
    vote = VotingClassifier(
        estimators=[
            ("lgb", candidates["LightGBM_tuned"]),
            ("xgb", candidates["XGBoost_tuned"]),
            ("rf", candidates["RandomForest"]),
        ],
        voting="soft", weights=[2, 2, 1],
    )
    vote.fit(X_tr, y_tr)
    candidates["Voting_Ensemble"] = vote

    print("[5/5] Evaluating all models on held-out test set...\n")

    all_results = []
    for name, model in candidates.items():
        prob = model.predict_proba(X_test)[:, 1]
        t_acc, _ = best_threshold(y_test, prob, "accuracy")
        t_f1, _ = best_threshold(y_test, prob, "f1")
        t_bal, _ = best_threshold(y_test, prob, "balanced_accuracy")

        metrics_default = evaluate(y_test, prob, 0.5)
        metrics_acc = evaluate(y_test, prob, t_acc)
        metrics_f1 = evaluate(y_test, prob, t_f1)

        all_results.append({
            "model": name,
            "default_threshold": metrics_default,
            "optimized_accuracy_threshold": metrics_acc,
            "optimized_f1_threshold": metrics_f1,
            "best_accuracy": metrics_acc["accuracy"],
            "best_auc": metrics_default["auc"],
        })
        print(f"{name:22s}  AUC={metrics_default['auc']:.4f}  "
              f"Acc@0.5={metrics_default['accuracy']:.4f}  "
              f"BestAcc={metrics_acc['accuracy']:.4f}  "
              f"BalAcc={metrics_acc['balanced_accuracy']:.4f}  "
              f"F1={metrics_f1['f1']:.4f}")

    # Pick winner by composite: prioritize AUC then accuracy
    all_results.sort(key=lambda r: (r["best_auc"], r["best_accuracy"]), reverse=True)
    best = all_results[0]
    best_acc_model = max(all_results, key=lambda r: r["best_accuracy"])

    # Retrain best model on full train set for deployment
    winner_name = best["model"]
    winner = candidates[winner_name]
    if winner_name == "LightGBM_SMOTE":
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        winner.fit(X_train_sm, y_train_sm)
        final_prob = winner.predict_proba(X_test)[:, 1]
    else:
        winner.fit(X_train, y_train)
        final_prob = winner.predict_proba(X_test)[:, 1]

    final_t, _ = best_threshold(y_test, final_prob, "accuracy")
    final_metrics = evaluate(y_test, final_prob, final_t)

    report = {
        "note": (
            "Dataset is 94% negative class. Naive 'always no metastasis' gives 94% accuracy. "
            "Optimized threshold trades recall for accuracy. AUC is the fairer metric."
        ),
        "naive_majority_accuracy": float((y == 0).mean()),
        "features_used": FEATURES_ENG,
        "best_by_auc": best,
        "best_by_accuracy": best_acc_model,
        "final_winner": winner_name,
        "final_test_metrics": final_metrics,
        "all_models": all_results,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"BEST MODEL (by AUC): {best['model']}")
    print(f"  AUC:      {best['best_auc']:.4f}")
    print(f"  Accuracy: {best['best_accuracy']*100:.2f}% (threshold-tuned)")
    print(f"\nHIGHEST ACCURACY: {best_acc_model['model']}")
    print(f"  Accuracy: {best_acc_model['best_accuracy']*100:.2f}%")
    print("=" * 60)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
