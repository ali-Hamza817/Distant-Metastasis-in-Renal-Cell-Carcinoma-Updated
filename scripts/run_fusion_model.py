import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def custom_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, ci=95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    lower = np.percentile(bootstrapped_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_scores, 100 - (100 - ci) / 2)
    mean_score = metric_func(y_true, y_pred)
    return {"mean": mean_score, "ci_lower": lower, "ci_upper": upper}

def get_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    
    # Handle possible zero division securely
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        "ROC-AUC": bootstrap_ci(y_true, y_pred, roc_auc_score),
        "PR-AUC": bootstrap_ci(y_true, y_pred, custom_pr_auc),
        "Recall": recall,
        "NPV": npv,
        "Confusion_Matrix": cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="e:/rcc/data/fusion_features.csv")
    parser.add_argument("--outdir", type=str, default="e:/rcc/results/fusion_model")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("e:/rcc/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(args.data)
    features = ["p_clinical", "p_genomic", "p_imaging", "has_genomic", "has_imaging"]
    X = df[features].values
    y = df["metastasis"].values
    
    # 2. Train Fusion Model (Logistic Regression) with 5-Fold CV
    # Using class_weight='balanced' to handle the imbalanced nature of metastasis
    model = LogisticRegression(class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Generate OOF p_final predictions
    p_final = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    df["p_final"] = p_final
    
    # Fit final model on all data to extract feature weights
    model.fit(X, y)
    weights = dict(zip(features, model.coef_[0]))
    logging.info(f"Learned Fusion Weights: {weights}")
    logging.info(f"Learned Bias: {model.intercept_[0]}")
    
    # 3. Evaluate and Compare Metrics
    p_clinical = df["p_clinical"].values
    p_imaging = df["p_imaging"].values
    
    metrics_clinical = get_metrics(y, p_clinical)
    metrics_imaging = get_metrics(y, p_imaging)
    metrics_fusion = get_metrics(y, p_final)
    
    # 4. Risk Tiers
    df["risk_tier"] = pd.cut(df["p_final"], bins=[-np.inf, 0.35, 0.65, np.inf], labels=["Low", "Medium", "High"])
    risk_distribution = df["risk_tier"].value_counts().to_dict()
    
    # 5. Save Report
    report = {
        "dataset_info": {
            "n_samples": len(y),
            "n_positive": int(sum(y))
        },
        "learned_weights": weights,
        "bias": float(model.intercept_[0]),
        "performance_comparison": {
            "Clinical_Baseline": metrics_clinical,
            "Imaging_Baseline": metrics_imaging,
            "Fusion_Model": metrics_fusion
        },
        "risk_tiers": risk_distribution
    }
    
    report_path = out_dir / "fusion_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Saved evaluation report to {report_path}")

    # 6. Plot Comparison Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    from sklearn.metrics import roc_curve
    # ROC Plot
    for name, p_arr in [("Clinical", p_clinical), ("Imaging", p_imaging), ("Fusion", p_final)]:
        fpr, tpr, _ = roc_curve(y, p_arr)
        auc_val = roc_auc_score(y, p_arr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve Comparison")
    axes[0].legend()
    
    # PR Plot
    baseline = sum(y) / len(y)
    for name, p_arr in [("Clinical", p_clinical), ("Imaging", p_imaging), ("Fusion", p_final)]:
        precision, recall, _ = precision_recall_curve(y, p_arr)
        pr_auc_val = custom_pr_auc(y, p_arr)
        axes[1].plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.3f})")
    
    axes[1].axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve Comparison")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "fusion_evaluation_curves.png", dpi=300)
    
    # Save final weights
    import pickle
    with open(models_dir / "model4_fusion.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
