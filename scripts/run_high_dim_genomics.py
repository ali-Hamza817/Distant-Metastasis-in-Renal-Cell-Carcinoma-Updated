import argparse
import json
import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_cct(filepath: Path) -> pd.DataFrame:
    """Load the .cct file into a pandas DataFrame."""
    logging.info(f"Loading CCT file from {filepath}")
    df = pd.read_csv(filepath, sep="\t")
    # Set the gene column as the index
    if "gene" in df.columns:
        df = df.set_index("gene")
    # Transpose so rows are patients and columns are genes
    df = df.T
    logging.info(f"Loaded RNA-seq matrix with shape {df.shape}")
    return df


def load_cptac_clinical(filepath: Path, rna_index: pd.Index) -> pd.DataFrame:
    """Load true CPTAC clinical labels and align with RNA-seq data."""
    logging.info(f"Loading CPTAC Clinical file from {filepath}")
    df = pd.read_csv(filepath, sep="\t")
    # Skip the type descriptor row
    df = df.iloc[1:]
    if "Case_ID" in df.columns:
        df = df.set_index("Case_ID")
    
    # Keep only patients present in RNA-seq
    common_ids = rna_index.intersection(df.index)
    logging.info(f"Found {len(common_ids)} patients with both RNA-seq and Clinical data.")
    
    aligned_df = df.loc[common_ids]
    
    # Metastasis is defined as Stage IV
    targets = (aligned_df["Tumor_Stage_Pathological"] == "Stage IV").astype(int)
    
    matched_df = pd.DataFrame({
        "metastasis": targets
    }, index=common_ids)
    
    return matched_df


def pre_filter_genes(X_train: np.ndarray, X_test: np.ndarray, n_keep: int = 1000):
    """Keep the top n_keep most variable genes based on training data only."""
    variances = np.var(X_train, axis=0)
    top_indices = np.argsort(variances)[-n_keep:]
    return X_train[:, top_indices], X_test[:, top_indices]


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence interval for a metric."""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except Exception:
            pass
    if not bootstrapped_scores:
        return np.nan, np.nan
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    return lower, upper


def custom_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def custom_recall_at_specificity(y_true, y_pred, target_specificity=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificities = 1 - fpr
    # Find the threshold closest to the target specificity, but ensuring it's >= target
    idx = np.where(specificities >= target_specificity)[0]
    if len(idx) > 0:
        return tpr[idx[-1]] # Take the recall at the lowest FPR that meets the condition
    return 0.0

def custom_f1(y_true, y_pred):
    return f1_score(y_true, (y_pred >= 0.5).astype(int))

def train_and_evaluate(X, y, model, name):
    logging.info(f"Starting Repeated Stratified 5-Fold CV for {name}...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    metrics = {
        "pr_auc": [],
        "roc_auc": [],
        "brier": [],
        "recall_at_90_spec": [],
        "recall_at_95_spec": [],
        "f1": []
    }
    
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Variance filtering based on X_train only
        X_train, X_test = pre_filter_genes(X_train, X_test, n_keep=1000)

        # 2. Scaling based on X_train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate fold metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics["pr_auc"].append(auc(recall, precision))
        metrics["roc_auc"].append(roc_auc_score(y_test, y_pred_proba))
        metrics["brier"].append(brier_score_loss(y_test, y_pred_proba))
        metrics["recall_at_90_spec"].append(custom_recall_at_specificity(y_test, y_pred_proba, 0.90))
        metrics["recall_at_95_spec"].append(custom_recall_at_specificity(y_test, y_pred_proba, 0.95))
        metrics["f1"].append(f1_score(y_test, (y_pred_proba >= 0.5).astype(int)))
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_proba)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    logging.info(f"Calculating bootstrap CIs for {name}...")
    results = {}
    
    # Calculate means and CIs
    for metric_name, func in [
        ("PR-AUC", custom_pr_auc),
        ("ROC-AUC", roc_auc_score),
        ("Brier Score", brier_score_loss),
        ("Recall @ 90% Spec", lambda yt, yp: custom_recall_at_specificity(yt, yp, 0.90)),
        ("Recall @ 95% Spec", lambda yt, yp: custom_recall_at_specificity(yt, yp, 0.95)),
        ("F1 Score", custom_f1)
    ]:
        mean_val = func(all_y_true, all_y_pred)
        lower, upper = bootstrap_ci(all_y_true, all_y_pred, func)
        results[metric_name] = {"mean": mean_val, "ci_lower": lower, "ci_upper": upper}
        logging.info(f"  {metric_name}: {mean_val:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

    return results, all_y_true, all_y_pred


def plot_results(y_true_en, y_pred_en, y_true_lgb, y_pred_lgb, out_dir):
    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr_en, tpr_en, _ = roc_curve(y_true_en, y_pred_en)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_true_lgb, y_pred_lgb)
    plt.plot(fpr_en, tpr_en, label=f"Elastic Net (AUC = {roc_auc_score(y_true_en, y_pred_en):.3f})")
    plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC = {roc_auc_score(y_true_lgb, y_pred_lgb):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # PR Curve
    plt.subplot(1, 2, 2)
    prec_en, rec_en, _ = precision_recall_curve(y_true_en, y_pred_en)
    prec_lgb, rec_lgb, _ = precision_recall_curve(y_true_lgb, y_pred_lgb)
    plt.plot(rec_en, prec_en, label=f"Elastic Net (AUC = {custom_pr_auc(y_true_en, y_pred_en):.3f})")
    plt.plot(rec_lgb, prec_lgb, label=f"LightGBM (AUC = {custom_pr_auc(y_true_lgb, y_pred_lgb):.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "genomics_evaluation_curves.png")
    logging.info(f"Saved evaluation curves to {out_dir / 'genomics_evaluation_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="High-Dimensional Genomics Pipeline")
    parser.add_argument("--cct", type=str, default="e:/rcc/RNA Sequences/HS_CPTAC_CCRCC_RNAseq_fpkm_log2_Tumor.cct")
    parser.add_argument("--cli", type=str, default="e:/rcc/RNA Sequences/HS_CPTAC_CCRCC_CLI.tsi")
    parser.add_argument("--target", type=str, default="metastasis")
    parser.add_argument("--outdir", type=str, default="e:/rcc/results/genomics_only")
    parser.add_argument("--n_genes", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    rna_df = load_cct(Path(args.cct))
    
    # 2. Extract true clinical labels for CPTAC
    matched_df = load_cptac_clinical(Path(args.cli), rna_df.index)
    
    # Join with RNA-seq features
    merged_df = rna_df.join(matched_df, how="inner")
    
    # Extract features and labels
    X = merged_df.drop(columns=[args.target]).values
    y = merged_df[args.target].values
    
    logging.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
    logging.info("Preprocessing (variance filtering & scaling) will be strictly performed within CV folds to prevent data leakage.")

    # 5. Define Models
    elastic_net = LogisticRegression(
        penalty='elasticnet', 
        solver='saga', 
        l1_ratio=0.5, 
        C=1.0, 
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    
    lightgbm = lgb.LGBMClassifier(
        max_depth=3,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        class_weight='balanced'
    )

    # 6. Train and Evaluate
    en_results, yt_en, yp_en = train_and_evaluate(X, y, elastic_net, "Elastic Net")
    lgb_results, yt_lgb, yp_lgb = train_and_evaluate(X, y, lightgbm, "LightGBM")

    # 7. Save Results
    final_report = {
        "dataset_info": {
            "n_samples": len(y),
            "n_genes_raw": X.shape[1],
            "n_genes_filtered": args.n_genes,
            "n_positive": int(sum(y))
        },
        "results": {
            "elastic_net": en_results,
            "lightgbm": lgb_results
        }
    }
    
    with open(out_dir / "genomics_validation_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
        
    logging.info(f"Saved validation report to {out_dir / 'genomics_validation_report.json'}")

    # 8. Plot Curves
    plot_results(yt_en, yp_en, yt_lgb, yp_lgb, out_dir)
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
