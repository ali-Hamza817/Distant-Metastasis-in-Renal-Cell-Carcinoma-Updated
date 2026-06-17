import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def custom_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def run_diagnostics():
    cct_path = Path("e:/rcc/RNA Sequences/HS_CPTAC_CCRCC_RNAseq_fpkm_log2_Tumor.cct")
    cli_path = Path("e:/rcc/RNA Sequences/HS_CPTAC_CCRCC_CLI.tsi")

    print("="*60)
    print("STEP 1 & 3: ID ALIGNMENT AND LABEL DISTRIBUTION")
    print("="*60)

    # Load CCT
    rna_df = pd.read_csv(cct_path, sep="\t")
    if "gene" in rna_df.columns:
        rna_df = rna_df.set_index("gene")
    rna_df = rna_df.T
    
    print(f"RNA-seq shape: {rna_df.shape} (Patients, Genes)")
    print(f"First 5 RNA-seq IDs: {rna_df.index[:5].tolist()}")

    # Load CLI
    cli_df = pd.read_csv(cli_path, sep="\t")
    # Skip the type row (NUM, BIN, CAT...)
    cli_df = cli_df.iloc[1:]
    cli_df = cli_df.set_index("Case_ID")

    print(f"\nClinical shape: {cli_df.shape}")
    print(f"First 5 Clinical IDs: {cli_df.index[:5].tolist()}")

    # Check alignment
    common_ids = rna_df.index.intersection(cli_df.index)
    print(f"\nMatching IDs between RNA and CLI: {len(common_ids)}")

    # Extract target: Stage IV as Metastasis (1) vs others (0)
    target_col = "Tumor_Stage_Pathological"
    aligned_cli = cli_df.loc[common_ids]
    
    # Check distribution of stages
    print("\nStage distribution in aligned cohort:")
    print(aligned_cli[target_col].value_counts(dropna=False))

    # Define metastasis
    y = (aligned_cli[target_col] == "Stage IV").astype(int).values
    print(f"\nMetastasis=1 count: {sum(y)} out of {len(y)} ({sum(y)/len(y)*100:.1f}%)")

    # Filter RNA to aligned
    X = rna_df.loc[common_ids].values

    print("\n"+"="*60)
    print("STEP 2 & 4: SANITY CHECK BASELINES (NO FILTERING, NO PCA)")
    print("="*60)

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    models = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
        "Dummy (Prior)": DummyClassifier(strategy="prior", random_state=42),
        "LogReg (L2 penalty)": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    for name, model in models.items():
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            
            # predict_proba for class 1
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test)
                
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        roc = roc_auc_score(all_y_true, all_y_pred)
        pr = custom_pr_auc(all_y_true, all_y_pred)
        brier = brier_score_loss(all_y_true, all_y_pred)
        
        print(f"\nModel: {name}")
        print(f"  ROC-AUC: {roc:.3f}")
        print(f"  PR-AUC:  {pr:.3f}")
        print(f"  Brier:   {brier:.3f}")

if __name__ == "__main__":
    run_diagnostics()
