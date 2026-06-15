"""
Exploratory Data Analysis — SEER RCC 2010–2018
Generates summary statistics and saves charts to eda_output/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).parent / "seer_rcc_2010_2018_clean.csv"
OUT_DIR = Path(__file__).parent / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "vital_status": {0: "Alive", 1: "Dead"},
    "metastasis": {0: "No", 1: "Yes"},
    "t_stage": {0: "T0/TX", 1: "T1", 2: "T2", 3: "T3", 4: "T4"},
    "n_stage": {0: "N0", 1: "N1"},
    "grade": {0: "Unknown", 1: "G1", 2: "G2", 3: "G3", 4: "G4"},
    "histology_enc": {0: "Clear cell", 1: "Papillary", 2: "Chromophobe", 3: "Other"},
}


def save_fig(name: str) -> None:
    plt.savefig(OUT_DIR / f"{name}.png")
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- 1. Overview ---
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    overview.to_csv(OUT_DIR / "01_overview.csv")
    df.describe(include="all").T.to_csv(OUT_DIR / "02_descriptive_stats.csv")

    print("=" * 60)
    print("SEER RCC 2010–2018 — Exploratory Data Analysis")
    print("=" * 60)
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nColumn overview:")
    print(overview.to_string())
    print("\nNumeric summary:")
    print(df[numeric_cols].describe().round(2).to_string())

    # --- 2. Age distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["age"], bins=20, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age (years)")
    sns.boxplot(x=df["vital_status"].map(LABELS["vital_status"]),
                y=df["age"], ax=axes[1], hue=df["vital_status"].map(LABELS["vital_status"]),
                legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Age by Vital Status")
    axes[1].set_xlabel("Vital Status")
    plt.tight_layout()
    save_fig("03_age_distribution")

    # --- 3. Sex & vital status ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sex_counts = df["sex"].map(LABELS["sex"]).value_counts()
    axes[0].pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%",
                colors=["#f39c12", "#3498db"], startangle=90)
    axes[0].set_title("Sex Distribution")
    vital_counts = df["vital_status"].map(LABELS["vital_status"]).value_counts()
    sns.barplot(x=vital_counts.index, y=vital_counts.values, ax=axes[1],
                hue=vital_counts.index, legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Vital Status")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(vital_counts.values):
        axes[1].text(i, v + 200, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("04_sex_vital_status")

    # --- 4. Diagnosis year trend ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    year_counts = df["year_diagnosis"].value_counts().sort_index()
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax, color="steelblue")
    ax.fill_between(year_counts.index, year_counts.values, alpha=0.2, color="steelblue")
    ax.set_title("Cases by Year of Diagnosis (2010–2018)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Patients")
    for x, y in zip(year_counts.index, year_counts.values):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    save_fig("05_year_diagnosis")

    # --- 5. Tumor size ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    size_clipped = df["tumor_size_cm"].clip(upper=20)
    sns.histplot(size_clipped, bins=30, kde=True, ax=axes[0], color="coral")
    axes[0].set_title("Tumor Size (cm, clipped at 20)")
    axes[0].set_xlabel("Tumor Size (cm)")
    sns.boxplot(x=df["t_stage"].map(LABELS["t_stage"]), y=size_clipped, ax=axes[1],
                hue=df["t_stage"].map(LABELS["t_stage"]), legend=False)
    axes[1].set_title("Tumor Size by T Stage")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_fig("06_tumor_size")

    # --- 6. Staging ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t_counts = df["t_stage"].map(LABELS["t_stage"]).value_counts().reindex(
        ["T0/TX", "T1", "T2", "T3", "T4"])
    sns.barplot(x=t_counts.index, y=t_counts.values, ax=axes[0],
                hue=t_counts.index, legend=False, palette="Blues_d")
    axes[0].set_title("T Stage Distribution")
    axes[0].tick_params(axis="x", rotation=30)
    n_counts = df["n_stage"].map(LABELS["n_stage"]).value_counts()
    sns.barplot(x=n_counts.index, y=n_counts.values, ax=axes[1],
                hue=n_counts.index, legend=False, palette="Purples_d")
    axes[1].set_title("N Stage Distribution")
    plt.tight_layout()
    save_fig("07_staging")

    # --- 7. Grade & histology ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grade_counts = df["grade"].map(LABELS["grade"]).value_counts().reindex(
        ["Unknown", "G1", "G2", "G3", "G4"])
    sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=axes[0],
                hue=grade_counts.index, legend=False, palette="YlOrRd")
    axes[0].set_title("Tumor Grade")
    axes[0].tick_params(axis="x", rotation=30)
    hist_counts = df["histology_enc"].map(LABELS["histology_enc"]).value_counts()
    sns.barplot(x=hist_counts.index, y=hist_counts.values, ax=axes[1],
                hue=hist_counts.index, legend=False, palette="Set2")
    axes[1].set_title("Histology")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    save_fig("08_grade_histology")

    # --- 8. Metastasis ---
    met_cols = ["metastasis", "lung_met", "bone_met", "liver_met", "brain_met"]
    met_labels = ["Any Metastasis", "Lung", "Bone", "Liver", "Brain"]
    met_rates = [(df[c].sum() / len(df) * 100) for c in met_cols]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(met_labels, met_rates, color=sns.color_palette("rocket", len(met_labels)))
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Metastasis Site Prevalence")
    ax.invert_yaxis()
    for bar, rate in zip(bars, met_rates):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=9)
    save_fig("09_metastasis_prevalence")

    # --- 9. Survival ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["survival_months"], bins=40, kde=True, ax=axes[0], color="teal")
    axes[0].set_title("Survival Time Distribution (months)")
    axes[0].set_xlabel("Survival (months)")
    axes[0].axvline(df["survival_months"].median(), color="red", ls="--", label=f"Median: {df['survival_months'].median():.0f} mo")
    axes[0].legend()
    sns.boxplot(x=df["vital_status"].map(LABELS["vital_status"]),
                y=df["survival_months"], ax=axes[1],
                hue=df["vital_status"].map(LABELS["vital_status"]),
                legend=False, palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Survival by Vital Status")
    axes[1].set_ylim(0, 180)
    plt.tight_layout()
    save_fig("10_survival")

    # Kaplan-Meier-style step plot (simple, no lifelines dependency)
    fig, ax = plt.subplots(figsize=(9, 5))
    for status, label, color in [(0, "Alive", "#2ecc71"), (1, "Dead", "#e74c3c")]:
        subset = np.sort(df.loc[df["vital_status"] == status, "survival_months"].values)
        if len(subset) == 0:
            continue
        y = np.arange(1, len(subset) + 1) / len(subset)
        ax.step(subset, y, where="post", label=label, color=color, linewidth=2)
    ax.set_xlabel("Survival Time (months)")
    ax.set_ylabel("Proportion Surviving (approx.)")
    ax.set_title("Survival Curve by Vital Status (approximate)")
    ax.legend()
    ax.set_xlim(0, 170)
    save_fig("11_survival_curve")

    # --- 10. Correlation heatmap ---
    corr_cols = [c for c in numeric_cols if c != "year_diagnosis"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Matrix (Numeric Features)")
    save_fig("12_correlation_heatmap")

    # --- 11. Crosstabs ---
    crosstab_met_vital = pd.crosstab(
        df["metastasis"].map(LABELS["metastasis"]),
        df["vital_status"].map(LABELS["vital_status"]),
        normalize="index"
    ).round(3)
    crosstab_met_vital.to_csv(OUT_DIR / "13_metastasis_vs_mortality.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(crosstab_met_vital, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax)
    ax.set_title("Mortality Rate by Metastasis Status")
    save_fig("14_metastasis_mortality")

    # --- 12. Age vs tumor size scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(min(5000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="age", y="tumor_size_cm",
                    hue=sample["vital_status"].map(LABELS["vital_status"]),
                    alpha=0.4, ax=ax, palette=["#2ecc71", "#e74c3c"])
    ax.set_title("Age vs Tumor Size (5k random sample)")
    ax.set_ylim(0, 25)
    save_fig("15_age_vs_tumor_size")

    # --- 13. Pairwise stage vs mortality ---
    stage_mort = df.groupby("t_stage")["vital_status"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [LABELS["t_stage"].get(i, str(i)) for i in stage_mort.index]
    sns.barplot(x=labels, y=stage_mort.values * 100, ax=ax, hue=labels, legend=False, palette="Reds")
    ax.set_title("Mortality Rate by T Stage")
    ax.set_ylabel("Mortality (%)")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(stage_mort.values * 100):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    save_fig("16_mortality_by_t_stage")

    # --- Summary report ---
    mortality_rate = df["vital_status"].mean() * 100
    met_rate = df["metastasis"].mean() * 100
    summary_lines = [
        "SEER RCC 2010–2018 — EDA Summary",
        "=" * 40,
        f"Total patients: {len(df):,}",
        f"Diagnosis years: {df['year_diagnosis'].min()}–{df['year_diagnosis'].max()}",
        f"Age: mean {df['age'].mean():.1f}, median {df['age'].median():.0f}, range {df['age'].min()}–{df['age'].max()}",
        f"Sex: {(df['sex']==1).mean()*100:.1f}% male",
        f"Mortality rate: {mortality_rate:.1f}%",
        f"Any metastasis: {met_rate:.1f}%",
        f"Median survival: {df['survival_months'].median():.0f} months",
        f"Mean tumor size: {df['tumor_size_cm'].mean():.2f} cm (median {df['tumor_size_cm'].median():.1f})",
        "",
        "Charts saved to: eda_output/",
    ]
    summary_text = "\n".join(summary_lines)
    (OUT_DIR / "00_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
