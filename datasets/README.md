# Datasets Directory

This folder is intended to hold the extremely large medical datasets used to train the Multimodal Renal Cell Carcinoma (RCC) Prognosis Model. Due to GitHub's file size constraints and patient data privacy regulations, the raw datasets are **not** pushed to this repository.

## 1. SEER Clinical Dataset (Tabular)
* **File Name:** `seer_rcc_2010_2018_clean.csv`
* **Description:** A large tabular dataset containing clinical parameters for 36,000+ RCC patients. Features include Age, Sex, Tumor Size (cm), T-Stage, N-Stage, Fuhrman Grade, and Histology.
* **Target:** Metastasis (Binary), Survival Months.

## 2. TCIA / TCGA-KIRC (Imaging)
* **Description:** 3D CT Scans (DICOM / NIfTI formats) covering 267 patients. Used to extract spatial tumor morphology via a 3D Swin Transformer.

## 3. CPTAC (Genomics / RNA-seq)
* **Description:** RNA-sequencing data (`augmented_star_gene_counts.tsv`) containing genetic expression profiles. Used to extract the tumor's genetic footprint via a Transformer Encoder.

### How the Datasets Collaborate
The architecture acts as a pipeline where these three distinct modalities are independently encoded and then merged:
1. **Imaging** provides texture and spatial structure.
2. **Genomics** provides molecular-level behavioral signals.
3. **Clinical data** provides the necessary baseline patient context.

These are fed into the **Cross-Attention Fusion** module, which computes attention weights between the modalities to discover complex interactions (e.g., a specific gene mutation combined with a specific tumor texture).
