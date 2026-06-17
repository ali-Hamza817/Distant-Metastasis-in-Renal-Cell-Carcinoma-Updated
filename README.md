# Multimodal Metastasis Prediction in Renal Cell Carcinoma Using Clinical, Genomic, and Imaging Data

**Student Name:** Ali Hamza  
**Degree:** Master of Science in Software Engineering  
**Research Domain:** Artificial Intelligence, Medical Imaging, Machine Learning, Deep Learning, Biomedical Informatics  

---

## Abstract

Renal Cell Carcinoma (RCC) patients are at a significant risk of developing distant metastasis, a condition that severely worsens prognosis and survival outcomes. Current clinical assessments rely heavily on staging systems and physician interpretation. While individual data modalities—such as clinical variables, genomic biomarkers, and imaging findings—provide partial information regarding metastatic risk, integrating these modalities may offer a more comprehensive risk profile. This research develops independent prediction models for clinical (SEER dataset), genomic (CPTAC dataset), and imaging (TCGA-KIRC dataset) modalities, evaluating each independently before investigating a late-fusion integration. A critical limitation discovered during the study was the complete lack of patient overlap between the CPTAC genomic and TCGA imaging cohorts, resulting in a final fusion model constrained to clinical and imaging data. Despite this limitation, the late-fusion Logistic Regression model demonstrated numerical improvements over the clinical-only baseline, achieving an ROC-AUC of 0.776 compared to 0.731, confirming that imaging contributes complementary predictive information beyond clinical variables. 

## Introduction

[Information Not Provided]

## Background and Literature Context

[Information Not Provided]

## Problem Statement

Renal Cell Carcinoma (RCC) patients may develop distant metastasis, which significantly worsens prognosis and survival outcomes. Current clinical assessment methods rely heavily on staging systems and physician interpretation. Individual data modalities such as clinical variables, genomic biomarkers, and imaging findings each provide only partial information regarding metastatic risk. The objective of this research is to investigate whether integrating multiple modalities can improve metastatic risk prediction compared to single-modality approaches.

## Research Objectives

The primary research objectives of this study are to:
1. Develop a clinical metastasis prediction model using SEER data.
2. Develop a genomic risk prediction model using CPTAC RNA-seq data.
3. Develop an imaging-based prediction model using CT DICOM volumes.
4. Evaluate each modality independently.
5. Investigate late-fusion integration of modality outputs.
6. Compare fusion performance against unimodal baselines.

## Dataset Collection and Description

This research utilizes three distinct datasets to represent the clinical, genomic, and imaging modalities. 

### Dataset 1: SEER Clinical Cohort
- **Source:** SEER Program
- **Patients:** 36,738
- **Metastasis Rate:** Approximately 6%
- **Clinical Features:** Age, Sex, T Stage, N Stage, Tumor Size, Tumor Grade, Histology
- **Engineered Features:** t_stage_x_n_stage, age_bin, size_cutoff
- **Purpose:** Clinical metastasis risk prediction

### Dataset 2: CPTAC-CCRCC Transcriptomic Cohort
- **Source:** CPTAC-CCRCC
- **Patients:** 110
- **Positive Cases:** 8
- **Prevalence:** 7.3%
- **Original Features:** 19,275 RNA-seq genes
- **Target Definition:** Stage IV disease used as a proxy for metastatic outcome based on CPTAC annotation schema.

### Dataset 3: TCGA-KIRC Imaging Cohort
- **Source:** TCGA-KIRC DICOM CT Volumes
- **Patients:** 251
- **Positive Cases:** 44
- **Negative Cases:** 207
- **Prevalence:** 17.5%

### Dataset Specification Table

| Dataset | Modality | Source | Patients | Positive Cases | Prevalence | Features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cohort 1 | Clinical | SEER Program | 36,738 | [Information Not Provided] | ~6.0% | Age, Sex, Stage, Size, Grade, Histology |
| Cohort 2 | Genomic | CPTAC-CCRCC | 110 | 8 | 7.3% | 19,275 RNA-seq genes |
| Cohort 3 | Imaging | TCGA-KIRC | 251 | 44 | 17.5% | DICOM CT Volumes |

## Data Preprocessing

[Information Not Provided]

## Clinical Modality (Model 1)

The clinical modality model was developed utilizing the SEER cohort to establish a baseline for metastasis prediction based solely on routinely collected clinical features. 

**Selected Model:** LightGBM  
**Clinical Interpretation:** The model functions primarily as a risk stratification tool rather than a standalone diagnostic system.

## Genomic Modality (Model 2)

The genomic model was designed to isolate transcriptomic signatures associated with Stage IV disease. To prevent data leakage, variance filtering (top 1,000 variable genes) and scaling were performed dynamically, strictly within the training folds. 

**Model:** Elastic Net Logistic Regression  
**Interpretation:** The transcriptomic signal appears to contain informative metastatic-risk information despite the extremely small cohort size.

## Imaging Modality (Model 3)

The imaging modality extracted radiomic signals directly from TCGA-KIRC DICOM CT volumes. The model utilized heavy augmentation and early stopping to act as regularizers against overfitting.

**Model:** Custom Lightweight 3D CNN  
**Parameters:** 72,001  
**Interpretation:** The model learned a modest but measurable radiomic signal above the prevalence baseline.

## Late Fusion Architecture (Model 4)

![Figure: System Architecture Diagram](placeholder_system_architecture.png)

A classic late-fusion architecture was selected to synthesize the independent predictive probabilities. 

**Fusion Type:** Late Fusion Logistic Regression  
**Inputs:** `p_clinical`, `p_imaging`

### Model Architecture Table

| Modality | Model Architecture | Parameters / Features | Validation Strategy |
| :--- | :--- | :--- | :--- |
| Clinical (Model 1) | LightGBM | 7 raw + 3 engineered | [Information Not Provided] |
| Genomic (Model 2) | Elastic Net Logistic Regression | Top 1,000 genes (dynamic) | Repeated Stratified 5-Fold CV |
| Imaging (Model 3) | Custom Lightweight 3D CNN | 72,001 parameters | 5-Fold Stratified CV |
| Late Fusion (Model 4) | Logistic Regression | 2 probabilities | [Information Not Provided] |

## Experimental Setup

The experimental validation strategy differed slightly per dataset due to cohort constraints. The genomic model utilized Repeated Stratified 5-Fold Cross-Validation, while the imaging model utilized 5-Fold Stratified Cross-Validation. 

## Evaluation Metrics

[Information Not Provided]

## Results

### Clinical Model Results
![Figure: Clinical Evaluation Curves](placeholder_clinical_curves.png)
- **ROC-AUC:** 0.719
- **PR-AUC:** 0.190
- **Recall (Sensitivity):** 0.486
- **Specificity:** 0.787
- **NPV:** 0.960
- **Accuracy:** 0.769

### Genomic Model Results
![Figure: Genomic Evaluation Curves](placeholder_genomic_curves.png)
- **ROC-AUC:** 0.793 (95% CI: 0.734–0.847)
- **PR-AUC:** 0.161 (95% CI: 0.111–0.230)
- **Recall @ 90% Specificity:** 0.200 (95% CI: 0.091–0.457)
- **Recall @ 95% Specificity:** 0.075 (95% CI: 0.000–0.205)
- **F1 Score:** 0.125 (95% CI: 0.029–0.229)
- **Brier Score:** 0.087 (95% CI: 0.071–0.103)

### Imaging Model Results
![Figure: Imaging Evaluation Curves](placeholder_imaging_curves.png)
- **ROC-AUC:** 0.603 (95% CI: 0.505–0.699)
- **PR-AUC:** 0.244 (95% CI: 0.165–0.376)
- **Confusion Matrix:** True Negative (165), False Positive (42), False Negative (27), True Positive (17)

### Results Comparison Table

| Modality | ROC-AUC | PR-AUC | Recall | NPV |
| :--- | :--- | :--- | :--- | :--- |
| SEER Clinical | 0.719 | 0.190 | 0.486 | 0.960 |
| CPTAC Genomic | 0.793 | 0.161 | 0.200* | [Information Not Provided] |
| TCGA Imaging | 0.603 | 0.244 | 0.386** | [Information Not Provided] |

*(Note: Genomic Recall is reported @ 90% Specificity. **Imaging Recall derived from Confusion Matrix: 17/(17+27) = 0.386)*

## Comparative Analysis

![Figure: Fusion Evaluation Curves](placeholder_fusion_curves.png)

A late-fusion Logistic Regression model was trained using the clinical and imaging predictions. The learned coefficients demonstrated that both modalities contributed heavily, with a Clinical Weight of 2.84 and an Imaging Weight of 2.40. 

### Fusion Improvement Table

| Metric | Clinical Baseline | Late Fusion (Clinical + Imaging) | Absolute Improvement |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.731 | 0.776 | +0.045 |
| **PR-AUC** | 0.271 | 0.344 | +0.073 |
| **Recall** | 0.773 | 0.795 | +0.022 |
| **NPV** | 0.931 | 0.938 | +0.007 |

*(Note: Baseline evaluation metrics here refer specifically to the subset evaluation directly compared during the fusion step).*

## Discussion

The comparative analysis indicates that the imaging modality contributes complementary predictive information beyond clinical variables. By fusing the predicted risk scores, the late-fusion model was able to numerically out-perform the clinical baseline across all evaluated metrics (ROC-AUC, PR-AUC, Recall, and NPV). 

## Limitations

A severe and explicit limitation of this study is the cohort mismatch between the genomic and imaging datasets. There was absolutely no patient overlap between the CPTAC genomic patients and the TCGA imaging patients. Consequently, the genomic modality was NOT part of the final evaluated fusion model. The validated fusion model represents a combination of Clinical + Imaging, NOT Clinical + Genomic + Imaging. Furthermore, the dataset sizes, particularly for the genomic (N=110) and imaging (N=251) cohorts, were extremely small, heavily limiting deep neural network performance. 

## Threats to Validity

[Information Not Provided]

## Future Work

[Information Not Provided]

## Conclusion

This research developed and evaluated independent clinical, genomic, and imaging models for predicting distant metastasis in Renal Cell Carcinoma. Despite the explicit limitation of cohort mismatch preventing a three-modality fusion, the integration of clinical and imaging modalities via a late-fusion Logistic Regression framework successfully demonstrated numeric improvements over a clinical-only baseline. These findings support the hypothesis that multimodal data integration provides complementary prognostic value over single-modality risk assessments. 

## References

[Information Not Provided]
