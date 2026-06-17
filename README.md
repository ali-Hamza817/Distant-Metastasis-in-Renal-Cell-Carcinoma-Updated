# 🔬 Multimodal Metastasis Prediction in Renal Cell Carcinoma

**Student Name:** Ali Hamza  
**Degree:** Master of Science in Software Engineering  
**Research Domain:** Artificial Intelligence, Medical Imaging, Machine Learning, Deep Learning, Biomedical Informatics  

---

## 🎯 Abstract & Problem Statement

Renal Cell Carcinoma (RCC) patients face significant risks of distant metastasis, which drastically worsens their prognosis. Current clinical assessments rely on standard staging, providing only a partial risk profile. 

**Our Objective:** Investigate whether integrating multiple modalities (Clinical, Genomic, Imaging) can significantly improve metastatic risk prediction.

## 📊 Dataset Specifications

We utilized three independent cohorts to train specialized models before fusing them.

| Dataset | Modality | Source | Patients | Positive Cases | Prevalence | Features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cohort 1** | Clinical | SEER Program | 36,738 | ~2,204 | ~6.0% | Age, Sex, Stage, Size, Grade, Histology |
| **Cohort 2** | Genomic | CPTAC-CCRCC | 110 | 8 | 7.3% | 19,275 RNA-seq genes |
| **Cohort 3** | Imaging | TCGA-KIRC | 251 | 44 | 17.5% | DICOM CT Volumes |

*(⚠️ **Constraint Identified:** There was zero patient overlap between the CPTAC genomic cohort and the TCGA imaging cohort).*

---

## 🏗️ Model Architectures

| Modality | Model Architecture | Parameters / Features | Validation Strategy |
| :--- | :--- | :--- | :--- |
| **Clinical (Model 1)** | LightGBM | 7 raw + 3 engineered | Standard CV |
| **Genomic (Model 2)** | Elastic Net Logistic Regression | Top 1,000 genes | Repeated Stratified 5-Fold CV |
| **Imaging (Model 3)** | Custom Lightweight 3D CNN | 72,001 parameters | 5-Fold Stratified CV |
| **Late Fusion (Model 4)** | Logistic Regression | 2 probabilities | 5-Fold Stratified CV |

---

## 📈 Unimodal Results (Independent Baselines)

Each modality was rigorously evaluated on its respective dataset.

| Modality | ROC-AUC | PR-AUC | Recall | NPV |
| :--- | :--- | :--- | :--- | :--- |
| **SEER Clinical** | 0.719 | 0.190 | 0.486 | 0.960 |
| **CPTAC Genomic** | 0.793 | 0.161 | 0.200* | N/A |
| **TCGA Imaging** | 0.603 | 0.244 | 0.386** | N/A |

*\*Genomic Recall is reported @ 90% Specificity. \*\*Imaging Recall derived from Confusion Matrix.*

---

## 🚀 Late Fusion Architecture & Multimodal Lift

To prove the value of multimodal integration, we fused the **Clinical** and **Imaging** predictions using a Logistic Regression framework. Due to the cohort mismatch limitation, Genomics could not be fused into this specific final layer.

The fusion model learned the following importance weights:
- **Clinical Weight:** `+2.84`
- **Imaging Weight:** `+2.40`

### 🏆 Final Improvement Metrics

| Metric | Clinical Baseline | Late Fusion (Clinical + Imaging) | Absolute Improvement |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.731 | **0.776** | 📈 +0.045 |
| **PR-AUC** | 0.271 | **0.344** | 📈 +0.073 |
| **Recall** | 0.773 | **0.795** | 📈 +0.022 |
| **NPV** | 0.931 | **0.938** | 📈 +0.007 |

---

## 💡 Conclusion

The integration of 3D CT radiomics with standard clinical variables provides **complementary predictive power**, proving that multimodal Late Fusion significantly outperforms clinical staging alone for predicting distant metastasis in RCC.
