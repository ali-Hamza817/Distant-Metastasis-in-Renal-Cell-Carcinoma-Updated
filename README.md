# Distant Metastasis Prediction in Renal Cell Carcinoma (RCC)

Welcome to the **Multimodal AI Prognosis Model for Renal Cell Carcinoma (RCC)** repository. This project focuses on pushing the boundaries of medical AI by fusing three entirely different data modalities into a single predictive pipeline.

## 🌟 Project Overview
Renal Cell Carcinoma (RCC) is the most common form of kidney cancer. A critical challenge in clinical oncology is accurately predicting **distant metastasis** (the spread of cancer to other organs like lungs, bones, or brain) and assessing survival risks. 

This project aims to solve this by creating an end-to-end, highly advanced deep learning architecture that makes predictions not just by looking at a patient's age or a tumor's size, but by simultaneously interpreting their 3D CT scans, their genetic mutations, and their clinical history.

## 🧬 How the Datasets Collaborate
To achieve true multimodal intelligence, this model learns from three distinct layers of human biology:

1. **SEER Clinical Data (Macro-Level):** 
   - A vast tabular dataset (36,000+ patients) containing age, sex, tumor size, T-stage, and histology. This provides the statistical baseline risk for the patient.
2. **TCIA / TCGA-KIRC Imaging (Anatomical-Level):** 
   - High-resolution 3D CT DICOM scans. These scans allow the model to visually assess the tumor's spatial texture, shape, and surrounding tissue invasion.
3. **CPTAC Genomics / RNA-seq (Molecular-Level):** 
   - Deep RNA-sequencing data (500+ dimensional arrays). This data exposes the genetic footprint of the tumor, revealing which genes are actively driving the cancer.

**The Collaboration:** These datasets do not just exist side-by-side; they actively *communicate* during training. Using a Cross-Attention Fusion mechanism, the model learns complex interaction rules. For example, the model can learn that a specific genetic mutation (from RNA-seq) combined with a specific rugged tumor texture (from CT scans) results in a massive spike in metastasis risk, an association a human doctor might miss.

---

## 🧠 Technical Architecture Details

The network is built in PyTorch and utilizes state-of-the-art architectures for each specific modality before fusing them.

### 1. Imaging Pipeline (3D Swin Transformer)
- **Input:** 3D CT spatial volumes (e.g., $32 \times 32 \times 32$ crops).
- **Process:** We utilize a **3D Swin Transformer**. Unlike traditional Convolutional Neural Networks (CNNs), the Swin Transformer uses shifted windows to capture both local cellular textures and global anatomical structures.
- **Output:** A dense `768-dimensional` imaging feature vector.

### 2. Genomics Pipeline (Transformer Encoder)
- **Input:** 1D array of RNA-seq gene expression counts.
- **Process:** The genetic markers are projected into a higher-dimensional space and passed through a multi-layer **Transformer Encoder**. This allows the network to find correlations across vast sequences of genes (epistasis).
- **Output:** A dense `768-dimensional` genomic feature vector.

### 3. Clinical Pipeline (Standardized MLP)
- **Input:** Normalized categorical and numerical features.
- **Process:** Processed via a simple Multi-Layer Perceptron (MLP) with Batch Normalization and GELU activations.
- **Output:** A `50-dimensional` clinical feature vector.

### 4. Cross-Attention Fusion Module
- The outputs from the three branches are concatenated into a unified sequence of tokens.
- We apply **Multi-Head Cross-Attention**. Here, the imaging features query the genomic features, and the clinical features query the imaging features. The network dynamically decides which modality to "pay attention to" based on the specific patient.

### 5. Multi-Task Output Heads
The fused representations are passed through Shared Dense Layers, splitting into distinct predictive tasks:
- **Metastasis Probability (Binary Classification):** Predicts the 0-100% chance of distant spread (Binary Cross-Entropy Loss).
- **Survival Risk Score (Regression):** Predicts relative survival months (Mean Squared Error Loss).
- **Clinical Decisions (Multi-Class):** Assists in therapeutic routing (Cross-Entropy Loss).

---

## 📁 Repository Structure
The repository is highly organized for readability and reproduction:

- `datasets/`: Contains dataset descriptions and manifests (Note: Raw DICOMs and large CSVs are ignored via `.gitignore` to protect PHI and save space).
- `eda/`: Exploratory Data Analysis. Contains Jupyter notebooks, Python scripts, and output graphs (correlation heatmaps, survival curves) analyzing the SEER dataset.
- `models/`: The core PyTorch neural network definitions (e.g., `swin3d_fusion.py`).
- `scripts/`: The execution pipelines for training (`train_multimodal.py`, `final_gpu_train.py`).
- `app.py`: A clean, interactive **Streamlit** frontend allowing users to upload CT scans, RNA-seq files, and input clinical parameters to receive real-time predictions.
- `utils/`: Helper functions for metrics (ROC AUC, Average Precision).
- `configs/`: Centralized hyperparameter configurations.

## 🚀 Running the Frontend UI
To visualize the AI in action locally:
```bash
pip install streamlit
python -m streamlit run app.py
```
This will launch a dashboard on `localhost:8501` where you can interact with the Multimodal Fusion Model.
