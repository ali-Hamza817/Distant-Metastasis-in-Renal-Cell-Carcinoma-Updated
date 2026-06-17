import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import joblib
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="RCC Metastasis Predictor", page_icon="🔬", layout="wide")
st.title("🔬 Multimodal Metastasis Prediction in RCC")
st.markdown("This dashboard integrates Clinical, Genomic, and Imaging data natively. Upload patient files below to run inference.")

# ==========================================
# MODEL DEFINITIONS & LOADING
# ==========================================
class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2) 

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x) 
        return x

@st.cache_resource
def load_all_models():
    # 1. Clinical
    model_clin = CatBoostClassifier()
    model_clin.load_model("e:/rcc/models/model1_catboost.cbm")
    
    # 2. Genomic
    model_gen = joblib.load("e:/rcc/models/model2_genomic_REAL.pkl")
    
    # 3. Imaging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_img = Small3DCNN()
    model_img.load_state_dict(torch.load("e:/rcc/models/model3_imaging.pt", map_location=device, weights_only=False))
    model_img.to(device)
    model_img.eval()
    
    # 4. Fusion
    try:
        model_fusion = joblib.load("e:/rcc/models/model4_fusion.pkl")
    except:
        with open("e:/rcc/models/model4_fusion.pkl", "rb") as f:
            model_fusion = pickle.load(f)
            
    return model_clin, model_gen, model_img, model_fusion, device

try:
    model_clin, dict_gen, model_img, model_fusion, device = load_all_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False


# ==========================================
# UI: CLINICAL INPUT
# ==========================================
st.sidebar.header("📋 1. Clinical Data (Patient File)")
age = st.sidebar.slider("Age", 18, 100, 60)
sex_str = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_str == "Male" else 0

tumor_size_cm = st.sidebar.slider("Tumor Size (cm)", 0.0, 20.0, 5.0, step=0.1)
t_stage = st.sidebar.slider("T-Stage (1-4)", 1, 4, 1)
n_stage = st.sidebar.slider("N-Stage (0-1)", 0, 1, 0)
grade = st.sidebar.slider("Fuhrman Grade (1-4)", 1, 4, 2)

histology_str = st.sidebar.selectbox("Histology Subtype", ["Clear Cell", "Papillary", "Chromophobe"])
hist_map = {"Clear Cell": 0, "Papillary": 1, "Chromophobe": 2}
histology_enc = hist_map[histology_str]

t_stage_x_n_stage = t_stage * n_stage
age_bin = age // 10
size_cutoff = 1 if tumor_size_cm > 7.0 else 0

input_data = [age, sex, t_stage, n_stage, tumor_size_cm, grade, histology_enc, 
              t_stage_x_n_stage, age_bin, size_cutoff]
              
if models_loaded:
    p_clin = model_clin.predict_proba([input_data])[0][1]
else:
    p_clin = 0.5


# ==========================================
# UI: GENOMIC INPUT (RNA-seq Upload)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🧬 2. Genomic Data Upload")
genomic_file = st.sidebar.file_uploader("Upload RNA-seq data (.csv or .cct)", type=['csv', 'cct', 'txt'])

p_genomic = 0.5
has_genomic = 0

if genomic_file is not None and models_loaded:
    try:
        # Load user file
        if genomic_file.name.endswith('.cct'):
            user_df = pd.read_csv(genomic_file, sep='\t')
            if "gene" in user_df.columns:
                user_df = user_df.set_index("gene")
            user_df = user_df.T
        else:
            user_df = pd.read_csv(genomic_file)
            
        st.sidebar.success(f"Loaded RNA-seq with {user_df.shape[1]} genes.")
        
        # Pipeline execution
        scaler = dict_gen['scaler']
        pca = dict_gen.get('pca', None)
        model = dict_gen['model']
        required_genes = dict_gen['genes']
        
        # Ensure all required genes exist (pad missing with 0 for demo purposes)
        missing_genes = [g for g in required_genes if g not in user_df.columns]
        if missing_genes:
            st.sidebar.warning(f"Missing {len(missing_genes)} required genes. Padding with zeros.")
            for g in missing_genes:
                user_df[g] = 0.0
                
        # Filter to exact feature space
        X_user = user_df[required_genes].values[0].reshape(1, -1) # take first patient
        
        X_scaled = scaler.transform(X_user)
        if pca is not None:
            X_scaled = pca.transform(X_scaled)
            
        p_genomic = model.predict_proba(X_scaled)[0][1]
        has_genomic = 1
        st.sidebar.success(f"🧬 Genomic P(Metastasis): {p_genomic*100:.1f}%")
    except Exception as e:
        st.sidebar.error(f"Genomic processing error: {e}")


# ==========================================
# UI: IMAGING INPUT (CT Scan Upload)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🩻 3. Imaging Data Upload")
imaging_file = st.sidebar.file_uploader("Upload 3D CT Tensor (.pt)", type=['pt'])

p_imaging = 0.5
has_imaging = 0

if imaging_file is not None and models_loaded:
    try:
        # Load user tensor
        user_tensor = torch.load(imaging_file, map_location=device, weights_only=False)
        
        # Ensure correct shape [1, 1, 64, 96, 96]
        if len(user_tensor.shape) == 4: # missing batch dim
            user_tensor = user_tensor.unsqueeze(0)
            
        st.sidebar.success(f"Loaded CT Tensor shape: {user_tensor.shape}")
        
        user_tensor = user_tensor.to(device)
        with torch.no_grad():
            logits = model_img(user_tensor)
            prob = torch.sigmoid(logits).item()
            
        p_imaging = prob
        has_imaging = 1
        st.sidebar.success(f"🩻 Imaging P(Metastasis): {p_imaging*100:.1f}%")
    except Exception as e:
        st.sidebar.error(f"Imaging processing error: {e}")


# ==========================================
# UI: FUSION & OUTPUT
# ==========================================
st.header("📊 Metastasis Risk Report")

if not models_loaded:
    st.error("Cannot display report: Models failed to load.")
else:
    X_fusion = [[p_clin, p_genomic, p_imaging, has_genomic, has_imaging]]
    p_final = model_fusion.predict_proba(X_fusion)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.info("### 📋 Clinical Baseline")
        st.markdown("Risk predicted using **only standard patient files**.")
        st.metric(label="Probability of Metastasis", value=f"{p_clin * 100:.1f}%")
        
    with col2:
        st.success("### 🚀 Multi-Modal Fusion")
        st.markdown("Risk predicted combining **Clinical + Genomic + Imaging**.")
        st.metric(label="Probability of Metastasis", value=f"{p_final * 100:.1f}%", delta=f"{(p_final - p_clin)*100:.1f}%")

    st.markdown("---")
    st.subheader("🚨 Final Risk Tier Diagnosis")
    if p_final >= 0.65:
        st.error("### HIGH RISK \n Patient is highly likely to develop distant metastasis. Immediate advanced screening recommended.")
    elif p_final >= 0.35:
        st.warning("### MEDIUM RISK \n Patient has elevated risk. Regular follow-up CT scans recommended.")
    else:
        st.success("### LOW RISK \n Patient is at low risk for metastasis.")
