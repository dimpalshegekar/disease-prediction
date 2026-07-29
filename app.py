"""
Multi-Disease Prediction System - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="wide")

st.title("🏥 Multi-Disease Prediction System")
st.markdown("AI-powered predictions for **Diabetes**, **Heart Disease**, and **Breast Cancer**")

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    base = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base, "models"),
        os.path.join(os.getcwd(), "models"),
    ]
    diseases = ["diabetes", "heart_disease", "breast_cancer"]
    loaded = {}
    for d in diseases:
        for folder in search_dirs:
            mp = os.path.join(folder, f"{d}_model.pkl")
            sp = os.path.join(folder, f"{d}_scaler.pkl")
            fp = os.path.join(folder, f"{d}_features.pkl")
            if os.path.exists(mp):
                loaded[d] = {
                    "model":    joblib.load(mp),
                    "scaler":   joblib.load(sp),
                    "features": joblib.load(fp),
                }
                break
    return loaded

models = load_all_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Navigation")
disease = st.sidebar.radio("Select Disease", ["Diabetes", "Heart Disease", "Breast Cancer", "📊 Model Comparison"])
st.sidebar.markdown("---")
st.sidebar.info("**Models Used:**\n- Logistic Regression\n- Random Forest\n- Gradient Boosting\n\n**Best model auto-selected per disease.**")

# ── Helper ────────────────────────────────────────────────────────────────────
def predict(tag, inputs):
    if tag not in models:
        st.error(f"⚠️ Model not found. Please run `python train.py` first.")
        return
    m = models[tag]
    df = pd.DataFrame([inputs], columns=m["features"])
    scaled = m["scaler"].transform(df)
    prob = m["model"].predict_proba(scaled)[0][1]
    pred = m["model"].predict(scaled)[0]
    return prob, pred

def show_result(prob, pred, disease_name):
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error(f"⚠️ **HIGH RISK of {disease_name}**\n\nConfidence: {prob*100:.1f}%")
        else:
            st.success(f"✅ **LOW RISK of {disease_name}**\n\nRisk probability: {prob*100:.1f}%")
    with col2:
        st.metric("Risk Probability", f"{prob*100:.2f}%")
        risk = "🔴 High" if prob > 0.6 else "🟡 Medium" if prob > 0.3 else "🟢 Low"
        st.metric("Risk Level", risk)
        st.progress(float(prob))
    st.warning("⚕️ This is an AI prediction only. Always consult a qualified doctor for medical advice.")

# ══════════════════════════════════════════════════════════════════════════════
# DIABETES
# ══════════════════════════════════════════════════════════════════════════════
if disease == "Diabetes":
    st.header("🩺 Diabetes Risk Prediction")
    st.markdown("Enter patient details below:")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose     = st.slider("Glucose Level (mg/dL)", 0, 200, 110)
        bp          = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
        skin        = st.slider("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin     = st.slider("Insulin (IU/mL)", 0, 900, 80)
        bmi         = st.slider("BMI", 0.0, 70.0, 28.0)
        dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.4)
        age         = st.slider("Age", 18, 80, 30)

    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        result = predict("diabetes", [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
        if result:
            show_result(result[0], result[1], "Diabetes")

# ══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Risk Prediction")
    st.markdown("Enter patient details below:")

    col1, col2 = st.columns(2)
    with col1:
        age      = st.slider("Age", 29, 77, 50)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        cp       = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 130)
        chol     = st.slider("Cholesterol (mg/dL)", 140, 400, 240)
        fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        restecg  = st.selectbox("Resting ECG", [0,1,2])
    with col2:
        thalach  = st.slider("Max Heart Rate", 90, 202, 150)
        exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak  = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope    = st.selectbox("ST Slope", [0,1,2])
        ca       = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal     = st.selectbox("Thalassemia", [0,1,2,3])

    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        result = predict("heart_disease", [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        if result:
            show_result(result[0], result[1], "Heart Disease")

# ══════════════════════════════════════════════════════════════════════════════
# BREAST CANCER
# ══════════════════════════════════════════════════════════════════════════════
elif disease == "Breast Cancer":
    st.header("🎗️ Breast Cancer Risk Prediction")
    st.markdown("Enter tumor measurement details below:")

    col1, col2 = st.columns(2)
    with col1:
        radius          = st.slider("Mean Radius", 5.0, 30.0, 14.0)
        texture         = st.slider("Mean Texture", 5.0, 40.0, 19.0)
        perimeter       = st.slider("Mean Perimeter", 40.0, 190.0, 92.0)
        area            = st.slider("Mean Area", 140.0, 2500.0, 655.0)
        smoothness      = st.slider("Mean Smoothness", 0.05, 0.16, 0.096)
    with col2:
        compactness     = st.slider("Mean Compactness", 0.02, 0.35, 0.104)
        concavity       = st.slider("Mean Concavity", 0.0, 0.43, 0.089)
        concave_points  = st.slider("Mean Concave Points", 0.0, 0.20, 0.049)
        symmetry        = st.slider("Mean Symmetry", 0.1, 0.3, 0.181)
        fractal_dim     = st.slider("Mean Fractal Dimension", 0.05, 0.10, 0.063)

    if st.button("🔍 Predict Cancer Risk", use_container_width=True):
        result = predict("breast_cancer", [radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dim])
        if result:
            show_result(result[0], result[1], "Breast Cancer")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif disease == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")

    base = os.path.dirname(os.path.abspath(__file__))
    tabs = st.tabs(["🩺 Diabetes", "❤️ Heart Disease", "🎗️ Breast Cancer"])
    tags = ["diabetes", "heart_disease", "breast_cancer"]

    for tab, tag in zip(tabs, tags):
        with tab:
            # Results table
            for folder in [os.path.join(base,"outputs"), os.path.join(os.getcwd(),"outputs")]:
                rp = os.path.join(folder, f"{tag}_results.csv")
                if os.path.exists(rp):
                    df_r = pd.read_csv(rp)
                    st.dataframe(df_r.style.highlight_max(
                        subset=["AUC-ROC","F1","Accuracy"], color="#d4edda"),
                        use_container_width=True)
                    break

            # Evaluation plot
            for folder in [os.path.join(base,"outputs"), os.path.join(os.getcwd(),"outputs")]:
                ep = os.path.join(folder, f"{tag}_evaluation.png")
                if os.path.exists(ep):
                    st.image(ep, use_column_width=True)
                    break
