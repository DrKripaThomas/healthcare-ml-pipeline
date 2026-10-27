"""
Healthcare Diabetes Risk Classification Pipeline - Streamlit Dashboard
========================================================================
Academic Project: MSc Health Informatics and Analytics
Module: Machine Learning Workflow & Clinical Decision Support Systems

This interactive dashboard demonstrates the deployment of a Scikit-learn
Pipeline (SimpleImputer -> StandardScaler -> LogisticRegression) for
diabetes risk stratification.

DISCLAIMER: This application is intended solely for educational demonstration
and research study. It is NOT clinically validated, does NOT constitute medical
advice, and must NEVER be used for clinical diagnostic decision-making.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Pipeline | MSc Health Informatics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #7b341e;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-high {
        background-color: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 8px;
        padding: 1.2rem;
        color: #9b2c2c;
    }
    .risk-moderate {
        background-color: #fffaf0;
        border: 2px solid #dd6b20;
        border-radius: 8px;
        padding: 1.2rem;
        color: #9c4221;
    }
    .risk-low {
        background-color: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 8px;
        padding: 1.2rem;
        color: #22543d;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants & Reference Definitions
# -----------------------------------------------------------------------------
MODEL_FILE = os.path.join("models", "diabetes_pipeline.joblib")
METADATA_FILE = os.path.join("models", "model_metadata.json")
DATA_FILE = os.path.join("data", "diabetes.csv")
FIGURES_DIR = os.path.join("reports", "figures")

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Normal physiological reference ranges for educational context
CLINICAL_REFERENCE_RANGES = {
    "Glucose": {"normal": "70 – 99 mg/dL (fasting) / <140 mg/dL (2-hr OGTT)", "unit": "mg/dL"},
    "BloodPressure": {"normal": "60 – 80 mm Hg (diastolic)", "unit": "mm Hg"},
    "SkinThickness": {"normal": "10 – 30 mm (triceps skinfold)", "unit": "mm"},
    "Insulin": {"normal": "16 – 166 μU/mL (2-hr post-load)", "unit": "μU/mL"},
    "BMI": {"normal": "18.5 – 24.9 kg/m² (normal weight)", "unit": "kg/m²"},
    "DiabetesPedigreeFunction": {"normal": "0.08 – 0.50 (low hereditary risk)", "unit": "score"},
    "Age": {"normal": "Adult cohort (21+ years)", "unit": "years"},
    "Pregnancies": {"normal": "Parity count (≥0)", "unit": "count"}
}

# -----------------------------------------------------------------------------
# Model & Metadata Loading Functions (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_pipeline():
    """Load the serialized scikit-learn pipeline."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        pipeline = joblib.load(MODEL_FILE)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_model_metadata():
    """Load the model evaluation metrics and feature importance metadata."""
    if not os.path.exists(METADATA_FILE):
        return None
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the underlying diabetes dataset."""
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🩺 Healthcare Diabetes Risk Classification Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">MSc Health Informatics & Analytics | Clinical Machine Learning Workflow Demonstration</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Health Informatics Academic & Regulatory Disclaimer:</strong><br>
    This machine learning system is an academic demonstration developed for university coursework. 
    It is <strong>NOT clinically validated</strong> by regulatory bodies (e.g., FDA SaMD, CE-MDR), does not incorporate vital 
    clinical indicators like HbA1c or full multi-ethnic cohorts, and must <strong>never</strong> be used as an independent clinical diagnostic instrument.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Pipeline & Metadata with Graceful Error Handling
# -----------------------------------------------------------------------------
pipeline = load_trained_pipeline()
metadata = load_model_metadata()
df = load_dataset()

if pipeline is None:
    st.error(
        "❌ **Model Artifact Not Found!**\n\n"
        f"The model file `{MODEL_FILE}` was not detected. "
        "Please run the training pipeline first from your terminal:\n\n"
        "```powershell\npython train_model.py\n```\n"
        "After running the script, refresh this dashboard."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: Presets, Thresholds & Architecture Details
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Star_of_life_caution.svg/200px-Star_of_life_caution.svg.png",
        width=70
    )
    st.title("Clinical Controls")

    st.subheader("1. Patient Profile Presets")
    st.caption("Select a synthetic clinical case or input custom values:")
    case_preset = st.selectbox(
        "Select Clinical Case:",
        [
            "Custom Input",
            "Case A: Healthy Low-Risk Female (23y)",
            "Case B: High-Risk Metabolic Syndrome (54y)",
            "Case C: Borderline Prediabetic Female (38y)"
        ]
    )

    # Preset profiles
    presets = {
        "Custom Input": {
            "Pregnancies": 2, "Glucose": 110.0, "BloodPressure": 70.0,
            "SkinThickness": 20.0, "Insulin": 80.0, "BMI": 25.4,
            "DiabetesPedigreeFunction": 0.35, "Age": 33
        },
        "Case A: Healthy Low-Risk Female (23y)": {
            "Pregnancies": 0, "Glucose": 88.0, "BloodPressure": 68.0,
            "SkinThickness": 18.0, "Insulin": 54.0, "BMI": 21.8,
            "DiabetesPedigreeFunction": 0.20, "Age": 23
        },
        "Case B: High-Risk Metabolic Syndrome (54y)": {
            "Pregnancies": 6, "Glucose": 178.0, "BloodPressure": 88.0,
            "SkinThickness": 38.0, "Insulin": 220.0, "BMI": 36.5,
            "DiabetesPedigreeFunction": 0.85, "Age": 54
        },
        "Case C: Borderline Prediabetic Female (38y)": {
            "Pregnancies": 3, "Glucose": 128.0, "BloodPressure": 76.0,
            "SkinThickness": 28.0, "Insulin": 115.0, "BMI": 29.2,
            "DiabetesPedigreeFunction": 0.48, "Age": 38
        }
    }
    selected_values = presets[case_preset]

    st.divider()
    st.subheader("2. Decision Threshold")
    decision_threshold = st.slider(
        "Classification Cut-off (Probability):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="In clinical screening, lowering the threshold increases Sensitivity (catches more true cases) at the expense of Specificity."
    )
    if decision_threshold < 0.50:
        st.info("ℹ️ **Screening Mode**: Prioritizes higher Sensitivity (Recall) to minimize dangerous False Negatives.")
    elif decision_threshold > 0.50:
        st.info("ℹ️ **Conservative Mode**: Prioritizes higher Specificity to minimize False Positives.")

    st.divider()
    st.subheader("3. Pipeline Architecture")
    st.markdown("""
    - **Imputer**: Median (`SimpleImputer`)
    - **Scaler**: Z-score (`StandardScaler`)
    - **Classifier**: `LogisticRegression`
    - **Data Leakage Guard**: Split before fitting
    """)
    if metadata and "timestamp" in metadata:
        st.caption(f"Trained: {metadata['timestamp'][:19]}")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_metrics, tab_eda, tab_governance = st.tabs([
    "🩺 Patient Risk Predictor",
    "📊 Model Performance & Metrics",
    "🔍 Exploratory Data Analysis",
    "🛡️ Governance, Drift & Limitations"
])

# =============================================================================
# TAB 1: PATIENT RISK PREDICTOR
# =============================================================================
with tab_predict:
    st.subheader("Patient Clinical Measurement Intake")
    st.markdown(
        "Enter observed clinical vital signs, anthropometrics, and laboratory findings. "
        "Biological zero handling automatically treats non-physiological 0 entries as missing."
    )

    with st.form("patient_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics & History")
            input_pregnancies = st.number_input(
                "Pregnancies (Parity count):",
                min_value=0,
                max_value=20,
                value=int(selected_values["Pregnancies"]),
                step=1,
                help="Number of times pregnant (0 is biologically valid)."
            )
            input_age = st.number_input(
                "Age (Years):",
                min_value=18,
                max_value=105,
                value=int(selected_values["Age"]),
                step=1,
                help="Patient age in completed calendar years."
            )
            input_pedigree = st.number_input(
                "Diabetes Pedigree Function (Hereditary Score):",
                min_value=0.05,
                max_value=3.00,
                value=float(selected_values["DiabetesPedigreeFunction"]),
                step=0.01,
                format="%.3f",
                help="Genetic risk score synthesized from family history of diabetes."
            )

        with col2:
            st.markdown("##### 🩸 Metabolic & Lab Tests")
            input_glucose = st.number_input(
                "Plasma Glucose (2-hr OGTT, mg/dL):",
                min_value=0.0,
                max_value=300.0,
                value=float(selected_values["Glucose"]),
                step=1.0,
                help="Plasma glucose concentration 2 hours after a 75g glucose load. Normal: 70–99 mg/dL fasting, <140 mg/dL 2-hr OGTT. (Enter 0 if unrecorded/missing)."
            )
            input_insulin = st.number_input(
                "Serum Insulin (2-hr post-load, μU/mL):",
                min_value=0.0,
                max_value=900.0,
                value=float(selected_values["Insulin"]),
                step=1.0,
                help="2-Hour serum insulin. Normal fasting: 16–166 μU/mL. (Enter 0 if unrecorded/missing)."
            )
            input_bp = st.number_input(
                "Diastolic Blood Pressure (mm Hg):",
                min_value=0.0,
                max_value=150.0,
                value=float(selected_values["BloodPressure"]),
                step=1.0,
                help="Diastolic blood pressure. Normal: 60–80 mm Hg. (Enter 0 if unrecorded/missing)."
            )

        with col3:
            st.markdown("##### 📏 Anthropometrics")
            input_bmi = st.number_input(
                "Body Mass Index (BMI, kg/m²):",
                min_value=0.0,
                max_value=70.0,
                value=float(selected_values["BMI"]),
                step=0.1,
                format="%.1f",
                help="BMI = weight (kg) / height (m)². Normal: 18.5–24.9; Overweight: 25.0–29.9; Obese: ≥30. (Enter 0 if unrecorded/missing)."
            )
            input_skin = st.number_input(
                "Triceps Skinfold Thickness (mm):",
                min_value=0.0,
                max_value=100.0,
                value=float(selected_values["SkinThickness"]),
                step=1.0,
                help="Triceps skin fold thickness in mm. Normal female: 18–25 mm. (Enter 0 if unrecorded/missing)."
            )

        submitted = st.form_submit_button("⚡ Run Diabetes Risk Assessment", use_container_width=True)

    # Process prediction upon form submission
    if submitted:
        # Prepare input DataFrame with EXACT feature names and order
        patient_dict = {
            "Pregnancies": [input_pregnancies],
            "Glucose": [np.nan if input_glucose == 0 else input_glucose],
            "BloodPressure": [np.nan if input_bp == 0 else input_bp],
            "SkinThickness": [np.nan if input_skin == 0 else input_skin],
            "Insulin": [np.nan if input_insulin == 0 else input_insulin],
            "BMI": [np.nan if input_bmi == 0 else input_bmi],
            "DiabetesPedigreeFunction": [input_pedigree],
            "Age": [input_age]
        }
        patient_df = pd.DataFrame(patient_dict)[FEATURE_COLUMNS]

        # Predict probability using pipeline (Pipeline handles imputation and scaling automatically)
        predicted_proba = pipeline.predict_proba(patient_df)[0, 1]
        predicted_class = int(predicted_proba >= decision_threshold)

        st.divider()
        st.subheader("Clinical Risk Assessment Results")

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            # Determine Risk Tier
            if predicted_proba >= 0.65:
                risk_tier = "High Risk"
                risk_class = "risk-high"
                risk_icon = "🚨"
                clinical_recommendation = (
                    "**Priority Clinical Follow-up Advised:** Patient exhibits elevated risk indicators. "
                    "Recommend diagnostic confirmation with Fasting Blood Glucose (FBG) and HbA1c testing."
                )
            elif predicted_proba >= 0.35:
                risk_tier = "Moderate / Borderline Risk"
                risk_class = "risk-moderate"
                risk_icon = "⚠️"
                clinical_recommendation = (
                    "**Lifestyle Monitoring Recommended:** Patient falls in intermediate risk category. "
                    "Recommend counseling on dietary habits, weight management, and periodic re-screening in 6–12 months."
                )
            else:
                risk_tier = "Low Risk"
                risk_class = "risk-low"
                risk_icon = "✅"
                clinical_recommendation = (
                    "**Routine Preventive Monitoring:** Patient currently exhibits low risk for diabetes. "
                    "Encourage maintenance of healthy lifestyle behaviors."
                )

            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin-top:0;">{risk_icon} {risk_tier}</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <strong>Predicted Risk Probability:</strong> {predicted_proba * 100:.1f}%
                </p>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;">
                    <strong>Binary Classification (Threshold = {decision_threshold:.2f}):</strong><br>
                    { '🔴 Positive for Diabetes Risk' if predicted_class == 1 else '🟢 Negative for Diabetes Risk' }
                </p>
                <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid rgba(0,0,0,0.15);">
                <p style="font-size: 0.88rem; line-height: 1.4;">
                    {clinical_recommendation}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### 🔬 Physiological Measurement Profile vs Normal Reference")
            
            comparison_data = [
                {
                    "Feature": "Plasma Glucose",
                    "Entered Value": f"{input_glucose:.0f} mg/dL" if input_glucose > 0 else "Unrecorded (Imputed by Pipeline)",
                    "Reference Range": CLINICAL_REFERENCE_RANGES["Glucose"]["normal"],
                    "Status": "⚠️ High" if input_glucose >= 140 else ("✅ Normal" if input_glucose >= 70 else "Missing/Low")
                },
                {
                    "Feature": "Body Mass Index (BMI)",
                    "Entered Value": f"{input_bmi:.1f} kg/m²" if input_bmi > 0 else "Unrecorded (Imputed)",
                    "Reference Range": CLINICAL_REFERENCE_RANGES["BMI"]["normal"],
                    "Status": "⚠️ Obese" if input_bmi >= 30 else ("⚠️ Overweight" if input_bmi >= 25 else "✅ Normal")
                },
                {
                    "Feature": "Blood Pressure (Diastolic)",
                    "Entered Value": f"{input_bp:.0f} mm Hg" if input_bp > 0 else "Unrecorded (Imputed)",
                    "Reference Range": CLINICAL_REFERENCE_RANGES["BloodPressure"]["normal"],
                    "Status": "⚠️ Elevated" if input_bp > 80 else ("✅ Normal" if input_bp >= 60 else "Missing/Low")
                },
                {
                    "Feature": "Serum Insulin",
                    "Entered Value": f"{input_insulin:.0f} μU/mL" if input_insulin > 0 else "Unrecorded (Imputed)",
                    "Reference Range": CLINICAL_REFERENCE_RANGES["Insulin"]["normal"],
                    "Status": "⚠️ Elevated" if input_insulin > 166 else "✅ Normal"
                },
                {
                    "Feature": "Age",
                    "Entered Value": f"{input_age} years",
                    "Reference Range": "Adult cohort (21+)",
                    "Status": "Elevated Risk" if input_age >= 45 else "Baseline"
                },
                {
                    "Feature": "Diabetes Pedigree",
                    "Entered Value": f"{input_pedigree:.3f}",
                    "Reference Range": CLINICAL_REFERENCE_RANGES["DiabetesPedigreeFunction"]["normal"],
                    "Status": "⚠️ High Familial History" if input_pedigree > 0.50 else "Average Familial Risk"
                }
            ]
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

            # Feature Impact Notice
            st.caption(
                "💡 **Clinical Decision Support Note:** In Logistic Regression, Glucose and BMI exert the highest "
                "positive odds ratios. Patients with elevated glucose and obesity will disproportionately push predicted probability above the decision threshold."
            )

# =============================================================================
# TAB 2: MODEL PERFORMANCE & CLINICAL METRICS
# =============================================================================
with tab_metrics:
    st.subheader("Hold-Out Test Set Performance & Evaluation")
    st.markdown(
        "Evaluation results computed strictly on the untouched 20% hold-out test set "
        "($N = 154$ patients, stratified by class balance)."
    )

    if metadata and "metrics" in metadata:
        m = metadata["metrics"]
        cm = metadata.get("confusion_matrix", {})

        # Top Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m.get('accuracy', 0)*100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m.get('sensitivity_recall', 0)*100:.1f}%</div>
                <div class="metric-label">Sensitivity (Recall)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m.get('specificity', 0)*100:.1f}%</div>
                <div class="metric-label">Specificity</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m.get('precision_ppv', 0)*100:.1f}%</div>
                <div class="metric-label">Precision (PPV)</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m.get('roc_auc', 0):.3f}</div>
                <div class="metric-label">ROC - AUC</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Detailed Breakdown & Visuals
        col_cm, col_curves = st.columns(2)

        with col_cm:
            st.markdown("##### 🔲 Clinical Confusion Matrix Breakdown")
            tn = cm.get("true_negatives", 0)
            fp = cm.get("false_positives", 0)
            fn = cm.get("false_negatives", 0)
            tp = cm.get("true_positives", 0)

            cm_display = pd.DataFrame(
                [
                    [f"True Negative (TN): {tn}", f"False Positive (FP): {fp}"],
                    [f"False Negative (FN): {fn}", f"True Positive (TP): {tp}"]
                ],
                index=["Actual Negative (Healthy)", "Actual Positive (Diabetic)"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            st.table(cm_display)

            cm_img_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Serialized Confusion Matrix (Saved during train_model.py execution)")

        with col_curves:
            st.markdown("##### 📈 Diagnostic Curves (ROC & Precision-Recall)")
            curves_img_path = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
            if os.path.exists(curves_img_path):
                st.image(curves_img_path, caption="Receiver Operating Characteristic and Precision-Recall Curves")
            else:
                st.info("Run `train_model.py` to generate high-resolution diagnostic curves.")

        st.divider()

        # In-depth Health Informatics Academic Discussion
        st.markdown("### 🎓 Health Informatics Concept: Why Accuracy Alone Fails in Healthcare")
        st.markdown(f"""
        1. **Prevalence and Class Imbalance:**
           - In clinical screening, target conditions are frequently in the minority. In this dataset, approximately **65%** of patients are non-diabetic and **35%** are diabetic.
           - A trivial 'dummy classifier' that predicts *every single patient is healthy* achieves **65% accuracy** without learning any pathology!
           - However, that dummy model has **0% Recall (Sensitivity)**, missing **100% of diabetic patients**.
        2. **Asymmetric Clinical Costs (False Negatives vs False Positives):**
           - **False Negative ({fn} patients in test set):** A diabetic patient is incorrectly diagnosed as healthy.
             *Clinical Result:* The patient receives no treatment, leading to unchecked hyperglycemia, accelerating nephropathy, retinopathy, and cardiovascular disease.
           - **False Positive ({fp} patients in test set):** A healthy individual is flagged as high risk.
             *Clinical Result:* The patient undergoes follow-up confirmatory testing (fasting glucose or HbA1c test) and experiences transient anxiety, which carries minimal permanent physical harm.
           - In medical diagnostics, **False Negatives are far more dangerous than False Positives**.
        3. **Clinical Recommendation:** Model performance in health informatics must always report **Sensitivity (Recall)**, **Specificity**, **Positive Predictive Value (PPV)**, and **ROC-AUC / PR-AUC** rather than raw Accuracy.
        """)

        # Odds Ratios Table
        if "feature_impact" in metadata:
            st.divider()
            st.markdown("##### 📊 Logistic Regression Clinical Interpretability (Odds Ratios)")
            st.markdown(
                "Because features were standardized ($Z$-scores), the exponentiated coefficients "
                "represent **Odds Ratios ($OR = e^\\beta$)** per standard deviation shift."
            )
            df_impact = pd.DataFrame(metadata["feature_impact"])
            st.dataframe(df_impact, hide_index=True, use_container_width=True)

# =============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab_eda:
    st.subheader("Cohort Exploration & Data Quality Inspection")
    st.markdown(
        "Inspection of the teaching dataset distributions, missing value diagnostics, "
        "and Pearson correlations."
    )

    if df is not None:
        col_eda1, col_eda2 = st.columns([1, 1])

        with col_eda1:
            st.markdown("##### 🎯 Target Class Distribution")
            class_img_path = os.path.join(FIGURES_DIR, "class_distribution.png")
            if os.path.exists(class_img_path):
                st.image(class_img_path, use_container_width=True)
            else:
                st.bar_chart(df["Outcome"].value_counts())

        with col_eda2:
            st.markdown("##### 🧬 Physiological Zero Value Audit")
            st.markdown(
                "In clinical data, zeros in physiological measurements are often invalid data collection artifacts rather than actual zero values."
            )
            zero_audit = []
            for col in FEATURE_COLUMNS:
                z_cnt = (df[col] == 0).sum()
                pct = (z_cnt / len(df)) * 100
                is_valid = "Physiologically Valid (Nulliparous)" if col == "Pregnancies" else (
                    "⚠️ Biologically Implausible (Treated as Missing)" if z_cnt > 0 else "✅ No Zeros"
                )
                zero_audit.append({
                    "Clinical Feature": col,
                    "Zero Count": z_cnt,
                    "Prevalence (%)": f"{pct:.1f}%",
                    "Health Informatics Classification": is_valid
                })
            st.dataframe(pd.DataFrame(zero_audit), hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("##### 🔬 Feature Distributions Stratified by Target Outcome")
        feat_img_path = os.path.join(FIGURES_DIR, "feature_distributions.png")
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, caption="Histograms and Kernel Density Estimates (KDE) per Feature", use_container_width=True)

        st.divider()
        st.markdown("##### 🌡️ Pearson Correlation Matrix")
        corr_img_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Pearson Correlation Heatmap (Invalid Zeros Treated as NaN)", use_container_width=True)

# =============================================================================
# TAB 4: GOVERNANCE, MONITORING & LIMITATIONS
# =============================================================================
with tab_governance:
    st.subheader("Model Limitations, Drift & Clinical AI Governance")
    st.markdown(
        "For an MSc in Health Informatics, deploying an ML algorithm requires deep understanding "
        "of algorithmic safety, data drift, population bias, and external multi-center validation."
    )

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("#### 1. Clinical & Algorithmic Limitations")
        st.markdown("""
        - **Demographic Homogeneity:** The training cohort represents solely adult females of Pima Indian heritage. It cannot generalize reliably to pediatric populations, adult males, or other global multi-ethnic cohorts without severe performance degradation.
        - **Absence of Modern Glycemic Biomarkers:** Modern clinical endocrinology heavily relies on **Hemoglobin A1c (HbA1c)**, which reflects average glycemia over the past 3 months. The dataset relies on older oral glucose tolerance tests (OGTT) and fasting insulin.
        - **Unmeasured Lifestyle & Clinical Confounders:** The dataset lacks critical determinants of health:
          - Physical activity levels
          - Dietary patterns and caloric intake
          - Concomitant medications (e.g., corticosteroids, antihypertensives)
          - Lipid panels (LDL, HDL, triglycerides)
        """)

        st.markdown("#### 2. Model Drift in Clinical Deployments")
        st.markdown("""
        Clinical AI models do not remain static in real-world healthcare environments:
        - **Data Drift (Covariate Shift):** $P(X)$ changes over time.
          *Example:* A hospital shifts its patient intake demographic from rural elderly to urban young adults, or lab assay manufacturers change calibration standards.
        - **Concept Drift:** $P(Y|X)$ changes over time.
          *Example:* The American Diabetes Association (ADA) revises the clinical diagnostic cutoff for fasting glucose (e.g., lowering prediabetes criteria from 100 mg/dL to 95 mg/dL).
        - **Monitoring Mitigation:** Clinical informatics pipelines must continuously compute statistical drift metrics (e.g., Population Stability Index [PSI], Wasserstein Distance) and trigger automated alerts when data distributions diverge.
        """)

    with g_col2:
        st.markdown("#### 3. Mandatory External Multi-Center Validation")
        st.markdown("""
        - **The Single-Center Overfitting Trap:** A machine learning model that achieves high accuracy on a single training hospital often experiences **catastrophic performance drop (15–30% drop in AUC)** when tested at an external clinic due to:
          - Differences in EHR documentation patterns
          - Measurement protocols (e.g., cuff size for blood pressure)
          - Socioeconomic and ethnic diversity
        - **Regulatory Pre-requisite:** Before any Clinical Decision Support System (CDSS) can receive regulatory clearance (e.g., US FDA 510(k), De Novo, or EU MDR Class IIa/IIb), it must undergo rigorous prospective, multi-center, multi-ethnic validation trials.
        """)

        st.markdown("#### 4. Algorithmic Fairness & Health Equity")
        st.markdown("""
        - **Disparate Impact:** Deploying a model trained on a specific subgroup into broader populations can systematically misallocate preventive resources.
        - **Health Informatics Standard:** Audit models for demographic parity, equalized odds, and calibration across protected demographic classes (age brackets, ethnicities, socioeconomic status).
        """)

    st.divider()
    st.info(
        "💡 **Examination Viva Summary:** When defending this ML pipeline, emphasize that the pipeline design "
        "is leak-free, interprets coefficients clinically via Odds Ratios, prioritizes Sensitivity over raw Accuracy, "
        "and recognizes the essential regulatory guardrails required prior to real-world healthcare translation."
    )
