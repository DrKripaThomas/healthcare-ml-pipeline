"""
Healthcare Diabetes Risk Classification Pipeline
================================================
Academic Project: MSc Health Informatics and Analytics
Module: Machine Learning Workflow & Clinical Decision Support Systems

This script executes an end-to-end Machine Learning pipeline:
    1. Problem Definition & Target Identification
    2. Data Ingestion & Quality Audit (Data Types, Duplicates, Invalid Values)
    3. Physiological Zero Handling (Treating Zeroes as Missing Values)
    4. Exploratory Data Analysis (EDA) & Figure Generation
    5. Leak-Free Train-Test Splitting (Stratified 80/20 Split)
    6. Scikit-learn Pipeline Construction (Imputation -> Scaling -> Logistic Regression)
    7. Model Training (Fitted Strictly on Training Data)
    8. Comprehensive Clinical Model Evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC)
    9. Feature Importance / Odds Ratio Interpretation
    10. Serialization of Pipeline Artifacts (.joblib) & Metadata (.json)

DISCLAIMER: For academic and educational demonstration only. Not clinically validated.
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DiabetesMLPipeline")

DATA_PATH = os.path.join("data", "diabetes.csv")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "figures")
PIPELINE_OUTPUT_PATH = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Biologically invalid zero columns in human physiology:
INVALID_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET_COLUMN = "Outcome"
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


def ensure_directories():
    """Ensure destination directories for artifacts and figures exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Step 1 & 2: Data Ingestion & Quality Audit
# -----------------------------------------------------------------------------
def load_and_audit_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset and perform health informatics data quality checks.
    
    Verifies:
      - File existence
      - Schema and column matching
      - Duplicate row detection
      - Data types
      - Biological zero prevalence
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please ensure 'data/diabetes.csv' exists "
            "before executing the training pipeline."
        )

    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)

    logger.info("Dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Data types:\n%s", df.dtypes.to_string())

    # Check for duplicate records
    duplicate_count = df.duplicated().sum()
    logger.info("Duplicate patient records detected: %d", duplicate_count)
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped duplicates. New shape: %s", df.shape)

    # Validate target presence
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target variable '{TARGET_COLUMN}' not found in dataset columns.")

    # Audit biological zeroes
    logger.info("--- Biological Zero Audit ---")
    for col in INVALID_ZERO_COLUMNS:
        zero_count = (df[col] == 0).sum()
        pct = (zero_count / len(df)) * 100
        logger.info(
            "Feature '%s': %d zero measurements (%.1f%%) [Physiologically implausible -> Missing]",
            col, zero_count, pct
        )

    valid_zero_pregnancies = (df["Pregnancies"] == 0).sum()
    logger.info(
        "Feature 'Pregnancies': %d zero values (%.1f%%) [Physiologically valid: Nulliparous]",
        valid_zero_pregnancies, (valid_zero_pregnancies / len(df)) * 100
    )

    return df


# -----------------------------------------------------------------------------
# Step 3: Exploratory Data Analysis & Visualizations
# -----------------------------------------------------------------------------
def generate_eda_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate and save academic EDA visualizations for the research report.
      - Class balance distribution
      - Feature distribution histograms
      - Feature correlation heatmap
    """
    logger.info("Generating EDA figures in: %s", output_dir)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Target Class Balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    bars = ax.bar(
        ["Negative (Non-Diabetic, 0)", "Positive (Diabetic, 1)"],
        counts.values,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        width=0.5
    )
    for bar, pct in zip(bars, percentages.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 10,
            f"{int(yval)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    ax.set_title("Diabetes Cohort Class Distribution (Class Imbalance)", fontsize=12, pad=12)
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 2. Feature Distributions Subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            ax=axes[idx],
            palette={0: "#2b5c8f", 1: "#d95f02"},
            alpha=0.6
        )
        axes[idx].set_title(f"Distribution: {feature}", fontsize=11, fontweight="semibold")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel("Frequency")
    plt.suptitle("Clinical Feature Distributions Stratified by Diabetes Outcome", fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    # Replace zeros temporarily to compute truthful correlations
    df_corr = df.copy()
    df_corr[INVALID_ZERO_COLUMNS] = df_corr[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    corr_matrix = df_corr.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Pearson Correlation Heatmap (Invalid Zeros Treated as NaN)", fontsize=12, pad=12)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logger.info("EDA figures successfully saved.")


# -----------------------------------------------------------------------------
# Step 4: Preprocessing & Data Leakage Prevention
# -----------------------------------------------------------------------------
def prepare_train_test_split(df: pd.DataFrame):
    """
    Converts physiologically invalid 0s to NaN, then partitions data.
    
    CRITICAL HEALTH INFORMATICS PRINCIPLE:
    We must split into train and test sets BEFORE applying any statistical
    imputation or scaling. Computing medians or standard deviations across
    the entire dataset prior to splitting leaks test distribution information
    into the training process ('Data Leakage').
    """
    df_clean = df.copy()
    
    # Replace physiologically implausible zeros with NaN
    df_clean[INVALID_ZERO_COLUMNS] = df_clean[INVALID_ZERO_COLUMNS].replace(0, np.nan)

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    logger.info(
        "Splitting dataset: 80%% Train, 20%% Test with Stratification (random_state=42)"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    logger.info(
        "Train set: %d samples (%d negative, %d positive)",
        len(y_train), (y_train == 0).sum(), (y_train == 1).sum()
    )
    logger.info(
        "Test set:  %d samples (%d negative, %d positive)",
        len(y_test), (y_test == 0).sum(), (y_test == 1).sum()
    )

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# Step 5 & 6: Pipeline Construction & Model Training
# -----------------------------------------------------------------------------
def build_and_train_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Constructs an end-to-end Scikit-learn Pipeline:
      1. SimpleImputer(strategy='median'):
         Imputes missing values (former 0s) using the median of training data.
         Median is preferred over mean in clinical data because lab values
         such as Insulin and SkinThickness have positive skewness.
      2. StandardScaler():
         Standardizes features to zero mean and unit variance (Z-score).
         Essential for Logistic Regression because unscaled features with large
         ranges (e.g. Insulin up to 800) would dominate gradient optimization
         and distort coefficient interpretation.
      3. LogisticRegression(random_state=42, max_iter=1000):
         Transparent, highly interpretable probabilistic generalized linear model
         widely accepted in Clinical Decision Support Systems (CDSS).
    """
    logger.info("Building Scikit-learn Pipeline [SimpleImputer -> StandardScaler -> LogisticRegression]")

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    logger.info("Fitting pipeline strictly on training data (X_train)...")
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline fitting complete.")

    return pipeline


# -----------------------------------------------------------------------------
# Step 7: Model Evaluation & Clinical Metric Analysis
# -----------------------------------------------------------------------------
def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str) -> dict:
    """
    Evaluates the fitted pipeline on the untouched hold-out test set.
    Generates academic visualizations and prints clinical interpretations.
    """
    logger.info("--- Hold-Out Test Set Evaluation ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Core Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("Accuracy:             %.4f (%.2f%%)", acc, acc * 100)
    logger.info("Sensitivity (Recall): %.4f (%.2f%%) [Ability to detect true diabetic patients]", rec, rec * 100)
    logger.info("Specificity:          %.4f (%.2f%%) [Ability to identify healthy individuals]", spec, spec * 100)
    logger.info("Precision (PPV):      %.4f (%.2f%%) [Confidence when predicting high risk]", prec, prec * 100)
    logger.info("F1-Score:             %.4f", f1)
    logger.info("ROC-AUC Score:        %.4f", roc_auc)
    logger.info("PR-AUC Score:         %.4f", pr_auc)

    print("\n" + "=" * 70)
    print("CLINICAL EVALUATION REPORT & METRIC EXPLANATION")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))

    print("-" * 70)
    print("HEALTH INFORMATICS NOTE: WHY ACCURACY ALONE IS MISLEADING")
    print("-" * 70)
    baseline_negative_ratio = (y_test == 0).sum() / len(y_test)
    print(
        f"1. Class Imbalance: In this cohort, {baseline_negative_ratio:.1%} of patients are negative.\n"
        f"   A naive 'dummy classifier' that predicts every patient is healthy would\n"
        f"   achieve {baseline_negative_ratio:.1%} accuracy, but have 0.0% Recall — missing 100% of diabetic patients!\n"
        f"2. Cost of Errors in Healthcare:\n"
        f"   - False Negative ({fn} patients): Diabetic individuals incorrectly classified as healthy.\n"
        f"     CLINICAL CONSEQUENCE: Delayed intervention, disease progression, organ damage.\n"
        f"   - False Positive ({fp} patients): Healthy individuals classified as high risk.\n"
        f"     CLINICAL CONSEQUENCE: Second-line confirmatory testing (HbA1c/OGTT) and transient anxiety,\n"
        f"     which is far less dangerous than an untreated condition.\n"
        f"3. Conclusion: In healthcare screening, Recall (Sensitivity) and ROC-AUC are vastly\n"
        f"   more clinically significant than raw overall Accuracy."
    )
    print("=" * 70 + "\n")

    # Generate Evaluation Plots
    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_labels = [
        [f"True Negative\n(TN = {tn})", f"False Positive\n(FP = {fp})"],
        [f"False Negative\n(FN = {fn})", f"True Positive\n(TP = {tp})"]
    ]
    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
        yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax.set_title("Clinical Confusion Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC and Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="semibold")
    ax1.legend(loc="lower right")

    # Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(rec_vals, prec_vals, color="#d95f02", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
    prevalence = (y_test == 1).mean()
    ax2.axhline(prevalence, color="grey", linestyle="--", label=f"Prevalence Baseline ({prevalence:.2f})")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (Positive Predictive Value)")
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="semibold")
    ax2.legend(loc="lower left")

    plt.suptitle("Clinical Diagnostic Performance Curves", fontsize=13, y=1.02)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # Calculate Feature Odds Ratios from Logistic Regression
    clf = pipeline.named_steps["classifier"]
    coefficients = clf.coef_[0]
    odds_ratios = np.exp(coefficients)

    feature_impact = []
    for feat, coef, odds in zip(FEATURE_COLUMNS, coefficients, odds_ratios):
        feature_impact.append({
            "feature": feat,
            "coefficient": float(round(coef, 4)),
            "odds_ratio": float(round(odds, 4)),
            "interpretation": (
                f"1 standard deviation increase multiplies diabetes odds by {odds:.2f}x"
                if odds > 1 else
                f"1 standard deviation increase reduces diabetes odds by {1/odds:.2f}x"
            )
        })

    logger.info("--- Logistic Regression Feature Odds Ratios ---")
    for item in feature_impact:
        logger.info("%s: Coef = %.4f | Odds Ratio = %.4f", item["feature"], item["coefficient"], item["odds_ratio"])

    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "Logistic Regression (Pipeline)",
        "train_samples": int(len(y_test) * 4),  # 80/20 split
        "test_samples": int(len(y_test)),
        "metrics": {
            "accuracy": float(round(acc, 4)),
            "sensitivity_recall": float(round(rec, 4)),
            "specificity": float(round(spec, 4)),
            "precision_ppv": float(round(prec, 4)),
            "f1_score": float(round(f1, 4)),
            "roc_auc": float(round(roc_auc, 4)),
            "pr_auc": float(round(pr_auc, 4))
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "features": FEATURE_COLUMNS,
        "feature_impact": feature_impact,
        "clinical_disclaimer": "For educational/research purposes only. Not clinically validated."
    }

    return metrics_payload


# -----------------------------------------------------------------------------
# Step 8: Serialization & Artifact Export
# -----------------------------------------------------------------------------
def save_artifacts(pipeline: Pipeline, metadata: dict):
    """Save the fitted pipeline and evaluation metadata to disk."""
    logger.info("Saving fitted model pipeline to: %s", PIPELINE_OUTPUT_PATH)
    joblib.dump(pipeline, PIPELINE_OUTPUT_PATH)

    logger.info("Saving model evaluation metadata to: %s", METADATA_OUTPUT_PATH)
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("All model artifacts saved successfully.")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting Healthcare Diabetes Risk Classification Training Pipeline")
    ensure_directories()

    # 1. Load and Audit
    df = load_and_audit_data(DATA_PATH)

    # 2. Exploratory Data Analysis & Plots
    generate_eda_visualizations(df, REPORTS_DIR)

    # 3. Train-Test Split (Avoiding Data Leakage)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    # 4. Build & Train Pipeline
    pipeline = build_and_train_pipeline(X_train, y_train)

    # 5. Model Evaluation
    metadata = evaluate_model(pipeline, X_test, y_test, REPORTS_DIR)

    # 6. Save Artifacts
    save_artifacts(pipeline, metadata)

    logger.info("Pipeline execution completed with complete success.")


if __name__ == "__main__":
    main()
