import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

st.subheader("Breast Cancer Wisconsin Dataset")

dataset = load_breast_cancer()
FEATURES = list(dataset.feature_names)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

missing = [
    f for f in MODEL_FILES.values()
    if not os.path.exists(os.path.join("model", f))
]

if missing:
    st.error("Models missing. Run train.py first.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required = FEATURES + ["target"]

    missing_columns = [
        c for c in required if c not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    if df.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    X_test = df[FEATURES]
    y_test = df["target"]

    if X_test.isnull().any().any() or y_test.isnull().any():
        st.error("CSV contains missing values.")
        st.stop()

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        st.error("Features must contain finite numeric values.")
        st.stop()

    if not set(y_test.unique()).issubset({0, 1}):
        st.error("Target must contain only 0 and 1.")
        st.stop()

    y_test = y_test.astype(int)

    if y_test.nunique() != 2:
        st.error("Both target classes are required for AUC.")
        st.stop()

    st.success("Dataset uploaded successfully!")

    # Load selected model
    model = joblib.load(
        os.path.join("model", MODEL_FILES[selected_model])
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    st.header("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", round(accuracy, 4))
    c2.metric("AUC", round(auc, 4))
    c3.metric("Precision", round(precision, 4))

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", round(recall, 4))
    c5.metric("F1 Score", round(f1, 4))
    c6.metric("MCC", round(mcc, 4))

    # Confusion matrix
    st.header("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.header("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    # Compare all models
    st.header("Comparison of All Five Models")

    comparison = []

    for name, filename in MODEL_FILES.items():

        current_model = joblib.load(
            os.path.join("model", filename)
        )

        pred = current_model.predict(X_test)

        prob = current_model.predict_proba(X_test)[:, 1]

        comparison.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred)
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True
    )

    # Accuracy chart
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Accuracy",
        hue="Model",
        legend=False,
        ax=ax
    )

    ax.tick_params(axis="x", rotation=20)

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Comparison CSV",
        comparison_df.to_csv(index=False),
        "model_comparison.csv",
        "text/csv"
    )

else:
    st.info("Upload test_data.csv to view evaluation results.")

