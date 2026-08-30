from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from reporting import build_prediction_pdf


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_approval.csv"
MODEL_PATH = ROOT / "models" / "loan_approval_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def applicant_form() -> dict:
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount", min_value=1, value=130, step=10)

    with col3:
        loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good / 1" if x == 1.0 else "Poor / 0")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    return {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }


def predict_one(artifact: dict, applicant: dict) -> tuple[str, float]:
    model = artifact["model"]
    row = pd.DataFrame([applicant], columns=artifact["features"])
    pred = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(probabilities[pred])
    return artifact["inverse_target_mapping"][pred], confidence


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Loan_Status", hue="Loan_Status", palette=["#C74E4E", "#2F855A"], legend=False, ax=ax)
    ax.set_title("Approval Outcome Count")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Applications")
    st.pyplot(fig)


def plot_approval_rate(df: pd.DataFrame, column: str) -> None:
    chart_df = df.groupby(column)["Loan_Status"].apply(lambda s: (s == "Y").mean()).reset_index(name="Approval Rate")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=chart_df, x=column, y="Approval Rate", color="#31708E", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Approval Rate")
    ax.set_title(f"Approval Rate by {column}")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


def feature_importance(artifact: dict) -> pd.DataFrame | None:
    model = artifact["model"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_") and not hasattr(classifier, "coef_"):
        return None

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        values = abs(classifier.coef_[0])

    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    importance["Feature"] = importance["Feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    return importance.sort_values("Importance", ascending=False).head(15)


def prediction_page(artifact: dict) -> None:
    st.subheader("Single Applicant Prediction")
    applicant = applicant_form()

    if st.button("Predict Loan Status", type="primary"):
        label, confidence = predict_one(artifact, applicant)
        status_type = "success" if label == "Approved" else "error"
        getattr(st, status_type)(f"Prediction: {label}")
        st.metric("Model confidence", f"{confidence:.1%}")

        pdf = build_prediction_pdf(applicant, label, confidence)
        st.download_button(
            "Download PDF Report",
            data=pdf,
            file_name="loan_prediction_report.pdf",
            mime="application/pdf",
        )


def dashboard_page(df: pd.DataFrame, artifact: dict) -> None:
    st.subheader("Loan Approval Analytics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applications", f"{len(df):,}")
    col2.metric("Approval Rate", f"{(df['Loan_Status'].eq('Y').mean()):.1%}")
    col3.metric("Median Loan Amount", f"{df['LoanAmount'].median():.0f}")
    col4.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    chart1, chart2 = st.columns(2)
    with chart1:
        plot_target_distribution(df)
    with chart2:
        plot_approval_rate(df, "Credit_History")

    chart3, chart4 = st.columns(2)
    with chart3:
        plot_approval_rate(df, "Property_Area")
    with chart4:
        plot_approval_rate(df, "Education")

    st.subheader("Model Comparison")
    if METRICS_PATH.exists():
        st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True)

    st.subheader("Feature Importance")
    importance = feature_importance(artifact)
    if importance is None:
        st.info("The selected model does not expose feature importance.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance, y="Feature", x="Importance", color="#4C78A8", ax=ax)
        ax.set_title("Top Model Drivers")
        st.pyplot(fig)
        st.dataframe(importance, use_container_width=True)


def batch_page(artifact: dict) -> None:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV with the same input columns as the training data, excluding `Loan_Status`.")
    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded is None:
        return

    batch = pd.read_csv(uploaded)
    missing = [col for col in artifact["features"] if col not in batch.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    model = artifact["model"]
    X = batch[artifact["features"]]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = batch.copy()
    result["Prediction"] = [artifact["inverse_target_mapping"][int(value)] for value in predictions]
    result["Confidence"] = [float(probabilities[i, int(predictions[i])]) for i in range(len(predictions))]

    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download Predictions CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="loan_batch_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Loan Approval Prediction System")
    st.caption("Machine-learning capstone app for predicting loan approval outcomes.")

    df = load_data()
    artifact = load_artifact()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page", ["Predict", "Dashboard", "Batch Prediction"])
    st.sidebar.divider()
    st.sidebar.write(f"Best model: **{artifact['model_name']}**")

    if page == "Predict":
        prediction_page(artifact)
    elif page == "Dashboard":
        dashboard_page(df, artifact)
    else:
        batch_page(artifact)


if __name__ == "__main__":
    main()
