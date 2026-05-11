import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(page_title="Loan Status - Logistic Regression", layout="wide")
st.title("Loan Status Prediction (Logistic Regression)")

DATA_PATH = "train_u6lujuX_CVtuZ9i.csv"

FEATURE_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]
TARGET_COL = "Loan_Status"

NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleanup: strip spaces from object columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "NaN", "None"]), col] = np.nan

    # Coerce numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    # Drop rows without target
    df = df.copy()
    df = df[df[TARGET_COL].notna()]

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Map target to ensure consistent labels
    y = y.astype(str).str.strip()
    y = y.replace({"Y": "Y", "N": "N"})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Y", "N"])

    return clf, acc, cm


df = load_data(DATA_PATH)

# Train and show metrics
with st.spinner("Training model..."):
    clf, accuracy, cm = train_model(df)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Test Accuracy", f"{accuracy:.3f}")
with col2:
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual Y", "Actual N"], columns=["Pred Y", "Pred N"]))

st.divider()

st.subheader("Enter Applicant Details")

# UI inputs
with st.form("predict_form"):
    gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
    married = st.selectbox("Married", options=["Yes", "No"], index=0)
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"], index=0)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"], index=0)
    self_employed = st.selectbox("Self_Employed", options=["Yes", "No"], index=0)

    applicant_income = st.number_input("ApplicantIncome", min_value=0, value=3000, step=100)
    coapplicant_income = st.number_input(
        "CoapplicantIncome", min_value=0, value=0, step=100
    )
    loan_amount = st.number_input("LoanAmount", min_value=0.0, value=120.0, step=1.0)
    loan_amount_term = st.selectbox(
        "Loan_Amount_Term",
        options=[
            6,
            12,
            18,
            24,
            30,
            36,
            42,
            48,
            60,
            72,
            84,
            96,
            120,
            180,
            240,
            300,
            360,
        ],
        index=13,
        help="Typical values appear in the dataset (in months).",
    )

    credit_history = st.selectbox(
        "Credit_History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Yes" if int(x) == 1 else "No",
    )
    property_area = st.selectbox("Property_Area", options=["Urban", "Rural", "Semiurban"], index=0)

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": int(credit_history),
                "Property_Area": property_area,
            }
        ]
    )

    pred = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]
    class_index = list(clf.named_steps["model"].classes_).index(pred)
    pred_prob = float(proba[class_index])

    st.success(f"Predicted Loan_Status: {pred}")
    st.write(f"Confidence (predicted probability): {pred_prob:.3f}")

    # Probability breakdown
    st.caption("Class probabilities")
    classes = clf.named_steps["model"].classes_
    prob_table = pd.DataFrame({"class": classes, "probability": proba})
    st.table(prob_table)

