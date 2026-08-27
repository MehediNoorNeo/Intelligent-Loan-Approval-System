import streamlit as st
import pandas as pd
import joblib

# Load saved objects
num_imp = joblib.load("num_imp.pkl")
cat_imp = joblib.load("cat_imp.pkl")
oe = joblib.load("ordinal_encoder.pkl")
ohe = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.title("🏦 Intelligent Loan Approval System")

st.write(
    "Enter the applicant's information to predict whether "
    "the loan is likely to be approved."
)

# -----------------------------
# User Inputs
# -----------------------------

Applicant_Income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0
)

Coapplicant_Income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=2000.0
)

Employment_Status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed"]
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

Dependents = st.number_input(
    "Dependents",
    min_value=0,
    max_value=10,
    value=0
)

Credit_Score = st.number_input(
    "Credit Score",
    min_value=0.0,
    max_value=900.0,
    value=650.0
)

Existing_Loans = st.number_input(
    "Existing Loans",
    min_value=0,
    value=1
)

DTI_Ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0,
    value=0.30
)

Savings = st.number_input(
    "Savings",
    min_value=0.0,
    value=10000.0
)

Collateral_Value = st.number_input(
    "Collateral Value",
    min_value=0.0,
    value=30000.0
)

Loan_Amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=20000.0
)

Loan_Term = st.number_input(
    "Loan Term",
    min_value=1,
    value=60
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

Property_Area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

Education_Level = st.selectbox(
    "Education Level",
    ["Not Graduate", "Graduate"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Loan Approval"):

    # Create dataframe
    input_df = pd.DataFrame([{
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Employment_Status": Employment_Status,
        "Age": Age,
        "Marital_Status": Marital_Status,
        "Dependents": Dependents,
        "Credit_Score": Credit_Score,
        "Existing_Loans": Existing_Loans,
        "DTI_Ratio": DTI_Ratio,
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Loan_Purpose": Loan_Purpose,
        "Property_Area": Property_Area,
        "Education_Level": Education_Level,
        "Gender": Gender,
        "Employer_Category": Employer_Category
    }])

    # Same numerical/categorical columns used during training
    num_cols = input_df.select_dtypes(include=["number"]).columns
    categ_cols = input_df.select_dtypes(include=["object"]).columns

    # Imputation
    input_df[num_cols] = num_imp.transform(input_df[num_cols])
    input_df[categ_cols] = cat_imp.transform(input_df[categ_cols])

    # Ordinal encoding
    input_df[["Education_Level"]] = (
        oe.transform(input_df[["Education_Level"]])
        .astype(int)
    )

    # One-hot encoding
    ohe_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    encoded_cols = ohe.get_feature_names_out(ohe_cols)

    input_df[encoded_cols] = ohe.transform(
        input_df[ohe_cols]
    )

    input_df.drop(columns=ohe_cols, inplace=True)

    # Feature engineering
    input_df["Credit_Score_sq"] = (
        input_df["Credit_Score"] ** 2
    )

    input_df["DTI_Ratio_sq"] = (
        input_df["DTI_Ratio"] ** 2
    )

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1]
        st.write(f"Approval Probability: **{probability:.2%}**")