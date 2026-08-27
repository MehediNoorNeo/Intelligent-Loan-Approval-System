
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        background: linear-gradient(135deg, #4c3bcf 0%, #5c3fd6 55%, #7048df 100%);
        padding: 28px 34px;
        border-radius: 0 0 24px 24px;
        margin: -1rem -1rem 1.5rem -1rem;
        color: white;
        box-shadow: 0 10px 30px rgba(76, 59, 207, 0.18);
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }

    .hero-subtitle {
        margin: 8px 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    .card {
        background: white;
        border: 1px solid #e7e9f2;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 6px 24px rgba(35, 38, 60, 0.06);
        margin-bottom: 18px;
    }

    .section-title {
        color: #4435b9;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 0.88rem;
        margin-bottom: 16px;
    }

    .result-approved {
        background: linear-gradient(135deg, #ecfdf5, #f0fdf9);
        border: 1px solid #b7efd2;
        border-radius: 18px;
        padding: 24px;
    }

    .result-rejected {
        background: linear-gradient(135deg, #fff1f2, #fff7f7);
        border: 1px solid #fecdd3;
        border-radius: 18px;
        padding: 24px;
    }

    .result-title-approved {
        color: #12915f;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 4px 0;
    }

    .result-title-rejected {
        color: #c53a4f;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 4px 0;
    }

    .result-text {
        color: #525866;
        margin: 0;
    }

    .prob-wrap {
        text-align: center;
        padding: 10px 0 2px;
    }

    .prob-ring {
        width: 180px;
        height: 180px;
        margin: 6px auto 18px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        background: conic-gradient(#4c3bcf var(--p), #e9eaf2 0);
    }

    .prob-ring::before {
        content: "";
        width: 132px;
        height: 132px;
        background: white;
        border-radius: 50%;
        position: absolute;
    }

    .prob-value {
        position: relative;
        z-index: 1;
        color: #24273a;
        font-size: 2rem;
        font-weight: 800;
    }

    .prob-caption {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 11px 0;
        border-bottom: 1px solid #eef0f5;
        font-size: 0.92rem;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: #6b7280;
    }

    .info-value {
        color: #25283a;
        font-weight: 650;
        text-align: right;
    }

    .note {
        background: #f5f3ff;
        border: 1px solid #e5e0ff;
        border-radius: 12px;
        padding: 12px 14px;
        color: #5147a7;
        font-size: 0.84rem;
        margin-top: 8px;
    }

    .footer-note {
        text-align: center;
        color: #8b90a0;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4c3bcf, #6646dc);
        border: none;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 1.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load fitted model + fitted preprocessing objects
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("model.pkl"),
        "num_imp": joblib.load("num_imp.pkl"),
        "cat_imp": joblib.load("cat_imp.pkl"),
        "oe": joblib.load("ordinal_encoder.pkl"),
        "ohe": joblib.load("onehot_encoder.pkl"),
        "scaler": joblib.load("scaler.pkl"),
    }


artifacts = load_artifacts()

model = artifacts["model"]
num_imp = artifacts["num_imp"]
cat_imp = artifacts["cat_imp"]
oe = artifacts["oe"]
ohe = artifacts["ohe"]
scaler = artifacts["scaler"]

# ------------------------------------------------------------
# Session state for result / reset
# ------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None


def reset_form():
    for key in list(st.session_state.keys()):
        if key.startswith("field_"):
            del st.session_state[key]
    st.session_state.prediction = None
    st.session_state.probability = None


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏦 Intelligent Loan Approval System</div>
        <div class="hero-subtitle">
            AI-powered loan approval prediction using your trained machine-learning model
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([7, 1.5])
with top_right:
    st.button("↻ Reset Form", on_click=reset_form, use_container_width=True)

# ------------------------------------------------------------
# Main two-column layout
# ------------------------------------------------------------
form_col, result_col = st.columns([1.15, 0.85], gap="large")

# ============================================================
# LEFT: Applicant form
# ============================================================
with form_col:

    st.markdown(
        """
        <div class="card">
            <div class="section-title">👤 Applicant Information</div>
            <div class="section-subtitle">
                Enter the applicant's financial and personal information.
            </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Financial ----------------
    st.markdown(
        '<div class="section-title">💼 Financial Information</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            key="field_applicant_income",
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            key="field_loan_amount",
        )

        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            key="field_credit_score",
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            value=1,
            step=1,
            key="field_existing_loans",
        )

        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            key="field_collateral_value",
        )

    with c2:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=2000.0,
            step=500.0,
            key="field_coapplicant_income",
        )

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=1,
            value=60,
            step=12,
            key="field_loan_term",
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            key="field_savings",
        )

        dti_ratio = st.slider(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01,
            key="field_dti_ratio",
        )

    # ---------------- Personal ----------------
    st.markdown("---")
    st.markdown(
        '<div class="section-title">👤 Personal Information</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        employment_status = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed"],
            key="field_employment_status",
        )

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            key="field_age",
        )

        education_level = st.selectbox(
            "Education Level",
            ["Not Graduate", "Graduate"],
            index=1,
            key="field_education_level",
        )

    with c2:
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married"],
            key="field_marital_status",
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            value=0,
            step=1,
            key="field_dependents",
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="field_gender",
        )

    # ---------------- Additional ----------------
    st.markdown("---")
    st.markdown(
        '<div class="section-title">📋 Additional Information</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"],
            key="field_loan_purpose",
        )

    with c2:
        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"],
            key="field_property_area",
        )

    with c3:
        employer_category = st.selectbox(
            "Employer Category",
            ["Government", "MNC", "Private", "Unemployed"],
            key="field_employer_category",
        )

    st.markdown("---")

    predict_clicked = st.button(
        "🚀 Predict Loan Approval",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<div class="footer-note">🔒 Your input is processed for prediction and is not stored by this interface.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Prediction helper
# ============================================================
def make_prediction():
    # Create one-row DataFrame with exactly the original feature names
    input_df = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment_status,
                "Age": age,
                "Marital_Status": marital_status,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education_level,
                "Gender": gender,
                "Employer_Category": employer_category,
            }
        ]
    )

    # 1. Same imputation as training
    num_cols = input_df.select_dtypes(include=["number"]).columns
    categ_cols = input_df.select_dtypes(include=["object"]).columns

    input_df[num_cols] = num_imp.transform(input_df[num_cols])
    input_df[categ_cols] = cat_imp.transform(input_df[categ_cols])

    # 2. Same ordinal encoding as training
    input_df[["Education_Level"]] = (
        oe.transform(input_df[["Education_Level"]]).astype(int)
    )

    # 3. Same one-hot encoding as training
    ohe_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category",
    ]

    encoded_cols = ohe.get_feature_names_out(ohe_cols)

    input_df[encoded_cols] = ohe.transform(input_df[ohe_cols])
    input_df.drop(columns=ohe_cols, inplace=True)

    # 4. Same feature engineering as training
    input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2

    # 5. Same scaling as training
    input_scaled = scaler.transform(input_df)

    # 6. Predict using the already-fitted instance
    prediction = int(model.predict(input_scaled)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_scaled)[0][1])

    return prediction, probability


if predict_clicked:
    with st.spinner("Analyzing applicant information..."):
        try:
            prediction, probability = make_prediction()
            st.session_state.prediction = prediction
            st.session_state.probability = probability
        except Exception as exc:
            st.error(
                "Prediction failed. Make sure the saved model and preprocessing "
                "files were created from the same notebook/environment."
            )
            st.exception(exc)

# ============================================================
# RIGHT: Results
# ============================================================
with result_col:

    prediction = st.session_state.prediction
    probability = st.session_state.probability

    if prediction is None:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">🛡️ Prediction Result</div>
                <div class="section-subtitle">
                    Your prediction will appear here after you submit the form.
                </div>
                <div class="note">
                    Complete the applicant information and click
                    <b>Predict Loan Approval</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Result card
        if prediction == 1:
            st.markdown(
                """
                <div class="result-approved">
                    <div class="section-title">🛡️ Prediction Result</div>
                    <div class="result-title-approved">✅ Loan Approved</div>
                    <p class="result-text">
                        The model predicts that the loan is likely to be approved.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-rejected">
                    <div class="section-title">🛡️ Prediction Result</div>
                    <div class="result-title-rejected">❌ Loan Not Approved</div>
                    <p class="result-text">
                        The model predicts that the loan is unlikely to be approved.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability card
        if probability is not None:
            pct = probability * 100
            pct_for_css = max(0, min(100, pct))

            if pct >= 75:
                caption = "High Probability"
            elif pct >= 50:
                caption = "Moderate Probability"
            else:
                caption = "Low Probability"

            st.markdown(
                f"""
                <div class="card">
                    <div class="section-title">📊 Approval Probability</div>
                    <div class="prob-wrap">
                        <div class="prob-ring" style="--p: {pct_for_css}%">
                            <div class="prob-value">{pct:.0f}%</div>
                        </div>
                        <div class="prob-caption">{caption}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Input summary
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">📌 Application Summary</div>

                <div class="info-row">
                    <span class="info-label">Credit Score</span>
                    <span class="info-value">{credit_score}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Applicant Income</span>
                    <span class="info-value">{applicant_income:,.0f}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Loan Amount</span>
                    <span class="info-value">{loan_amount:,.0f}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">DTI Ratio</span>
                    <span class="info-value">{dti_ratio:.2f}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Employment</span>
                    <span class="info-value">{employment_status}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Education</span>
                    <span class="info-value">{education_level}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model information
    st.markdown(
        """
        <div class="card">
            <div class="section-title">🤖 About the Model</div>
            <div class="section-subtitle">
                This application uses the fitted model and preprocessing objects
                produced by your Jupyter notebook.
            </div>

            <div class="info-row">
                <span class="info-label">Model</span>
                <span class="info-value">Gaussian Naive Bayes</span>
            </div>

            <div class="info-row">
                <span class="info-label">Accuracy</span>
                <span class="info-value">88.9%</span>
            </div>

            <div class="info-row">
                <span class="info-label">ROC-AUC</span>
                <span class="info-value">95.2%</span>
            </div>

            <div class="note">
                These metrics come from the model evaluation in your notebook.
                The deployed model should use the same preprocessing steps as training.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
