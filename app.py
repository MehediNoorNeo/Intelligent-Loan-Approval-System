
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Loan Decision Studio",
    page_icon="◆",
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

# Professional dashboard theme — navy communicates trust, teal signals action,
# and amber provides a restrained highlight colour for key information.
st.markdown(
    """
    <style>
    :root { --navy:#102A43; --deep:#071A2B; --teal:#0F8B8D; --teal-dark:#0B6B6D; --gold:#F4B942; --ink:#1F3448; --muted:#61758A; --line:#DCE5EE; }
    .stApp {
        background: radial-gradient(circle at 93% 0%, rgba(15,139,141,.10), transparent 22rem), radial-gradient(circle at 1% 26%, rgba(244,185,66,.10), transparent 18rem), #F4F7FB;
        color:var(--ink);
    }
    [data-testid="stHeader"] { background:transparent; }
    #MainMenu, footer { visibility:hidden; }
    .block-container { max-width:1440px; padding-top:1.15rem; padding-bottom:2.75rem; }
    .hero { background:linear-gradient(118deg, #071A2B 0%, #102A43 58%, #0B6B6D 145%); padding:30px 36px; border-radius:22px; margin:0 0 1rem; box-shadow:0 18px 45px rgba(7,26,43,.20); position:relative; overflow:hidden; }
    .hero::after { content:""; position:absolute; width:230px; height:230px; border:1px solid rgba(255,255,255,.14); border-radius:50%; right:-58px; top:-110px; box-shadow:-38px 42px 0 -1px rgba(255,255,255,.08); }
    .hero-eyebrow { color:#8BE3DD; letter-spacing:.13em; text-transform:uppercase; font-size:.69rem; font-weight:800; margin-bottom:8px; }
    .hero-title { font-size:2.28rem; letter-spacing:-.035em; line-height:1.12; }
    .hero-subtitle { max-width:660px; color:#D5E4ED; font-size:.96rem; }
    .hero-stat { color:#D5E4ED; font-size:.8rem; margin-top:16px; } .hero-stat b { color:#F6CD72; }
    .card { background:rgba(255,255,255,.94); border:1px solid var(--line); border-radius:16px; padding:22px 24px; box-shadow:0 8px 25px rgba(28,58,84,.055); margin-bottom:16px; }
    .section-title { color:var(--navy); font-size:1.02rem; font-weight:780; letter-spacing:-.012em; }
    .section-subtitle, .info-label, .prob-caption { color:var(--muted); }
    .result-approved { background:linear-gradient(135deg,#E9FAF6,#F7FFFC); border-color:#AEE4D3; border-radius:16px; box-shadow:0 8px 24px rgba(10,128,101,.08); }
    .result-rejected { background:linear-gradient(135deg,#FFF2F2,#FFF9F8); border-color:#F4C5C5; border-radius:16px; box-shadow:0 8px 24px rgba(177,54,54,.07); }
    .result-title-approved { color:#087A62; } .result-title-rejected { color:#B33B45; } .result-text { color:#536779; }
    .prob-ring { background:conic-gradient(var(--teal) var(--p), #E7EDF2 0); box-shadow:inset 0 0 0 7px rgba(255,255,255,.76); } .prob-value, .info-value { color:var(--navy); }
    .info-row { border-bottom-color:#EDF1F5; } .note { background:#F0F7F8; border-color:#C7E2E2; color:#176F72; } .footer-note { color:#718399; }
    div.stButton > button { border-radius:10px; min-height:44px; font-weight:750; transition:transform .18s ease, box-shadow .18s ease; }
    div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#0F8B8D,#0B6B6D); border:none; box-shadow:0 8px 16px rgba(15,139,141,.22); }
    div.stButton > button[kind="primary"]:hover { background:linear-gradient(100deg,#0B7779,#075A5C); transform:translateY(-1px); box-shadow:0 11px 20px rgba(15,139,141,.25); }
    div.stButton > button[kind="secondary"] { color:var(--navy); border:1px solid #BED0DD; background:rgba(255,255,255,.72); }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-color:#C8D5E0 !important; border-radius:9px !important; background:#FFF !important; }
    div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:var(--teal) !important; box-shadow:0 0 0 3px rgba(15,139,141,.12) !important; }
    [data-testid="stWidgetLabel"] p { color:#385166 !important; font-size:.86rem !important; font-weight:650 !important; }
    [data-testid="stSlider"] [role="slider"] { background-color:var(--teal); } hr { border-color:#E3EAF0; margin:1.35rem 0 !important; }
    .form-kicker { display:inline-flex; color:var(--teal-dark); background:#E7F5F4; border:1px solid #C7E6E3; border-radius:999px; padding:5px 10px; font-size:.72rem; font-weight:800; letter-spacing:.04em; text-transform:uppercase; margin-bottom:10px; }
    .empty-visual { width:48px; height:48px; display:grid; place-items:center; background:#E7F5F4; color:var(--teal-dark); border-radius:13px; font-size:1.35rem; margin-bottom:14px; }
    @media (max-width:900px) { .hero { padding:25px; } .hero-title { font-size:1.65rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Accent layer: colour is used as navigation and emphasis, not decoration alone.
st.markdown(
    """
    <style>
    :root { --sky:#3DA9FC; --violet:#7C5CFC; --mint:#47D7AC; --coral:#FF7A7A; --sun:#F8C85E; }
    .hero { background:linear-gradient(118deg,#071A2B 0%,#102A43 48%,#165C70 100%); }
    .hero::before { content:""; position:absolute; width:310px; height:310px; border-radius:50%; right:80px; bottom:-235px; background:radial-gradient(circle,rgba(71,215,172,.30),transparent 67%); }
    .palette-strip { display:flex; gap:7px; position:absolute; right:34px; bottom:28px; z-index:2; }
    .palette-strip span { width:11px; height:11px; border-radius:50%; box-shadow:0 0 0 4px rgba(255,255,255,.10); }
    .palette-strip .p-teal { background:var(--mint); } .palette-strip .p-sky { background:var(--sky); } .palette-strip .p-violet { background:var(--violet); } .palette-strip .p-gold { background:var(--sun); }
    .form-kicker { color:#5C43CA; background:#F0EDFF; border-color:#D9D1FF; }
    .stApp { background:radial-gradient(circle at 8% 8%,rgba(124,92,252,.12),transparent 20rem), radial-gradient(circle at 95% 28%,rgba(61,169,252,.14),transparent 25rem), radial-gradient(circle at 51% 97%,rgba(71,215,172,.13),transparent 27rem), #EEF3F8; }
    .card { border:1px solid #D7E2EC; border-top-width:1px; box-shadow:0 13px 30px rgba(31,65,94,.08); }
    .form-card { border-color:#D8CFFF; background:linear-gradient(135deg,#F6F3FF 0%,#EEE9FF 100%); }
    .empty-card { border-color:#BFDFFF; background:linear-gradient(135deg,#EEF8FF 0%,#DFF2FF 100%); }
    .probability-card { border-color:#B9E7DF; background:linear-gradient(135deg,#ECFBF7 0%,#DDF7EF 100%); }
    .summary-card { border-color:#F1D99B; background:linear-gradient(135deg,#FFF9E9 0%,#FFF0C9 100%); }
    .model-card { border-color:#C5D9FA; background:linear-gradient(135deg,#F0F5FF 0%,#E4EEFF 100%); }
    .empty-visual { background:linear-gradient(135deg,#E4F4FF,#E8FBF4); color:#147A85; box-shadow:0 8px 18px rgba(61,169,252,.16); }
    .note { background:linear-gradient(100deg,#F0FAFA,#F5F9FF); }
    .result-approved { border-left:5px solid var(--mint); } .result-rejected { border-left:5px solid var(--coral); }
    .prob-ring::before { background:linear-gradient(135deg,#FFFFFF,#F6FCFC); }
    .info-row { position:relative; } .info-row:hover { background:#F6FAFC; margin-inline:-8px; padding-inline:8px; border-radius:7px; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { box-shadow:0 2px 6px rgba(16,42,67,.025); }
    div.stButton > button[kind="secondary"]:hover { color:var(--violet); border-color:#B9ABFF; background:#F5F3FF; }
    @media (max-width:900px) { .palette-strip { display:none; } }
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
        <div class="hero-eyebrow">Loan decision intelligence</div>
        <div class="hero-title">Loan Decision Studio</div>
        <div class="hero-subtitle">
            A clear, data-led view of each loan application—powered by your trained machine-learning model.
        </div>
        <div class="hero-stat"><b>●</b> Secure prediction workspace &nbsp;·&nbsp; Model confidence included</div>
        <div class="palette-strip" aria-label="Brand colour palette"><span class="p-teal"></span><span class="p-sky"></span><span class="p-violet"></span><span class="p-gold"></span></div>
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
        <div class="card form-card">
            <div class="form-kicker">Application workspace</div>
            <div class="section-title">👤 Applicant Information</div>
            <div class="section-subtitle">
                Complete the profile below to generate an informed lending recommendation.
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
            <div class="card empty-card">
                <div class="empty-visual">✦</div>
                <div class="section-title">Ready for a decision</div>
                <div class="section-subtitle">
                    Your approval outcome and model confidence will appear here.
                </div>
                <div class="note">
                    Review the applicant profile, then select <b>Predict Loan Approval</b> to begin the assessment.
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
                f"""<div class="card probability-card">
<div class="section-title">📊 Approval Probability</div>
<div class="prob-wrap">
<div class="prob-ring" style="--p: {pct_for_css}%"><div class="prob-value">{pct:.0f}%</div></div>
<div class="prob-caption">{caption}</div>
</div>
</div>""",
                unsafe_allow_html=True,
            )

        # Input summary
        # Keep HTML flush-left: Markdown treats four leading spaces as a code block.
        st.markdown(
            f"""<div class="card summary-card">
<div class="section-title">📌 Application Summary</div>
<div class="info-row"><span class="info-label">Credit Score</span><span class="info-value">{credit_score}</span></div>
<div class="info-row"><span class="info-label">Applicant Income</span><span class="info-value">{applicant_income:,.0f}</span></div>
<div class="info-row"><span class="info-label">Loan Amount</span><span class="info-value">{loan_amount:,.0f}</span></div>
<div class="info-row"><span class="info-label">DTI Ratio</span><span class="info-value">{dti_ratio:.2f}</span></div>
<div class="info-row"><span class="info-label">Employment</span><span class="info-value">{employment_status}</span></div>
<div class="info-row"><span class="info-label">Education</span><span class="info-value">{education_level}</span></div>
</div>""",
            unsafe_allow_html=True,
        )

    # Model information
    # Keep HTML flush-left: indented lines are interpreted as a Markdown code block.
    st.markdown(
        """<div class="card model-card">
<div class="section-title">🤖 About the Model</div>
<div class="section-subtitle">This application uses the fitted model and preprocessing objects produced by your Jupyter notebook.</div>
<div class="info-row"><span class="info-label">Model</span><span class="info-value">Gaussian Naive Bayes</span></div>
<div class="info-row"><span class="info-label">Accuracy</span><span class="info-value">88.9%</span></div>
<div class="info-row"><span class="info-label">ROC-AUC</span><span class="info-value">95.2%</span></div>
<div class="note">These metrics come from the model evaluation in your notebook. The deployed model should use the same preprocessing steps as training.</div>
</div>""",
        unsafe_allow_html=True,
    )
