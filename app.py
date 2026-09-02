"""
app.py
------
Streamlit Web Application for CreditWise Loan Approval System.
Provides an interactive form for entering applicant details and predicts
loan approval status using the trained Random Forest model and pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CreditWise - AI Loan Approval System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern UI look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
    }
    .approved-badge {
        background-color: #DCFCE7;
        color: #15803D;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #86EFAC;
    }
    .rejected-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #FCA5A5;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_saved_model(filepath: str = 'loan_approval_model.joblib'):
    """Load cached model pipeline artifact."""
    if not os.path.exists(filepath):
        st.error(f"Model file '{filepath}' not found! Please run `python train_model.py` first.")
        st.stop()
    artifact = joblib.load(filepath)
    return artifact


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=70)
    st.sidebar.title("CreditWise Portal")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About the System")
    st.sidebar.info(
        "CreditWise uses machine learning trained on historical borrower data "
        "to evaluate credit risk and predict loan approval decisions instantly."
    )

    # Load Model Artifact
    artifact = load_saved_model()
    preprocessor = artifact['preprocessor']
    model = artifact['model']
    model_name = artifact['model_name']
    feature_names = artifact['feature_names']
    metrics_df = artifact['comparison_metrics']

    best_f1 = metrics_df.loc[metrics_df['Model'] == model_name, 'F1-Score'].values[0]
    best_acc = metrics_df.loc[metrics_df['Model'] == model_name, 'Accuracy'].values[0]

    st.sidebar.markdown("### 📊 Active Model Info")
    st.sidebar.write(f"**Model Engine:** {model_name}")
    st.sidebar.write(f"**Test F1-Score:** `{best_f1:.2%}`")
    st.sidebar.write(f"**Test Accuracy:** `{best_acc:.2%}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Instructions")
    st.sidebar.write("1. Complete applicant income and credit information.")
    st.sidebar.write("2. Input requested loan amount and terms.")
    st.sidebar.write("3. Click **Evaluate Loan Application** to view AI prediction and risk factors.")

    # Header
    st.markdown('<div class="main-header">💳 CreditWise Loan Approval System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Risk Assessment & Loan Underwriting Platform</div>', unsafe_allow_html=True)

    # Application Form
    st.subheader("📋 Enter Applicant Details")

    with st.form(key='loan_application_form'):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Financial & Demographic Profile")
            applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=100000, value=12000, step=500)
            coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=500)
            savings = st.number_input("Savings / Liquid Assets ($)", min_value=0, max_value=200000, value=15000, step=1000)
            age = st.slider("Applicant Age", min_value=18, max_value=80, value=35)
            gender = st.selectbox("Gender", options=['Male', 'Female'])
            marital_status = st.selectbox("Marital Status", options=['Married', 'Single'])
            education_level = st.selectbox("Education Level", options=['Graduate', 'Not Graduate'])
            employment_status = st.selectbox("Employment Status", options=['Salaried', 'Self-employed', 'Contract', 'Unemployed'])
            employer_category = st.selectbox("Employer Category", options=['Private', 'Government', 'MNC', 'Business', 'Unemployed'])

        with col2:
            st.markdown("#### 🏦 Credit & Loan Request Details")
            credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=740)
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=200000, value=25000, step=1000)
            loan_term = st.selectbox("Loan Term (Months)", options=[12, 24, 36, 48, 60, 72, 84], index=4)
            loan_purpose = st.selectbox("Loan Purpose", options=['Home', 'Car', 'Business', 'Education', 'Personal'])
            collateral_value = st.number_input("Collateral Value ($)", min_value=0, max_value=500000, value=30000, step=1000)
            dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
            dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3])
            existing_loans = st.selectbox("Number of Existing Loans", options=[0, 1, 2, 3, 4])
            property_area = st.selectbox("Property Area", options=['Urban', 'Semiurban', 'Rural'])

        submit_button = st.form_submit_button(label="🔍 Evaluate Loan Application", use_container_width=True)

    if submit_button:
        # Construct input DataFrame matching original schema
        input_data = pd.DataFrame([{
            'Applicant_Income': float(applicant_income),
            'Coapplicant_Income': float(coapplicant_income),
            'Employment_Status': employment_status,
            'Age': float(age),
            'Marital_Status': marital_status,
            'Dependents': float(dependents),
            'Credit_Score': float(credit_score),
            'Existing_Loans': float(existing_loans),
            'DTI_Ratio': float(dti_ratio),
            'Savings': float(savings),
            'Collateral_Value': float(collateral_value),
            'Loan_Amount': float(loan_amount),
            'Loan_Term': float(loan_term),
            'Loan_Purpose': loan_purpose,
            'Property_Area': property_area,
            'Education_Level': education_level,
            'Gender': gender,
            'Employer_Category': employer_category
        }])

        # Transform using preprocessing pipeline
        input_processed = preprocessor.transform(input_data)

        # Make Prediction
        pred_class = model.predict(input_processed)[0]

        if hasattr(model, 'predict_proba'):
            approval_prob = model.predict_proba(input_processed)[0, 1]
        else:
            approval_prob = float(pred_class)

        st.markdown("---")
        st.subheader("🎯 Evaluation Results & Risk Assessment")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### Decision Outcome")
            if pred_class == 1:
                st.markdown('<div class="approved-badge">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
                st.success(f"Application met all underwriting risk guidelines with an approval confidence of **{approval_prob:.1%}**.")
            else:
                st.markdown('<div class="rejected-badge">❌ LOAN NOT APPROVED</div>', unsafe_allow_html=True)
                st.error(f"Application exceeds acceptable risk limits. Approval probability is only **{approval_prob:.1%}**.")

        with res_col2:
            st.markdown("#### Approval Probability Score")
            st.metric("Approval Likelihood", f"{approval_prob:.1%}")
            st.progress(float(approval_prob))

        # Key Factors / Feature Importance Explanation
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.subheader("💡 Key Decision Drivers (Model Feature Importances)")
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(7)

            imp_df['Importance (%)'] = (imp_df['Importance'] * 100).round(2)

            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.bar_chart(imp_df.set_index('Feature')['Importance'])

            with table_col:
                st.write("**Top 7 Influential Factors:**")
                st.dataframe(imp_df[['Feature', 'Importance (%)']], use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
