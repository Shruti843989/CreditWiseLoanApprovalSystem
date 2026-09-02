"""
data_pipeline.py
----------------
Data Loading, Cleaning, Feature Engineering, and Preprocessing Pipeline
for CreditWise Loan Approval System.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


def load_and_clean_data(file_path: str = 'loan_approval_data.csv'):
    """
    Load data from CSV, handle target missing values, and drop unneeded columns.

    Parameters:
    -----------
    file_path : str
        Path to the loan approval CSV file.

    Returns:
    --------
    X : pd.DataFrame
        Feature dataset.
    y : pd.Series
        Binary target variable (1 for 'Yes', 0 for 'No').
    """
    df = pd.read_csv(file_path)

    # Drop rows where target variable Loan_Approved is missing
    df = df.dropna(subset=['Loan_Approved']).copy()

    # Convert target to binary (1 for 'Yes', 0 for 'No')
    df['Loan_Approved'] = df['Loan_Approved'].map({'Yes': 1, 'No': 0})

    # Drop Applicant_ID if present as it has no predictive power
    if 'Applicant_ID' in df.columns:
        df = df.drop(columns=['Applicant_ID'])

    X = df.drop(columns=['Loan_Approved'])
    y = df['Loan_Approved'].astype(int)

    return X, y


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for domain-specific feature engineering.

    Features Created & Justification:
    ---------------------------------
    1. Total_Income = Applicant_Income + Coapplicant_Income
       - Justification: Combines primary applicant and coapplicant income to evaluate overall household earning power.
    2. Income_to_Loan_Ratio = Total_Income / (Loan_Amount + 1)
       - Justification: Measures total income relative to requested loan amount to assess repayment capacity.
    3. Collateral_to_Loan_Ratio = Collateral_Value / (Loan_Amount + 1)
       - Justification: Evaluates asset coverage protecting lender capital in event of default.
    4. Savings_to_Loan_Ratio = Savings / (Loan_Amount + 1)
       - Justification: Assesses liquid cash reserves available to buffer temporary financial shocks.
    5. Credit_Score_Band = Binned Credit Score into 4 discrete tiers: Poor, Fair, Good, Excellent
       - Justification: Captures non-linear credit risk thresholds used in standard credit scoring models.
    6. DTI_Risk_Category = Binned Debt-to-Income ratio into 3 risk tiers: Low, Medium, High
       - Justification: Categorizes borrower debt burden into policy risk bands.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        app_inc = X_out['Applicant_Income'].fillna(0)
        coapp_inc = X_out['Coapplicant_Income'].fillna(0)
        loan_amt = X_out['Loan_Amount'].fillna(1)
        collateral = X_out['Collateral_Value'].fillna(0)
        savings = X_out['Savings'].fillna(0)

        # Feature 1: Total Household Income
        X_out['Total_Income'] = app_inc + coapp_inc

        # Feature 2: Income to Loan Ratio
        X_out['Income_to_Loan_Ratio'] = X_out['Total_Income'] / (loan_amt + 1.0)

        # Feature 3: Collateral to Loan Ratio
        X_out['Collateral_to_Loan_Ratio'] = collateral / (loan_amt + 1.0)

        # Feature 4: Savings to Loan Ratio
        X_out['Savings_to_Loan_Ratio'] = savings / (loan_amt + 1.0)

        # Feature 5: Credit Score Band (Categorical/Ordinal)
        def categorize_credit_score(score):
            if pd.isna(score):
                return 'Fair'
            if score < 600:
                return 'Poor'
            elif score < 700:
                return 'Fair'
            elif score < 800:
                return 'Good'
            else:
                return 'Excellent'

        X_out['Credit_Score_Band'] = X_out['Credit_Score'].apply(categorize_credit_score)

        # Feature 6: DTI Risk Category (Categorical/Ordinal)
        def categorize_dti(dti):
            if pd.isna(dti):
                return 'Medium'
            if dti < 0.30:
                return 'Low'
            elif dti <= 0.50:
                return 'Medium'
            else:
                return 'High'

        X_out['DTI_Risk_Category'] = X_out['DTI_Ratio'].apply(categorize_dti)

        return X_out


def get_feature_names(pipeline):
    """Utility function to extract feature names out of ColumnTransformer."""
    feature_names = []

    # Features from ColumnTransformer inside pipeline
    transformer = pipeline.named_steps['preprocessor']
    num_cols = transformer.transformers_[0][2]
    ord_cols = transformer.transformers_[1][2]
    nom_encoder = transformer.transformers_[2][1].named_steps['encoder']
    nom_cols_input = transformer.transformers_[2][2]
    nom_cols = list(nom_encoder.get_feature_names_out(nom_cols_input))

    feature_names.extend(num_cols)
    feature_names.extend(ord_cols)
    feature_names.extend(nom_cols)

    return feature_names


def build_preprocessing_pipeline():
    """
    Builds the complete Scikit-Learn preprocessing pipeline incorporating imputation,
    feature engineering, categorical encoding (Ordinal & OneHot), and numerical scaling.

    Returns:
    --------
    pipeline : Pipeline
        Full preprocessing pipeline.
    """
    num_cols = [
        'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
        'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
        'Collateral_Value', 'Loan_Amount', 'Loan_Term',
        'Total_Income', 'Income_to_Loan_Ratio', 'Collateral_to_Loan_Ratio',
        'Savings_to_Loan_Ratio'
    ]

    ordinal_cols = ['Education_Level', 'Property_Area', 'Credit_Score_Band', 'DTI_Risk_Category']
    ordinal_categories = [
        ['Not Graduate', 'Graduate'],
        ['Rural', 'Semiurban', 'Urban'],
        ['Poor', 'Fair', 'Good', 'Excellent'],
        ['Low', 'Medium', 'High']
    ]

    nominal_cols = [
        'Employment_Status', 'Marital_Status', 'Loan_Purpose',
        'Gender', 'Employer_Category'
    ]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scaler', StandardScaler())
    ])

    nom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('ord', ord_pipeline, ordinal_cols),
            ('nom', nom_pipeline, nominal_cols)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', column_transformer)
    ])

    return full_pipeline


def save_pipeline(pipeline, file_path: str = 'preprocessing_pipeline.joblib'):
    """Save fitted preprocessing pipeline using joblib."""
    joblib.dump(pipeline, file_path)
    print(f"Preprocessing pipeline saved successfully to {file_path}")


def load_pipeline(file_path: str = 'preprocessing_pipeline.joblib'):
    """Load preprocessing pipeline using joblib."""
    pipeline = joblib.load(file_path)
    print(f"Preprocessing pipeline loaded from {file_path}")
    return pipeline


if __name__ == '__main__':
    X, y = load_and_clean_data('loan_approval_data.csv')
    print(f"Dataset loaded successfully. X shape: {X.shape}, Target distribution:\n{y.value_counts()}")

    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X)
    print(f"Preprocessing pipeline executed successfully. Output shape: {X_processed.shape}")
    feature_names = get_feature_names(pipeline)
    print(f"Total features created ({len(feature_names)}): {feature_names}")

    save_pipeline(pipeline)
