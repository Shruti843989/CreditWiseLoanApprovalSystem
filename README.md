# 💳 CreditWise Loan Approval System

An automated, machine learning-powered loan underwriting and risk assessment system. Rebuilt from scratch into a modular, production-ready Python application featuring automated data preprocessing, 5-fold cross-validated hyperparameter tuning, model comparison, joblib pipeline serialization, and an interactive Streamlit UI.

---

## 📌 Project Overview & Features

1. **Automated Data Pipeline (`data_pipeline.py`)**:
   - Cleans raw applicant data and handles target missingness (`Loan_Approved = NaN`).
   - Imputes missing numerical features using **median** and categorical features using **mode**.
   - Derives 6 financial risk features:
     - `Total_Income`: Combines primary applicant and coapplicant income to capture overall household earning power.
     - `Income_to_Loan_Ratio`: Measures household income relative to requested loan amount to evaluate debt service capacity.
     - `Collateral_to_Loan_Ratio`: Assesses asset backing protecting lender capital in default scenarios.
     - `Savings_to_Loan_Ratio`: Evaluates liquid savings cushion available to handle income disruption.
     - `Credit_Score_Band`: Categorizes credit score into discrete policy risk tiers (Poor, Fair, Good, Excellent).
     - `DTI_Risk_Category`: Bins Debt-To-Income ratios into policy risk buckets (Low, Medium, High).
   - Encodes ordinal features (`OrdinalEncoder`) and nominal features (`OneHotEncoder`).
   - Scales numeric variables with `StandardScaler`.
   - Exports the reusable pipeline via `joblib`.

2. **Model Training & Comparison (`train_model.py`)**:
   - Performs stratified train/test split (80/20).
   - Trains and tunes 5 algorithms using 5-fold Stratified Cross-Validation (`GridSearchCV`):
     1. Logistic Regression
     2. Random Forest Classifier
     3. Gradient Boosting Classifier
     4. K-Nearest Neighbors (KNN)
     5. Gaussian Naive Bayes
   - Evaluates all models on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Selects the best performing model based on **F1-Score**.
   - Exports the combined preprocessing pipeline and trained model into `loan_approval_model.joblib`.

3. **Interactive Streamlit Frontend (`app.py`)**:
   - Pure Python Streamlit application (`streamlit run app.py`).
   - Clean, 2-column input layout for financial & demographic profile and credit & loan request details.
   - Instant decision output with color-coded approval badge, confidence score, and top decision driver chart.

---

## 📊 Model Performance Comparison Results

Evaluation on the holdout test set (190 samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest** (Selected) | **95.26%** | **89.23%** | **96.67%** | **0.9280** | **0.9791** |
| 🚀 Gradient Boosting | 95.26% | 90.48% | 95.00% | 0.9268 | 0.9849 |
| 📈 Logistic Regression | 84.21% | 70.83% | 85.00% | 0.7727 | 0.9249 |
| 🔔 Naive Bayes | 83.68% | 73.02% | 76.67% | 0.7480 | 0.8906 |
| 🎯 KNN | 80.53% | 71.70% | 63.33% | 0.6726 | 0.8796 |

> **Selected Model**: **Random Forest** achieved the highest test F1-Score (**0.9280**) balancing exceptional recall (**96.67%**) and precision (**89.23%**).

---

## 🛠️ Project Structure

```text
CreditWiseLoanApprovalSystem/
├── data_pipeline.py          # Data cleaning, feature engineering, and pipeline fitting
├── train_model.py           # Model cross-validation, hyperparameter tuning & export
├── app.py                   # Streamlit web application frontend
├── loan_approval_data.csv   # Raw dataset (1000 rows, 20 columns)
├── loan_approval_model.joblib # Serialized model + pipeline artifact
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & instructions
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Preprocessing
```bash
python data_pipeline.py
```

### 3. Run Model Training & Evaluation
```bash
python train_model.py
```

### 4. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
