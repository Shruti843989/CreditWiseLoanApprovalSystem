"""
train_model.py
--------------
Model Training, Cross-Validation, Hyperparameter Tuning, Model Comparison,
and Model Export for CreditWise Loan Approval System.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from data_pipeline import (
    load_and_clean_data,
    build_preprocessing_pipeline,
    get_feature_names
)


def train_and_evaluate_models(data_path: str = 'loan_approval_data.csv'):
    """
    Load data, split into train/test, tune hyperparameters across 5 candidate models,
    evaluate on metrics (Accuracy, Precision, Recall, F1, ROC-AUC), select the best
    model by F1-Score, and export the combined pipeline & model to joblib.
    """
    print("==================================================")
    print("STEP 1: LOADING & SPLITTING DATA")
    print("==================================================")
    X, y = load_and_clean_data(data_path)
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: Approved={y.sum()} ({y.mean():.1%}), Rejected={(y==0).sum()} ({(1-y.mean()):.1%})")

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    print("\n==================================================")
    print("STEP 2: FITTING PREPROCESSING PIPELINE")
    print("==================================================")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"Processed feature matrix shape: {X_train_proc.shape}")

    print("\n==================================================")
    print("STEP 3: HYPERPARAMETER TUNING & MODEL TRAINING (5-FOLD CV)")
    print("==================================================")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Candidate models & hyperparameter grids
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs'],
                'class_weight': ['balanced', None]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-7, 1e-5]
            }
        }
    }

    results = []
    trained_models = {}

    for name, config in models_config.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            error_score=0
        )
        grid.fit(X_train_proc, y_train)

        best_estimator = grid.best_estimator_
        trained_models[name] = best_estimator

        # Predictions on Test set
        y_pred = best_estimator.predict(X_test_proc)

        if hasattr(best_estimator, "predict_proba"):
            y_proba = best_estimator.predict_proba(X_test_proc)[:, 1]
        elif hasattr(best_estimator, "decision_function"):
            y_proba = best_estimator.decision_function(X_test_proc)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Best Parameters': str(grid.best_params_),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    # Summary Comparison Table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)

    print("\n==================================================")
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("==================================================")
    printable_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    print(printable_df.to_string(index=False))

    # Select Best Model based on F1-Score
    best_model_name = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1-Score']
    best_model = trained_models[best_model_name]

    print("\n==================================================")
    print(f"BEST MODEL SELECTED: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("==================================================")

    # Save artifact containing preprocessing pipeline + model + metadata
    export_object = {
        'preprocessor': preprocessor,
        'model': best_model,
        'model_name': best_model_name,
        'feature_names': feature_names,
        'comparison_metrics': results_df
    }

    output_path = 'loan_approval_model.joblib'
    joblib.dump(export_object, output_path)
    print(f"Full model & preprocessing pipeline saved to {output_path}")

    return results_df, best_model_name, export_object


if __name__ == '__main__':
    train_and_evaluate_models()
