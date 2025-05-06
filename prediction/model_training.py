import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
import streamlit as st


def train_model(X, y, preprocessor, model_type='RandomForest'):
    """
    Entraîne un modèle de prédiction

    Args:
        X: Features
        y: Target (Churn)
        preprocessor: ColumnTransformer pour prétraitement
        model_type: Type de modèle ('RandomForest', 'GradientBoosting', 'DecisionTree')

    Returns:
        pipeline, metrics
    """
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=3, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'Model_Type': model_type,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUCROC': roc_auc_score(y_test, y_proba),
        'Confusion_Matrix': confusion_matrix(y_test, y_pred),
        'Classification_Report': classification_report(y_test, y_pred, output_dict=True),
        'Features': list(X.columns)
    }

    return pipeline, metrics