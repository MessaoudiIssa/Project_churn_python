import pandas as pd
import re
import streamlit as st


def clean_customer_ids(df):
    """
    Uniformise les formats des CustomerID

    Args:
        df: DataFrame contenant les données client

    Returns:
        DataFrame avec CustomerID standardisés
    """
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = ['CUST_' + str(i).zfill(6) for i in range(1, len(df) + 1)]
        return df

    # Conversion en string et nettoyage
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip().str.upper()

    # Standardisation du format
    def standardize_id(cust_id):
        # Extraction des chiffres
        numbers = re.sub(r'[^0-9]', '', cust_id)
        if not numbers:
            return None
        return 'CUST_' + numbers.zfill(6)

    df['CustomerID'] = df['CustomerID'].apply(standardize_id)

    # Réindexation si des IDs sont invalides
    if df['CustomerID'].isnull().any():
        st.warning("Certains CustomerID étaient invalides et ont été réinitialisés")
        df['CustomerID'] = ['CUST_' + str(i).zfill(6) for i in range(1, len(df) + 1)]

    # Vérification des doublons
    if df['CustomerID'].duplicated().any():
        st.warning("Doublons détectés dans les CustomerID - réinitialisation")
        df['CustomerID'] = ['CUST_' + str(i).zfill(6) for i in range(1, len(df) + 1)]

    return df


def prepare_data(df):
    """
    Prépare les données pour le machine learning

    Args:
        df: DataFrame contenant les données client

    Returns:
        X, y, preprocessor, numeric_features, categorical_features
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    numeric_features = ['Age', 'Tenure (Months)', 'Monthly Charges', 'Total Charges',
                        'Data Usage (GB)', 'Call Usage (Minutes)', 'Support Calls',
                        'Satisfaction Score']

    categorical_features = ['Location', 'Contract Type', 'Payment Method']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X = df[numeric_features + categorical_features]
    y = df['Churn']

    return X, y, preprocessor, numeric_features, categorical_features