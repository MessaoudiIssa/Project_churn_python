import pandas as pd
import streamlit as st
from preprocessing.data_cleaning import clean_customer_ids


@st.cache_data
def load_data():
    """
    Charge les données du dataset de churn Tunisie Telecom.
    Applique un nettoyage initial et retourne un DataFrame.

    Returns:
        DataFrame ou None en cas d'erreur
    """
    try:
        path = "C:/Users/issam/Desktop/PFE_master/churn_dataset_tunisie_telecom_project.csv"  # Assuming the file is in the project root
        df = pd.read_csv(path)

        # Nettoyage des noms de colonnes
        df.columns = [col.strip() for col in df.columns]

        # Vérification des colonnes
        expected_cols = {
            'numeric': ['Age', 'Tenure (Months)', 'Monthly Charges', 'Total Charges',
                        'Data Usage (GB)', 'Call Usage (Minutes)', 'Support Calls',
                        'Satisfaction Score', 'Churn'],
            'categorical': ['Location', 'Contract Type', 'Payment Method']
        }

        # Vérification et mapping des noms de colonnes
        col_mapping = {}
        missing_cols = []

        for col in expected_cols['numeric'] + expected_cols['categorical']:
            if col not in df.columns:
                simplified = col.lower().replace(' ', '').replace('(', '').replace(')', '')
                found = False
                for actual_col in df.columns:
                    if actual_col.lower().replace(' ', '').replace('(', '').replace(')', '') == simplified:
                        col_mapping[col] = actual_col
                        found = True
                        break
                if not found:
                    missing_cols.append(col)

        if missing_cols:
            st.error(f"Colonnes manquantes: {', '.join(missing_cols)}")
            return None

        # Renommage des colonnes
        df = df.rename(columns=col_mapping)

        # Conversion des types
        df['Churn'] = df['Churn'].astype(int)

        # Calcul CLV
        df['CLV'] = df['Total Charges'] * (1 - df['Churn'])

        # Gestion des CustomerID
        df = clean_customer_ids(df)

        return df
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None