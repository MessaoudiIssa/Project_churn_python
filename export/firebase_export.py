import pandas as pd
from datetime import datetime
import streamlit as st
from config.firebase_config import init_firebase
from preprocessing.data_cleaning import clean_customer_ids


def export_to_firestore(df, model_results=None, predictions=None, limit=1000):
    """
    Exporte les données vers Firestore

    Args:
        df: DataFrame contenant les données client
        model_results: Résultats des modèles (dict)
        predictions: DataFrame contenant les prédictions
        limit: Nombre maximum de documents à exporter

    Returns:
        Boolean indiquant si l'export a réussi
    """
    try:
        db = init_firebase()
        if db is None:
            return False

        # Préparation des données
        export_data = df.head(limit).copy()

        # Nettoyage des CustomerID
        export_data = clean_customer_ids(export_data)

        # Ajout des prédictions si disponibles
        if predictions is not None:
            pred_columns = ['Future_Churn_Probability', 'Predicted_Churn']
            for col in pred_columns:
                if col in predictions.columns:
                    export_data[col] = predictions[col].values[:limit]

        # Conversion des données pour Firestore
        records = export_data.to_dict('records')

        # Export Firestore
        batch_size = 500
        client_ref = db.collection("Clients")

        with st.progress(0.0) as progress_bar:
            for i in range(0, len(records), batch_size):
                batch = db.batch()
                batch_records = records[i:i + batch_size]

                for record in batch_records:
                    doc_ref = client_ref.document(record['CustomerID'])
                    batch.set(doc_ref, record)

                batch.commit()
                progress_bar.progress(min(1.0, (i + batch_size) / len(records)))

        # Export des résultats des modèles si fournis
        if model_results is not None:
            results_ref = db.collection("ModelResults").document("Latest")
            results_ref.set({
                "Timestamp": datetime.now().isoformat(),
                "Results": model_results
            })

        return True
    except Exception as e:
        st.error(f"Erreur d'export: {str(e)}")
        return False