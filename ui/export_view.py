import streamlit as st
from data.data_loader import load_data
from export.firebase_export import export_to_firestore
from datetime import datetime


def render_export():
    """Affiche la page d'export vers Firebase"""
    st.header("Export vers Firebase")

    # Chargement des données
    if 'df' not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Impossible de charger les données. Vérifiez le chemin du fichier CSV.")
        return

    df = st.session_state.df

    # Interface utilisateur
    st.write("""
    Cette page vous permet d'exporter vos données et prédictions vers Firebase
    pour les rendre accessibles à d'autres applications ou services.
    """)

    st.info("Seuls les 1000 premiers clients seront exportés par défaut.")

    # Options d'export
    export_limit = st.slider(
        "Nombre de clients à exporter",
        min_value=10,
        max_value=min(10000, len(df)),
        value=min(1000, len(df))
    )

    export_predictions = st.checkbox(
        "Inclure les prédictions",
        value=True if 'predictions' in st.session_state and st.session_state.predictions is not None else False
    )

    export_models = st.checkbox(
        "Inclure les résultats des modèles",
        value=True if 'model_metrics' in st.session_state and st.session_state.model_metrics else False
    )

    # Vérification de la configuration Firebase
    import os
    firebase_config_exists = os.path.exists("serviceAccountKey.json")

    if not firebase_config_exists:
        st.error("""
        Le fichier de configuration Firebase (serviceAccountKey.json) est introuvable.
        Veuillez placer ce fichier à la racine du projet pour activer l'export.
        """)

    # Bouton d'export
    export_disabled = not firebase_config_exists

    if st.button("Exporter les données", disabled=export_disabled):
        with st.spinner("Export en cours..."):
            # Préparation des résultats de modèle
            model_results = None
            if export_models and 'model_metrics' in st.session_state and st.session_state.model_metrics:
                model_results = [
                    {
                        'Model': key,
                        'Accuracy': value['Accuracy'],
                        'Precision': value['Precision'],
                        'Recall': value['Recall'],
                        'F1': value['F1'],
                        'AUCROC': value['AUCROC'],
                        'Timestamp': datetime.now().isoformat()
                    }
                    for key, value in st.session_state.model_metrics.items()
                ]

            # Préparation des prédictions
            predictions = None
            if export_predictions and 'predictions' in st.session_state and st.session_state.predictions is not None:
                predictions = st.session_state.predictions

            # Export
            success = export_to_firestore(
                df,
                model_results if export_models else None,
                predictions if export_predictions else None,
                limit=export_limit
            )

            if success:
                st.success(f"Export réussi de {export_limit} clients vers Firebase!")
                st.balloons()
            else:
                st.error("Échec de l'export. Vérifiez la configuration Firebase.")

    # Informations sur l'utilisation des données exportées
    st.subheader("Utilisation des données exportées")
    st.write("""
    Les données exportées vers Firebase peuvent être utilisées pour:

    1. **Visualisation dans des tableaux de bord externes**
    2. **Intégration avec des applications mobiles**
    3. **Partage avec d'autres services Tunisie Telecom**
    4. **Alimentation des systèmes CRM**
    5. **Automatisation des campagnes marketing**
    """)

    # Structure des données
    with st.expander("Structure des données exportées"):
        st.code("""
{
  "Clients": {
    "CUST_000001": {
      "CustomerID": "CUST_000001",
      "Age": 35,
      "Tenure (Months)": 24,
      "Monthly Charges": 65.4,
      "Total Charges": 1569.6,
      "Data Usage (GB)": 45.2,
      "Call Usage (Minutes)": 328.5,
      "Support Calls": 2,
      "Satisfaction Score": 3,
      "Location": "Tunis",
      "Contract Type": "Month-to-month",
      "Payment Method": "Credit card",
      "Churn": 0,
      "CLV": 1569.6,
      "Future_Churn_Probability": 0.23,
      "Predicted_Churn": 0
    },
    // Autres clients...
  },
  "ModelResults": {
    "Latest": {
      "Timestamp": "2023-10-15T14:32:45.123456",
      "Results": [
        {
          "Model": "RandomForest",
          "Accuracy": 0.85,
          "Precision": 0.78,
          "Recall": 0.72,
          "F1": 0.75,
          "AUCROC": 0.82,
          "Timestamp": "2023-10-15T14:32:45.123456"
        },
        // Autres modèles...
      ]
    }
  }
}
        """, language="json")