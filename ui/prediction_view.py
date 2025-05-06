import streamlit as st
import pandas as pd
from data.data_loader import load_data
from preprocessing.data_cleaning import prepare_data
from prediction.model_training import train_model
from prediction.model_prediction import predict_future_churn, predict_for_individual, visualize_predictions
from utils.helpers import display_model_evaluation


def render_prediction(evaluation_only=False):
    """
    Affiche la page de prédiction de churn

    Args:
        evaluation_only: Si True, n'affiche que l'évaluation des modèles
    """
    if evaluation_only:
        st.header("Évaluation des Modèles")
    else:
        st.header("Prédiction de Churn")

    # Chargement des données
    if 'df' not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Impossible de charger les données. Vérifiez le chemin du fichier CSV.")
        return

    df = st.session_state.df

    # Préparation des données
    X, y, preprocessor, numeric_features, categorical_features = prepare_data(df)

    # Initialisation session state pour les modèles
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    if evaluation_only:
        # Interface d'évaluation des modèles
        st.write("""
        Cette page vous permet d'évaluer et comparer différents modèles
        de prédiction de churn pour trouver celui qui convient le mieux à vos données.
        """)

        # Options d'évaluation
        col1, col2 = st.columns(2)
        with col1:
            evaluate_rf = st.checkbox("Évaluer Random Forest", True)
        with col2:
            evaluate_gb = st.checkbox("Évaluer Gradient Boosting", True)
        evaluate_dt = st.checkbox("Évaluer Arbre de Décision", True)

        if st.button("Lancer l'évaluation des modèles"):
            with st.spinner("Évaluation en cours..."):
                if evaluate_rf and 'RandomForest' not in st.session_state.model_metrics:
                    pipeline, metrics = train_model(X, y, preprocessor, 'RandomForest')
                    st.session_state.trained_models['RandomForest'] = pipeline
                    st.session_state.model_metrics['RandomForest'] = metrics

                if evaluate_gb and 'GradientBoosting' not in st.session_state.model_metrics:
                    pipeline, metrics = train_model(X, y, preprocessor, 'GradientBoosting')
                    st.session_state.trained_models['GradientBoosting'] = pipeline
                    st.session_state.model_metrics['GradientBoosting'] = metrics

                if evaluate_dt and 'DecisionTree' not in st.session_state.model_metrics:
                    pipeline, metrics = train_model(X, y, preprocessor, 'DecisionTree')
                    st.session_state.trained_models['DecisionTree'] = pipeline
                    st.session_state.model_metrics['DecisionTree'] = metrics

        # Affichage des résultats
        if st.session_state.model_metrics:
            display_model_evaluation(st.session_state.model_metrics)
        else:
            st.warning("Aucun modèle évalué pour le moment. Cliquez sur 'Lancer l'évaluation'.")

    else:
        # Interface de prédiction
        st.write("""
        Cette page vous permet de prédire le risque de churn pour vos clients,
        soit individuellement, soit pour un groupe de clients.
        """)

        # Choix du type de prédiction
        prediction_type = st.radio("Type de prédiction",
                                   ["Prédire pour un seul client", "Prédire pour un groupe de clients"])

        model_type = st.selectbox("Modèle à utiliser",
                                  ['RandomForest', 'GradientBoosting', 'DecisionTree'])

        months = st.slider("Période de prédiction (mois)", 1, 12, 3)

        if prediction_type == "Prédire pour un seul client":
            # Interface pour prédiction individuelle
            st.subheader("Saisie des caractéristiques du client")

            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=120, value=12)
                monthly_charges = st.number_input("Charges mensuelles", min_value=0, value=50)
            with col2:
                total_charges = st.number_input("Charges totales", min_value=0, value=600)
                data_usage = st.number_input("Usage données (GB)", min_value=0, value=10)
                call_usage = st.number_input("Usage appel (minutes)", min_value=0, value=300)
            with col3:
                support_calls = st.number_input("Appels support", min_value=0, value=1)
                satisfaction = st.number_input("Score satisfaction", min_value=1, max_value=5, value=3)
                location = st.selectbox("Localisation", df['Location'].unique())
                contract_type = st.selectbox("Type de contrat", df['Contract Type'].unique())
                payment_method = st.selectbox("Méthode paiement", df['Payment Method'].unique())

            if st.button("Prédire le churn pour ce client"):
                # Création d'un dataframe avec les données du client
                client_data = pd.DataFrame([[
                    age, tenure, monthly_charges, total_charges,
                    data_usage, call_usage, support_calls, satisfaction,
                    location, contract_type, payment_method
                ]], columns=numeric_features + categorical_features)

                # Entraînement du modèle s'il n'existe pas déjà
                if model_type not in st.session_state.trained_models:
                    with st.spinner(f"Entraînement du modèle {model_type} en cours..."):
                        pipeline, metrics = train_model(X, y, preprocessor, model_type)
                        st.session_state.trained_models[model_type] = pipeline
                        st.session_state.model_metrics[model_type] = metrics
                else:
                    pipeline = st.session_state.trained_models[model_type]

                # Prédiction
                client_data = predict_for_individual(pipeline, client_data)

                # Affichage résultat
                st.subheader("Résultat de la prédiction")
                proba = client_data['Future_Churn_Probability'].iloc[0]
                prediction = "OUI" if client_data['Predicted_Churn'].iloc[0] else "NON"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité de churn", f"{proba:.2%}")
                with col2:
                    st.metric("Prédiction de churn", prediction)

                # Recommandations basées sur la prédiction
                st.subheader("Recommandations")
                if proba > 0.7:
                    st.error("Client à très haut risque de churn!")
                    st.write("""
                    Actions recommandées:
                    - Contacter immédiatement le client
                    - Offrir une réduction ou un avantage significatif
                    - Résoudre les problèmes potentiels en priorité
                    """)
                elif proba > 0.4:
                    st.warning("Client à risque modéré de churn")
                    st.write("""
                    Actions recommandées:
                    - Proposer une offre de fidélisation
                    - Enquête de satisfaction
                    - Améliorer les services utilisés fréquemment
                    """)
                else:
                    st.success("Client à faible risque de churn")
                    st.write("""
                    Actions recommandées:
                    - Maintenir la qualité de service
                    - Proposer des services complémentaires
                    - Programme de parrainage
                    """)

        else:  # Prédiction pour groupe de clients
            st.subheader("Prédiction pour un groupe de clients")
            n_clients = st.slider("Nombre de clients à prédire", 1, min(10000, len(df)), min(1000, len(df)))

            if st.button("Lancer la prédiction pour le groupe"):
                # Entraînement du modèle s'il n'existe pas déjà
                if model_type not in st.session_state.trained_models:
                    with st.spinner(f"Entraînement du modèle {model_type} en cours..."):
                        pipeline, metrics = train_model(X, y, preprocessor, model_type)
                        st.session_state.trained_models[model_type] = pipeline
                        st.session_state.model_metrics[model_type] = metrics
                else:
                    pipeline = st.session_state.trained_models[model_type]

                # Prédiction
                with st.spinner("Prédiction en cours..."):
                    st.session_state.predictions = predict_future_churn(
                        pipeline,
                        df.head(n_clients),
                        months
                    )
                    st.success(f"Prédiction terminée pour {n_clients} clients sur {months} mois")

            if st.session_state.predictions is not None:
                st.write("Résultats de prédiction:")
                st.dataframe(st.session_state.predictions[[
                    'CustomerID', 'Age', 'Tenure (Months)', 'Monthly Charges',
                    'Future_Churn_Probability', 'Predicted_Churn'
                ]].head())

                # Visualisation
                visualize_predictions(st.session_state.predictions)

                # Résumé des prédictions
                predicted_churn_count = st.session_state.predictions['Predicted_Churn'].sum()
                total_clients = len(st.session_state.predictions)
                predicted_churn_rate = predicted_churn_count / total_clients

                st.subheader("Résumé des prédictions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clients analysés", total_clients)
                with col2:
                    st.metric("Clients à risque de churn", predicted_churn_count)
                with col3:
                    st.metric("Taux de churn prédit", f"{predicted_churn_rate:.2%}")