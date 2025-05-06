import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


def predict_future_churn(pipeline, df, months=3):
    """
    Prédit le churn dans X mois

    Args:
        pipeline: Pipeline entrainé
        df: DataFrame contenant les données client
        months: Nombre de mois dans le futur pour la prédiction

    Returns:
        DataFrame avec prédictions
    """
    # Copie des données pour simulation future
    future_df = df.copy()

    # Simulation de l'évolution dans X mois
    future_df['Tenure (Months)'] += months
    future_df['Total Charges'] += future_df['Monthly Charges'] * months

    # Prédiction
    X_future = future_df[[col for col in pipeline.feature_names_in_ if col in future_df.columns]]
    future_df['Future_Churn_Probability'] = pipeline.predict_proba(X_future)[:, 1]
    future_df['Predicted_Churn'] = (future_df['Future_Churn_Probability'] > 0.5).astype(int)

    return future_df


def predict_for_individual(pipeline, client_data):
    """
    Prédit le churn pour un client individuel

    Args:
        pipeline: Pipeline entrainé
        client_data: DataFrame contenant les données d'un seul client

    Returns:
        DataFrame avec prédictions
    """
    client_data['Future_Churn_Probability'] = pipeline.predict_proba(client_data)[:, 1]
    client_data['Predicted_Churn'] = (client_data['Future_Churn_Probability'] > 0.5).astype(int)
    return client_data


def visualize_predictions(predictions):
    """
    Visualise les résultats de prédiction

    Args:
        predictions: DataFrame contenant les prédictions

    Returns:
        None (affiche des graphiques via Streamlit)
    """
    # Distribution des probabilités
    fig1 = px.histogram(
        predictions,
        x='Future_Churn_Probability',
        nbins=20,
        title="Distribution des probabilités de churn",
        color_discrete_sequence=['#3366CC']
    )
    st.plotly_chart(fig1)

    # Relation entre ancienneté et churn
    fig2 = px.scatter(
        predictions,
        x='Tenure (Months)',
        y='Future_Churn_Probability',
        color='Predicted_Churn',
        title="Relation entre ancienneté et risque de churn",
        color_discrete_sequence=['#33CC66', '#CC3366']
    )
    st.plotly_chart(fig2)

    # Top clients à risque
    high_risk = predictions.sort_values('Future_Churn_Probability', ascending=False).head(10)
    st.subheader("Top 10 clients à risque élevé de churn")
    st.dataframe(high_risk[[
        'CustomerID', 'Age', 'Tenure (Months)', 'Monthly Charges',
        'Satisfaction Score', 'Future_Churn_Probability'
    ]])