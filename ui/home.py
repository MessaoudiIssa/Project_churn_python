import streamlit as st
import pandas as pd
import plotly.express as px
from data.data_loader import load_data


def render_home():
    """Affiche la page d'aperçu des données"""
    st.header("Aperçu des données")

    # Chargement des données
    if 'df' not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Impossible de charger les données. Vérifiez le chemin du fichier CSV.")
        return

    df = st.session_state.df

    # Affichage des informations de base
    st.write(f"Nombre total de clients: {len(df)}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Taux de Churn Global", f"{df['Churn'].mean():.2%}")
    with col2:
        st.metric("Valeur Client Moyenne", f"{df['Total Charges'].mean():.2f} TND")

    # Aperçu des données
    with st.expander("Aperçu des données"):
        st.dataframe(df.head())

    # Statistiques descriptives
    with st.expander("Statistiques descriptives"):
        st.write(df.describe())

    # Distribution des variables importantes
    st.subheader("Distribution des variables clés")
    col1, col2 = st.columns(2)

    with col1:
        # Distribution de l'ancienneté
        fig1 = px.histogram(df, x='Tenure (Months)',
                            title="Distribution de l'ancienneté",
                            color='Churn',
                            barmode='overlay',
                            color_discrete_sequence=['#3366CC', '#CC3366'])
        st.plotly_chart(fig1)

    with col2:
        # Distribution des charges mensuelles
        fig2 = px.histogram(df, x='Monthly Charges',
                            title="Distribution des charges mensuelles",
                            color='Churn',
                            barmode='overlay',
                            color_discrete_sequence=['#3366CC', '#CC3366'])
        st.plotly_chart(fig2)

    # Répartition du churn par type de contrat
    fig3 = px.bar(df.groupby('Contract Type')['Churn'].mean().reset_index(),
                  x='Contract Type',
                  y='Churn',
                  title="Taux de churn par type de contrat",
                  color='Contract Type')
    fig3.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    st.plotly_chart(fig3)

    # Satisfaction vs. Churn
    fig4 = px.box(df, x='Satisfaction Score', y='Churn',
                  title="Relation entre satisfaction et churn")
    st.plotly_chart(fig4)

    # Vérification des CustomerID
    with st.expander("Vérification des CustomerID"):
        st.write("Exemples d'ID clients:", df['CustomerID'].head(10).tolist())
        st.write("Nombre de doublons:", df['CustomerID'].duplicated().sum())