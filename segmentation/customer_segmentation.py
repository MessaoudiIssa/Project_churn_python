import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st


def perform_segmentation(df, features, n_clusters=3):
    """
    Effectue la segmentation client

    Args:
        df: DataFrame contenant les données client
        features: Liste des caractéristiques à utiliser pour la segmentation
        n_clusters: Nombre de segments à créer

    Returns:
        DataFrame avec la colonne 'Segment' ajoutée
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(X)
    return df


def plot_segments(df, features):
    """
    Visualise les segments de clients

    Args:
        df: DataFrame contenant les données segmentées
        features: Caractéristiques utilisées pour la visualisation

    Returns:
        None (affiche des graphiques via Streamlit)
    """
    if len(features) < 2:
        st.warning("Au moins 2 caractéristiques sont nécessaires pour la visualisation")
        return

    # Distribution des segments
    fig1 = px.pie(
        df,
        names='Segment',
        title="Répartition des segments",
        color='Segment',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig1)

    # Visualisation 2D des segments
    fig2 = px.scatter(
        df,
        x=features[0],
        y=features[1],
        color='Segment',
        title=f"Segmentation des clients ({features[0]} vs {features[1]})",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig2)

    # Analyse par segment
    st.subheader("Caractéristiques par segment")
    segment_stats = df.groupby('Segment')[features + ['Churn']].mean().reset_index()

    fig3 = px.bar(
        segment_stats.melt(id_vars='Segment', value_vars=features),
        x='variable',
        y='value',
        color='Segment',
        barmode='group',
        title="Comparaison des caractéristiques par segment",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig3)

    # Taux de churn par segment
    fig4 = px.bar(
        segment_stats,
        x='Segment',
        y='Churn',
        title="Taux de churn par segment",
        color='Segment',
        text_auto='.0%',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig4.update_traces(texttemplate='%{y:.1%}')
    st.plotly_chart(fig4)