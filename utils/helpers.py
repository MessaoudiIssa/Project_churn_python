import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st


def display_model_evaluation(metrics_dict):
    """
    Affiche l'évaluation des modèles

    Args:
        metrics_dict: Dictionnaire contenant les métriques d'évaluation

    Returns:
        None (affiche des graphiques via Streamlit)
    """
    st.header("Évaluation Comparative des Modèles")

    # Création d'un dataframe pour l'affichage
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')

    # Affichage des métriques principales
    st.subheader("Métriques de Performance")
    st.dataframe(metrics_df[['Accuracy', 'Precision', 'Recall', 'F1', 'AUCROC']].style.format("{:.2%}"))

    # Graphique comparatif
    fig = px.bar(metrics_df.reset_index(),
                 x='index',
                 y=['Accuracy', 'Precision', 'Recall', 'F1'],
                 barmode='group',
                 title="Comparaison des Modèles")
    st.plotly_chart(fig)

    # Détails par modèle
    selected_model = st.selectbox("Voir les détails pour le modèle:", list(metrics_dict.keys()))

    st.subheader(f"Détails pour le modèle {selected_model}")
    model_metrics = metrics_dict[selected_model]

    # Matrice de confusion
    st.write("**Matrice de confusion:**")
    fig = ff.create_annotated_heatmap(
        z=model_metrics['Confusion_Matrix'],
        x=['Non-Churn', 'Churn'],
        y=['Non-Churn', 'Churn'],
        colorscale='Blues'
    )
    st.plotly_chart(fig)

    # Rapport de classification
    st.write("**Rapport de classification:**")
    report_df = pd.DataFrame(model_metrics['Classification_Report']).transpose()
    st.dataframe(report_df.style.format("{:.2%}"))

    # Features utilisées
    st.write("**Variables utilisées:**")
    st.write(", ".join(model_metrics['Features']))