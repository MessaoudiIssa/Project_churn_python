import streamlit as st
from ui.home import render_home
from ui.segmentation_view import render_segmentation
from ui.prediction_view import render_prediction
from ui.export_view import render_export

# Configuration de la page
st.set_page_config(page_title="Analyse Churn Tunisie Telecom", layout="wide")
st.title("Analyse Prédictive de Churn")

# Menu principal
def main():
    # Menu principal
    page = st.sidebar.selectbox("Menu", [
        "Aperçu",
        "Segmentation",
        "Prédiction",
        "Évaluation des Modèles",
        "Export Firebase"
    ])

    # Routing based on selected page
    if page == "Aperçu":
        render_home()
    elif page == "Segmentation":
        render_segmentation()
    elif page == "Prédiction":
        render_prediction()
    elif page == "Évaluation des Modèles":
        render_prediction(evaluation_only=True)
    elif page == "Export Firebase":
        render_export()

if __name__ == "__main__":
    main()