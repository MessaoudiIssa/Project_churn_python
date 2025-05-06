import streamlit as st
from data.data_loader import load_data
from preprocessing.data_cleaning import prepare_data
from segmentation.customer_segmentation import perform_segmentation, plot_segments


def render_segmentation():
    """Affiche la page de segmentation des clients"""
    st.header("Segmentation des clients")

    # Chargement des données
    if 'df' not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Impossible de charger les données. Vérifiez le chemin du fichier CSV.")
        return

    df = st.session_state.df

    # Préparation des données
    _, _, _, numeric_features, _ = prepare_data(df)

    # Interface utilisateur
    st.write("""
    La segmentation client vous permet de regrouper vos clients en segments homogènes
    pour mieux comprendre leur comportement et adapter vos stratégies.
    """)

    # Options de segmentation
    st.subheader("Configuration de la segmentation")

    selected_features = st.multiselect(
        "Variables pour segmentation",
        numeric_features,
        default=['Tenure (Months)', 'Monthly Charges', 'Total Charges']
    )

    if len(selected_features) < 2:
        st.warning("Sélectionnez au moins deux variables pour la segmentation")
        return

    n_clusters = st.slider("Nombre de segments", 2, 5, 3)

    # Exécution de la segmentation
    if st.button("Exécuter la segmentation"):
        with st.spinner("Segmentation en cours..."):
            segmented_df = perform_segmentation(df, selected_features, n_clusters)
            st.session_state.df = segmented_df
            st.session_state.segment_done = True
            st.success(f"Segmentation terminée! {n_clusters} segments créés.")

    # Affichage des résultats si la segmentation a été effectuée
    if 'segment_done' in st.session_state and st.session_state.segment_done:
        if 'Segment' in st.session_state.df.columns:
            st.subheader("Résultats de la segmentation")
            plot_segments(st.session_state.df, selected_features)

            # Tableau de distribution par segment
            st.subheader("Distribution détaillée par segment")
            segment_profile = st.session_state.df.groupby('Segment')[selected_features + ['Churn']].mean().reset_index()
            st.dataframe(segment_profile.style.format({col: "{:.2f}" for col in selected_features}))

            # Recommandations
            st.subheader("Recommandations par segment")

            for segment in range(n_clusters):
                segment_data = st.session_state.df[st.session_state.df['Segment'] == segment]
                churn_rate = segment_data['Churn'].mean()

                with st.expander(f"Segment {segment} - Taux de churn: {churn_rate:.2%}"):
                    if churn_rate > 0.4:
                        st.warning("Segment à risque élevé de churn!")
                        st.write("""
                        Recommandations:
                        - Mettre en place des offres de fidélisation prioritaires
                        - Contacter proactivement pour résoudre les problèmes
                        - Proposer des incitations pour prolonger le contrat
                        """)
                    elif churn_rate > 0.2:
                        st.info("Segment à risque modéré de churn")
                        st.write("""
                        Recommandations:
                        - Améliorer l'expérience client
                        - Offrir des services à valeur ajoutée
                        - Surveiller les indicateurs de satisfaction
                        """)
                    else:
                        st.success("Segment fidèle")
                        st.write("""
                        Recommandations:
                        - Programmes de référence et parrainage
                        - Ventes croisées de services supplémentaires
                        - Solliciter des avis et témoignages
                        """)
        else:
            st.warning("Veuillez exécuter la segmentation d'abord")