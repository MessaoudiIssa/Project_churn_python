import os
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st


def init_firebase():
    """
    Initialise la connexion à Firebase si ce n'est pas déjà fait.
    Retourne un client Firestore ou None en cas d'erreur.
    """
    try:
        if not firebase_admin._apps:
            # Chemin absolu vers le dossier credentials
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            credentials_path = os.path.join(base_path, "credentials", "firebase_credentials.json")

            if not os.path.exists(credentials_path):
                st.error(f"""
                Le fichier firebase_credentials.json est introuvable.
                Chemin recherché: {credentials_path}

                Veuillez vérifier que:
                1. Le dossier 'credentials' existe à la racine du projet
                2. Le fichier 'firebase_credentials.json' est présent dans ce dossier
                """)
                return None

            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"""
        Erreur d'initialisation Firebase: {str(e)}

        Vérifiez que:
        1. Le fichier de configuration est valide
        2. Les permissions sont correctement configurées
        3. Le service Firebase est actif
        """)
        return None