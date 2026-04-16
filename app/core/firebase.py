import json

import firebase_admin
from firebase_admin import credentials, firestore

from app.core.config import FIREBASE_API_KEY

db = None


def initialize_firestore():
    """Inicializa la conexión con Firestore usando las credenciales del archivo JSON."""
    global db

    if db is not None:
        return db

    try:
        if not FIREBASE_API_KEY:
            raise ValueError(
                "La variable de entorno FIREBASE_API_KEY no está configurada. "
                "Asegúrate de configurar el JSON completo de las credenciales de Firebase."
            )

        firebase_config = json.loads(FIREBASE_API_KEY) if isinstance(FIREBASE_API_KEY, str) else FIREBASE_API_KEY

        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("✅ Firestore y Firebase Auth inicializados correctamente usando variables de entorno")
        return db

    except Exception as e:
        print(f"❌ Error al inicializar Firestore: {str(e)}")
        raise e


def get_firestore_client():
    """Obtiene el cliente de Firestore, inicializándolo si es necesario."""
    global db
    if db is None:
        db = initialize_firestore()
    return db


def test_connection():
    """Prueba la conexión con Firestore."""
    try:
        client = get_firestore_client()
        client.collections()
        print("✅ Conexión con Firestore exitosa")
        return True
    except Exception as e:
        print(f"❌ Error en la conexión con Firestore: {str(e)}")
        return False
