import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
from dotenv import load_dotenv
import json

# Variable global para la instancia de Firestore
db = None
load_dotenv() # Carga las variables de entorno

def initialize_firestore():
    """
    Inicializa la conexión con Firestore usando las credenciales del archivo JSON
    """
    global db
    
    if db is not None:
        return db
    
    try:
        # Obtener credenciales JSON desde variable de entorno
        firebase_credentials_json = os.getenv("FIREBASE_API_KEY")
        
        if not firebase_credentials_json:
            raise ValueError(
                "La variable de entorno FIREBASE_API_KEY no está configurada. "
                "Asegúrate de configurar el JSON completo de las credenciales de Firebase."
            )
        
        # Parsear el JSON de credenciales si es necesario
        if isinstance(firebase_credentials_json, str):
            firebase_config = json.loads(firebase_credentials_json)
        else:
            firebase_config = firebase_credentials_json
        
        # Inicializar Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
        
        # Obtener cliente de Firestore
        db = firestore.client()
        print("✅ Firestore y Firebase Auth inicializados correctamente usando variables de entorno")
        return db
        
    except Exception as e:
        print(f"❌ Error al inicializar Firestore: {str(e)}")
        raise e

def get_firestore_client():
    """
    Obtiene el cliente de Firestore, inicializándolo si es necesario
    """
    global db
    if db is None:
        db = initialize_firestore()
    return db

def test_connection():
    """
    Prueba la conexión con Firestore
    """
    try:
        db = get_firestore_client()
        # Intentar hacer una consulta simple
        collections = db.collections()
        print("✅ Conexión con Firestore exitosa")
        return True
    except Exception as e:
        print(f"❌ Error en la conexión con Firestore: {str(e)}")
        return False

if __name__ == "__main__":
    # Prueba la conexión cuando se ejecuta directamente
    test_connection()