import firebase_admin
from firebase_admin import credentials, firestore, auth
import os
import json

# Variable global para la instancia de Firestore
db = None

def initialize_firestore():
    """
    Inicializa la conexión con Firestore usando las credenciales del archivo JSON
    """
    global db
    
    if db is not None:
        return db
    
    try:
        # Ruta directa al archivo de credenciales
        credential_path = 'firebase-credentials.json'
        
        if not os.path.exists(credential_path):
            raise FileNotFoundError(
                "No se encontró el archivo firebase-credentials.json. "
                "Asegúrate de que el archivo esté en el directorio del proyecto."
            )
        
        # Inicializar Firebase Admin SDK
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred)
        
        # Obtener cliente de Firestore
        db = firestore.client()
        print(f"✅ Firestore y Firebase Auth inicializados correctamente usando {credential_path}")
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