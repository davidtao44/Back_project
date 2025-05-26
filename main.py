from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
import tensorflow as tf
import os
import time

# Importar desde nuestros módulos
from models import CNNConfig
from utils import create_cnn, generate_model_filename

app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite toda ruta
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

@app.post("/create_cnn/")
def create_cnn_endpoint(config: CNNConfig):
    try:
        model = create_cnn(config)
        model.summary() # para corroborar la arquitectura del modelo
        
        # Generar nombre de archivo para el modelo
        model_filename = generate_model_filename(config.model_name)
        
        # Guardar el modelo en disco
        model.save(model_filename)
        
        return {
            "message": "Modelo creado con éxito", 
            "layers": [layer.name for layer in model.layers],
            "model_path": model_filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "¡FastAPI está corriendo!"}

@app.get("/list_models/")
def list_models():
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return {"models": []}
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        models_info = []
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            # Obtener tiempo de creación y tamaño
            creation_time = os.path.getctime(model_path)
            size_kb = os.path.getsize(model_path) / 1024
            
            # Cargar modelo para obtener información de resumen
            try:
                model = tf.keras.models.load_model(model_path)
                # Obtener información de capas
                layers_info = []
                for layer in model.layers:
                    layer_info = {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                    }
                    if hasattr(layer, 'units'):
                        layer_info["units"] = layer.units
                    if hasattr(layer, 'filters'):
                        layer_info["filters"] = layer.filters
                    layers_info.append(layer_info)
                
                models_info.append({
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "layers": layers_info
                })
            except Exception as e:
                # Si hay un error al cargar el modelo, incluir información básica
                models_info.append({
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "error": str(e)
                })
        
        return {"models": models_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_models/")
def delete_models(model_paths: List[str] = Body(...)):
    try:
        deleted_models = []
        errors = []
        
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    deleted_models.append(os.path.basename(model_path))
                else:
                    errors.append(f"Modelo no encontrado: {os.path.basename(model_path)}")
            except Exception as e:
                errors.append(f"Error al eliminar {os.path.basename(model_path)}: {str(e)}")
        
        return {
            "success": len(deleted_models) > 0,
            "deleted_models": deleted_models,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configuración del servidor
    host = "0.0.0.0"  # Esto permite conexiones desde cualquier IP
    port = 8000       # Puerto estándar para FastAPI
    
    print(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)