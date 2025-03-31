from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from keras import layers, models
import os
import time

app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite toda ruta
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

class ConvLayer(BaseModel):
    filters: int
    kernel_size: int
    activation: str = "relu"
    pooling: Optional[str] = None  # "max" o "average"

class DenseLayer(BaseModel):
    units: int
    activation: str

class CNNConfig(BaseModel):
    input_shape: List[int]
    conv_layers: List[ConvLayer]
    dense_layers: List[DenseLayer]
    output_units: int
    output_activation: str
    model_name: Optional[str] = "model"  # Add this field to receive the name from frontend

@app.post("/create_cnn/")
def create_cnn_endpoint(config: CNNConfig):
    try:
        model = create_cnn(config)
        model.summary() #para corroborar la arquitectura del modelo
        
        # Format the date as DD_MM_YYYY
        current_date = time.strftime('%d_%m_%Y', time.localtime())
        
        # Use the user-provided name or default to "model"
        user_name = config.model_name.replace(" ", "_")  # Replace spaces with underscores
        
        # Create filename with the format DD_MM_YYYY_userName
        model_filename = f"models/architecture_{current_date}_{user_name}.h5"
        
        # Save the model to disk
        model.save(model_filename)
        
        return {
            "message": "Modelo creado con éxito", 
            "layers": [layer.name for layer in model.layers],
            "model_path": model_filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def create_cnn(config: CNNConfig):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=tuple(config.input_shape)))
    
    for layer in config.conv_layers:
        model.add(layers.Conv2D(layer.filters, (layer.kernel_size, layer.kernel_size), activation=layer.activation))
        if layer.pooling:
            if layer.pooling == "max":
                model.add(layers.MaxPooling2D((2, 2)))
            elif layer.pooling == "average":
                model.add(layers.AveragePooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    for layer in config.dense_layers:
        model.add(layers.Dense(layer.units, activation=layer.activation))
    
    model.add(layers.Dense(config.output_units, activation=config.output_activation))
    return model

# Create a directory for saving models if it doesn't exist
os.makedirs("models", exist_ok=True)

#Para probar que este en curso 
@app.get("/")
def read_root():
    return {"message": "¡FastAPI está corriendo!"}

# Add this new endpoint to list all saved models
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
            # Get file creation time and size
            creation_time = os.path.getctime(model_path)
            size_kb = os.path.getsize(model_path) / 1024
            
            # Load model to get summary info
            try:
                model = tf.keras.models.load_model(model_path)
                # Get layer information
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
                # If there's an error loading the model, still include basic info
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

# Add a new endpoint to delete models
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
