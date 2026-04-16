import os
import shutil
import time
from typing import List

import tensorflow as tf
from fastapi import HTTPException, UploadFile


def list_models():
    """Listar modelos .h5 disponibles con información detallada de capas."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"models": []}

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
    models_info = []

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        creation_time = os.path.getctime(model_path)
        size_kb = os.path.getsize(model_path) / 1024

        try:
            model = tf.keras.models.load_model(model_path)
            layers_info = []
            for layer in model.layers:
                layer_info = {"name": layer.name, "type": layer.__class__.__name__}
                if hasattr(layer, "units"):
                    layer_info["units"] = layer.units
                if hasattr(layer, "filters"):
                    layer_info["filters"] = layer.filters
                layers_info.append(layer_info)

            models_info.append(
                {
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "layers": layers_info,
                }
            )
        except Exception as e:
            models_info.append(
                {
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "error": str(e),
                }
            )

    return {"models": models_info}


async def upload_model(file: UploadFile):
    """Subir y validar un modelo CNN preentrenado (.h5 / .keras)."""
    if not file.filename.endswith((".h5", ".keras")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .h5 o .keras")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(models_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        model = tf.keras.models.load_model(file_path)
        model_info = {
            "name": filename,
            "path": file_path,
            "layers": len(model.layers),
            "parameters": model.count_params(),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"El archivo no es un modelo válido: {str(e)}")

    return {"message": "Modelo subido exitosamente", "model_info": model_info}


def delete_models(model_paths: List[str]):
    """Eliminar varios modelos por ruta."""
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
        "errors": errors,
    }


def quantize_model(model_path: str, multiplication_factor: int = 100):
    """Cuantiza el modelo multiplicando pesos por un factor entero."""
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: {os.path.basename(model_path)}",
        )

    base_name = os.path.basename(model_path)
    name_parts = os.path.splitext(base_name)
    quantized_filename = f"{name_parts[0]}_quantized_{multiplication_factor}{name_parts[1]}"
    save_path = os.path.join("models", quantized_filename)

    result_path = modify_and_save_weights(model_path, save_path, multiplication_factor)  # noqa: F821

    return {
        "success": True,
        "message": f"Modelo cuantizado con factor {multiplication_factor} exitosamente",
        "original_model": os.path.basename(model_path),
        "quantized_model": os.path.basename(result_path),
        "quantized_model_path": result_path,
    }


def get_available_models_for_campaign():
    """Listar modelos disponibles para campañas de fallos."""
    models_dir = "models"
    available_models = []

    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith((".h5", ".keras")):
                model_path = os.path.join(models_dir, filename)
                available_models.append(
                    {
                        "name": filename,
                        "path": model_path,
                        "size": os.path.getsize(model_path),
                        "modified": os.path.getmtime(model_path),
                    }
                )

    return {"success": True, "models": available_models, "count": len(available_models)}
