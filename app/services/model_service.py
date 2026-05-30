import os
import shutil
import time
from typing import List

import tensorflow as tf
from fastapi import HTTPException, UploadFile


def _safe_shape(shape):
    """Convertir una forma de Keras (posibles None / tuplas anidadas) en algo JSON-seguro."""
    if shape is None:
        return None
    # Modelos con múltiples entradas/salidas devuelven listas de tuplas
    if isinstance(shape, list):
        return [_safe_shape(s) for s in shape]
    return [int(d) if d is not None else None for d in shape]


def _layer_output_shape(layer):
    """Obtener la forma de salida de una capa de forma robusta (Keras 2 y 3)."""
    shape = getattr(layer, "output_shape", None)
    if shape is not None:
        return shape
    try:
        out = layer.output
        if isinstance(out, list):
            return [tuple(o.shape) for o in out]
        return tuple(out.shape)
    except Exception:
        return None


def _categorize(layer):
    """Clasificar una capa de Keras en una categoría funcional para el frontend."""
    cls = layer.__class__.__name__
    if "Conv" in cls:
        return "convolutional"
    if "Pooling" in cls:
        return "pooling"
    if cls == "Dense":
        return "dense"
    if cls == "Flatten" or cls.startswith("Global"):
        return "flatten"
    return "other"


def inspect_model_layers(model_path: str):
    """Inspeccionar capa por capa un modelo Keras (fuente única de verdad: layer.name)."""
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: {os.path.basename(model_path)}",
        )

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"El archivo no es un modelo válido: {str(e)}")

    layers = []
    for idx, layer in enumerate(model.layers):
        weights = layer.get_weights()
        info = {
            "index": idx,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "category": _categorize(layer),
            "output_shape": _safe_shape(_layer_output_shape(layer)),
            "has_weights": bool(weights),
        }
        if weights:
            info["kernel_shape"] = [int(d) for d in weights[0].shape]
            info["bias_shape"] = [int(d) for d in weights[1].shape] if len(weights) > 1 else None
        layers.append(info)

    output_shape = model.output_shape
    num_classes = None
    if output_shape is not None and not isinstance(output_shape, list):
        num_classes = output_shape[-1]

    return {
        "input_shape": _safe_shape(model.input_shape),
        "output_shape": _safe_shape(output_shape),
        "num_classes": int(num_classes) if num_classes is not None else None,
        "layers": layers,
    }


def list_models():
    """Listar modelos disponibles (.h5 / .keras) con información de capas."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"models": []}

    model_files = [
        f for f in os.listdir(models_dir) if f.endswith((".h5", ".keras"))
    ]
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

            output_shape = model.output_shape
            num_classes = (
                output_shape[-1]
                if output_shape is not None and not isinstance(output_shape, list)
                else None
            )

            models_info.append(
                {
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "layers": layers_info,
                    "input_shape": _safe_shape(model.input_shape),
                    "num_classes": int(num_classes) if num_classes is not None else None,
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
        output_shape = model.output_shape
        num_classes = (
            output_shape[-1]
            if output_shape is not None and not isinstance(output_shape, list)
            else None
        )
        model_info = {
            "name": filename,
            "path": file_path,
            "layers": len(model.layers),
            "parameters": model.count_params(),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "num_classes": int(num_classes) if num_classes is not None else None,
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
