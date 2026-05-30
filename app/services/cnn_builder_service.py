"""Constructor de modelos CNN secuenciales de Keras a partir de un spec JSON.

El frontend define la arquitectura como una lista de capas; este servicio la
traduce a un tf.keras.Sequential, lo valida y opcionalmente lo guarda en models/.
"""

import os
import time

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import models

MODELS_DIR = "models"

# Registro de tipos de capa soportados. Para cada parámetro se declara su tipo,
# valor por defecto y, si aplica, las opciones válidas. El frontend usa esto
# para renderizar el formulario del constructor capa por capa.
LAYER_REGISTRY = {
    "Conv2D": {
        "category": "convolutional",
        "params": {
            "filters": {"type": "int", "default": 16},
            "kernel_size": {"type": "int", "default": 3},
            "strides": {"type": "int", "default": 1},
            "padding": {"type": "enum", "default": "valid", "options": ["valid", "same"]},
            "activation": {
                "type": "enum",
                "default": "relu",
                "options": ["relu", "tanh", "sigmoid", "linear", "softmax"],
            },
        },
    },
    "MaxPooling2D": {
        "category": "pooling",
        "params": {
            "pool_size": {"type": "int", "default": 2},
            "strides": {"type": "int", "default": 2},
        },
    },
    "AveragePooling2D": {
        "category": "pooling",
        "params": {
            "pool_size": {"type": "int", "default": 2},
            "strides": {"type": "int", "default": 2},
        },
    },
    "Dense": {
        "category": "dense",
        "params": {
            "units": {"type": "int", "default": 64},
            "activation": {
                "type": "enum",
                "default": "relu",
                "options": ["relu", "tanh", "sigmoid", "linear", "softmax"],
            },
        },
    },
    "Flatten": {"category": "flatten", "params": {}},
    "Dropout": {
        "category": "regularization",
        "params": {"rate": {"type": "float", "default": 0.5}},
    },
    "BatchNormalization": {"category": "regularization", "params": {}},
    "Activation": {
        "category": "activation",
        "params": {
            "activation": {
                "type": "enum",
                "default": "relu",
                "options": ["relu", "tanh", "sigmoid", "linear", "softmax"],
            }
        },
    },
}

# Plantillas de arquitectura. La capa final Dense lleva units=None: get_template
# la sustituye por el número de clases del dataset elegido.
TEMPLATES = {
    "lenet5": {
        "description": "LeNet-5 clásica (2 conv + 3 densas)",
        "layers": [
            {"type": "Conv2D", "params": {"filters": 6, "kernel_size": 5, "activation": "relu"}},
            {"type": "AveragePooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Conv2D", "params": {"filters": 16, "kernel_size": 5, "activation": "relu"}},
            {"type": "AveragePooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 120, "activation": "relu"}},
            {"type": "Dense", "params": {"units": 84, "activation": "relu"}},
            {"type": "Dense", "params": {"units": None, "activation": "softmax"}},
        ],
    },
    "simple_cnn": {
        "description": "CNN simple (2 bloques conv + densa con dropout)",
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 128, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Dense", "params": {"units": None, "activation": "softmax"}},
        ],
    },
    "vgg_mini": {
        "description": "VGG reducida (bloques conv con BatchNorm)",
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"type": "BatchNormalization", "params": {}},
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same", "activation": "relu"}},
            {"type": "BatchNormalization", "params": {}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2, "strides": 2}},
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 256, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Dense", "params": {"units": None, "activation": "softmax"}},
        ],
    },
}


def list_layer_types():
    """Tipos de capa soportados con sus parámetros (para el constructor en la UI)."""
    return {
        "layer_types": [
            {"type": name, "category": meta["category"], "params": meta["params"]}
            for name, meta in LAYER_REGISTRY.items()
        ]
    }


def list_templates():
    """Plantillas de arquitectura disponibles."""
    return {
        "templates": [
            {"name": name, "description": tpl["description"], "layers": tpl["layers"]}
            for name, tpl in TEMPLATES.items()
        ]
    }


def get_template(name: str, input_shape, num_classes: int):
    """Devolver una plantilla como spec completo, lista para construir/entrenar.

    Sustituye la capa final (units=None) por el número de clases del dataset.
    """
    if name not in TEMPLATES:
        raise ValueError(f"Plantilla desconocida: '{name}'. Disponibles: {sorted(TEMPLATES)}")

    import copy

    layers_spec = copy.deepcopy(TEMPLATES[name]["layers"])
    for layer in layers_spec:
        if layer["type"] == "Dense" and layer["params"].get("units") is None:
            layer["params"]["units"] = num_classes

    return {
        "name": f"{name}_model",
        "input_shape": list(input_shape),
        "layers": layers_spec,
    }


def _resolve_params(layer_type: str, provided: dict) -> dict:
    """Combinar los parámetros indicados con los valores por defecto del registro."""
    registry = LAYER_REGISTRY[layer_type]["params"]
    resolved = {}

    for pname, pdef in registry.items():
        value = provided.get(pname, pdef["default"])
        if value is None:
            raise ValueError(
                f"El parámetro '{pname}' de la capa {layer_type} no puede ser nulo"
            )
        if pdef["type"] == "int":
            value = int(value)
        elif pdef["type"] == "float":
            value = float(value)
        elif pdef["type"] == "enum" and value not in pdef["options"]:
            raise ValueError(
                f"Valor inválido '{value}' para '{pname}' en {layer_type}. "
                f"Opciones: {pdef['options']}"
            )
        resolved[pname] = value

    return resolved


def build_model(spec: dict) -> tf.keras.Model:
    """Construir un tf.keras.Sequential a partir de un spec de arquitectura.

    spec = {name, input_shape: [H, W, C], layers: [{type, params}, ...]}
    """
    name = spec.get("name", "cnn_model")
    input_shape = spec.get("input_shape")
    layer_specs = spec.get("layers", [])

    if not input_shape or len(input_shape) != 3:
        raise ValueError("input_shape debe ser una lista [alto, ancho, canales]")
    if not layer_specs:
        raise ValueError("El modelo debe tener al menos una capa")

    model = models.Sequential(name=name)
    model.add(L.Input(shape=tuple(input_shape)))

    for i, layer_spec in enumerate(layer_specs):
        layer_type = layer_spec.get("type")
        if layer_type not in LAYER_REGISTRY:
            raise ValueError(
                f"Capa {i}: tipo no soportado '{layer_type}'. "
                f"Soportados: {sorted(LAYER_REGISTRY)}"
            )

        params = _resolve_params(layer_type, layer_spec.get("params", {}))
        builder = getattr(L, layer_type)
        try:
            model.add(builder(**params))
        except Exception as e:
            raise ValueError(f"Capa {i} ({layer_type}): error al construir — {e}")

    # Forzar la construcción para detectar errores de forma de inmediato
    try:
        model.build((None, *input_shape))
    except Exception as e:
        raise ValueError(f"Arquitectura inválida (incompatibilidad de formas): {e}")

    return model


def save_built_model(model: tf.keras.Model, name: str) -> str:
    """Guardar el modelo construido en models/ y devolver su ruta."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")) or "cnn_model"
    filename = f"{int(time.time())}_{safe_name}.keras"
    path = os.path.join(MODELS_DIR, filename)
    model.save(path)
    return path
