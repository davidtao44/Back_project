"""Servicio de datasets integrados de Keras para construir y entrenar CNN.

Los datasets se descargan de los servidores de Keras la primera vez que se usan
y quedan cacheados en ~/.keras/datasets/, por lo que después funcionan offline.
"""

import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image

# Registro de datasets soportados. La descarga solo ocurre al llamar a load_dataset.
DATASETS = {
    "mnist": {
        "loader": tf.keras.datasets.mnist,
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "color_mode": "grayscale",
        "description": "Dígitos manuscritos 28x28 en escala de grises (10 clases)",
        "class_names": [str(i) for i in range(10)],
        "approx_download_mb": 11,
    },
    "fashion_mnist": {
        "loader": tf.keras.datasets.fashion_mnist,
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "color_mode": "grayscale",
        "description": "Prendas de ropa 28x28 en escala de grises (10 clases)",
        "class_names": [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "approx_download_mb": 30,
    },
    "cifar10": {
        "loader": tf.keras.datasets.cifar10,
        "input_shape": [32, 32, 3],
        "num_classes": 10,
        "color_mode": "rgb",
        "description": "Objetos 32x32 RGB en 10 categorías",
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "approx_download_mb": 170,
    },
    "cifar100": {
        "loader": tf.keras.datasets.cifar100,
        "input_shape": [32, 32, 3],
        "num_classes": 100,
        "color_mode": "rgb",
        "description": "Objetos 32x32 RGB en 100 categorías",
        "class_names": None,
        "approx_download_mb": 169,
    },
}


def list_datasets():
    """Metadata de los datasets disponibles (no descarga nada)."""
    return {
        "datasets": [
            {
                "name": name,
                "input_shape": meta["input_shape"],
                "num_classes": meta["num_classes"],
                "color_mode": meta["color_mode"],
                "description": meta["description"],
                "approx_download_mb": meta["approx_download_mb"],
            }
            for name, meta in DATASETS.items()
        ]
    }


def _prepare_x(x: np.ndarray) -> np.ndarray:
    """Normalizar a [0, 1] float32 y garantizar la dimensión de canal."""
    x = x.astype("float32") / 255.0
    if x.ndim == 3:  # escala de grises sin canal explícito
        x = np.expand_dims(x, axis=-1)
    return x


def _prepare_y(y: np.ndarray) -> np.ndarray:
    """Aplanar etiquetas a un vector 1D de enteros de clase."""
    return np.asarray(y).reshape(-1).astype("int64")


def load_dataset(name: str, limit: int = None):
    """Cargar un dataset de Keras normalizado a [0, 1].

    Devuelve (x_train, y_train), (x_test, y_test) con x en float32 [0, 1] y
    forma (n, H, W, C); y como vector de enteros de clase.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Dataset desconocido: '{name}'. Disponibles: {sorted(DATASETS)}"
        )

    (x_train, y_train), (x_test, y_test) = DATASETS[name]["loader"].load_data()

    x_train, x_test = _prepare_x(x_train), _prepare_x(x_test)
    y_train, y_test = _prepare_y(y_train), _prepare_y(y_test)

    if limit:
        x_train, y_train = x_train[:limit], y_train[:limit]
        x_test, y_test = x_test[:limit], y_test[:limit]

    return (x_train, y_train), (x_test, y_test)


def get_dataset_preview(name: str, n: int = 8):
    """Devolver n imágenes de muestra codificadas en base64 con sus etiquetas."""
    if name not in DATASETS:
        raise ValueError(f"Dataset desconocido: '{name}'")

    (x_train, y_train), _ = load_dataset(name, limit=n)
    meta = DATASETS[name]
    samples = []

    for i in range(min(n, len(x_train))):
        arr = (x_train[i] * 255).astype("uint8")
        mode = "L" if arr.shape[-1] == 1 else "RGB"
        img = Image.fromarray(arr.squeeze() if mode == "L" else arr, mode=mode)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        samples.append(
            {"image": f"data:image/png;base64,{encoded}", "label": int(y_train[i])}
        )

    return {"name": name, "samples": samples, "class_names": meta["class_names"]}
