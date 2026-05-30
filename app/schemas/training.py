"""Esquemas Pydantic para el módulo de construcción y entrenamiento de CNN."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LayerSpec(BaseModel):
    """Una capa dentro de un spec de arquitectura."""

    type: str
    params: Dict[str, Any] = {}


class ModelBuildRequest(BaseModel):
    """Spec completo para construir un modelo CNN secuencial."""

    name: str = "cnn_model"
    input_shape: List[int]
    layers: List[LayerSpec]


class TrainingRequest(BaseModel):
    """Petición de entrenamiento.

    Se entrena o bien un modelo nuevo (model_spec) o bien uno ya guardado
    (model_path). El dataset es uno de los integrados de Keras.
    """

    model_spec: Optional[ModelBuildRequest] = None
    model_path: Optional[str] = None
    dataset: str
    epochs: int = 5
    batch_size: int = 64
    optimizer: str = "adam"
    learning_rate: float = 0.001
    validation_split: float = 0.1
    train_limit: Optional[int] = None  # limitar muestras de entrenamiento (pruebas)
