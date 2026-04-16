from typing import List, Optional

from pydantic import BaseModel


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
    model_name: Optional[str] = "model"


class ModelWeightsRequest(BaseModel):
    model_path: str
    output_dir: str = "model_weights"
    bits_value: int = 8
