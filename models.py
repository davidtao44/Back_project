from pydantic import BaseModel
from typing import List, Optional

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
    model_name: Optional[str] = "model"  # Campo para recibir el nombre desde el frontend

# Nuevo modelo para la conversión de imágenes a VHDL
class ImageToVHDLRequest(BaseModel):
    image_data: str  # Datos de imagen en base64
    output_format: str = "hex"  # Formato de salida: "hex" o "decimal"
    width: int = 32  # Ancho de la imagen redimensionada
    height: int = 32  # Alto de la imagen redimensionada

# Modelo para las solicitudes de FaultInjector
class FaultInjectorRequest(BaseModel):
    model_path: str  # Ruta del modelo a usar
    image_data: Optional[str] = None  # Datos de imagen en base64 (opcional)
    fault_type: Optional[str] = None  # Tipo de fallo a inyectar (opcional)
    fault_parameters: Optional[dict] = None  # Parámetros específicos del fallo (opcional)