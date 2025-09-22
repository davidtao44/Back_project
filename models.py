from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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

# Modelos para configuración de inyección de fallos
class LayerFaultConfig(BaseModel):
    enabled: bool = False
    num_faults: int = 1
    fault_type: str = "random"  # "random" o "specific"
    positions: Optional[List[List[int]]] = None  # Posiciones específicas para fault_type="specific"
    bit_positions: Optional[List[int]] = None  # Posiciones de bits específicas

class FaultInjectionConfig(BaseModel):
    enabled: bool = False
    layers: Dict[str, LayerFaultConfig] = {}

# Modelo para las solicitudes de FaultInjector con configuración de fallos
class FaultInjectorRequest(BaseModel):
    model_path: str  # Ruta del modelo a usar
    image_data: Optional[str] = None  # Datos de imagen en base64 (opcional)
    fault_config: Optional[FaultInjectionConfig] = None  # Configuración de inyección de fallos

# Modelo para solicitudes de inferencia con inyección de fallos
class FaultInjectorInferenceRequest(BaseModel):
    model_path: str
    fault_config: Optional[FaultInjectionConfig] = None