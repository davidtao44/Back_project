from pydantic import BaseModel


class ImageToVHDLRequest(BaseModel):
    image_data: str  # Datos de imagen en base64
    output_format: str = "hex"  # "hex" o "decimal"
    width: int = 32
    height: int = 32
