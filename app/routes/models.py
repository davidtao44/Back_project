from typing import List

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from app.services.auth_service import get_current_user
from app.services.model_service import (
    delete_models as delete_models_service,
)
from app.services.model_service import (
    list_models as list_models_service,
)
from app.services.model_service import (
    quantize_model as quantize_model_service,
)
from app.services.model_service import (
    upload_model as upload_model_service,
)

router = APIRouter(tags=["models"])


@router.get("/models/")
def get_models(current_user: dict = Depends(get_current_user)):
    """Lista los modelos disponibles (requiere autenticación)."""
    try:
        return list_models_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_models/")
def list_models(current_user: dict = Depends(get_current_user)):
    """Endpoint legacy para compatibilidad."""
    try:
        return list_models_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_model/")
async def upload_model(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    """Sube un modelo CNN preentrenado."""
    try:
        return await upload_model_service(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir el modelo: {str(e)}")


@router.post("/delete_models/")
def delete_models(
    model_paths: List[str] = Body(...), current_user: dict = Depends(get_current_user)
):
    try:
        return delete_models_service(model_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantize_model/")
def quantize_model(
    model_path: str = Body(...),
    multiplication_factor: int = Body(100),
    current_user: dict = Depends(get_current_user),
):
    try:
        return quantize_model_service(model_path, multiplication_factor)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
