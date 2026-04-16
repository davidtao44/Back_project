from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from app.schemas.fault import FaultInjectionConfig
from app.services.auth_service import get_current_user
from app.services.fault_injection_service import (
    configure_fault_injection as configure_fault_injection_service,
)
from app.services.fault_injection_service import run_fault_injector_inference

router = APIRouter(tags=["fault-injection"])


@router.post("/fault_injector/configure/")
async def configure_fault_injection(
    fault_config: FaultInjectionConfig,
    current_user: dict = Depends(get_current_user),
):
    try:
        return configure_fault_injection_service(fault_config)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al configurar inyección de fallos: {str(e)}"
        )


@router.post("/fault_injector/inference/")
async def fault_injector_inference(
    file: UploadFile = File(...),
    model_path: str = Body(...),
    fault_config: str = Body(None),
    current_user: dict = Depends(get_current_user),
):
    try:
        return await run_fault_injector_inference(file, model_path, fault_config, current_user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(e)}")
