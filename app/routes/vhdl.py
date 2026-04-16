from fastapi import APIRouter, Depends, Form, HTTPException

from app.schemas.cnn import ModelWeightsRequest
from app.schemas.vhdl import ImageToVHDLRequest
from app.services.auth_service import get_current_user
from app.services.vhdl_service import (
    convert_image_to_vhdl as convert_image_to_vhdl_service,
)
from app.services.vhdl_service import (
    extract_model_weights as extract_model_weights_service,
)
from app.services.vhdl_service import (
    get_supported_faults as get_supported_faults_service,
)
from app.services.vhdl_service import (
    get_vhdl_file_status as get_vhdl_file_status_service,
)
from app.services.vhdl_service import (
    inject_vhdl_faults as inject_vhdl_faults_service,
)
from app.services.vhdl_service import (
    run_golden_simulation as run_golden_simulation_service,
)
from app.services.vhdl_service import (
    validate_vivado as validate_vivado_service,
)

router = APIRouter(tags=["vhdl"])


@router.post("/convert_image_to_vhdl/")
async def convert_image_to_vhdl(
    request: ImageToVHDLRequest, current_user: dict = Depends(get_current_user)
):
    return convert_image_to_vhdl_service(request, current_user)


@router.post("/extract_model_weights/")
def extract_model_weights(
    request: ModelWeightsRequest, current_user: dict = Depends(get_current_user)
):
    return extract_model_weights_service(request, current_user)


@router.get("/vhdl/supported_faults/")
def get_supported_faults():
    try:
        return get_supported_faults_service()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener fallos soportados: {str(e)}"
        )


@router.get("/vhdl/validate_vivado/")
def validate_vivado(vivado_path: str = None):
    return validate_vivado_service(vivado_path)


@router.get("/vhdl/file_status/")
async def get_vhdl_file_status(
    file_path: str, current_user: dict = Depends(get_current_user)
):
    try:
        return get_vhdl_file_status_service(file_path)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR obteniendo estado del archivo: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo estado del archivo: {str(e)}"
        )


@router.post("/vhdl/inject_faults/")
async def inject_vhdl_faults(
    filter_faults: str = Form(...),
    bias_faults: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    return inject_vhdl_faults_service(filter_faults, bias_faults, current_user)


@router.post("/vhdl/golden_simulation/")
async def golden_simulation(current_user: dict = Depends(get_current_user)):
    return run_golden_simulation_service(current_user)
