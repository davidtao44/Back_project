from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.schemas.hls import HLSConvertRequest, QuantizeRequest
from app.services.auth_service import get_current_user
from app.services.hls_synthesis_service import (
    convert_model_to_hls,
    download_hls_project,
    quantize_model_for_hardware,
)

router = APIRouter(prefix="/hls", tags=["hls-synthesis"])


@router.post("/quantize/")
def quantize_for_hardware(
    request: QuantizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Paso 1 — PTQ: redondea pesos a ap_fixed<total_bits,int_bits>."""
    return quantize_model_for_hardware(request)


@router.post("/convert/")
def convert_to_hls(
    request: HLSConvertRequest,
    current_user: dict = Depends(get_current_user),
):
    """Paso 2 — Genera proyecto HLS C++ con hls4ml (no requiere Vitis HLS)."""
    return convert_model_to_hls(request, current_user)


@router.get("/download/{session_id}")
def download_hls(
    session_id: str,
    current_user: dict = Depends(get_current_user),
) -> FileResponse:
    """Descarga el proyecto HLS C++ como .zip."""
    return download_hls_project(session_id, current_user)
