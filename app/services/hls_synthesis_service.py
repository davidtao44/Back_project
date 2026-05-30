import os

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.core.config import HLS_OUTPUTS_DIR
from app.schemas.hls import HLSConvertRequest, QuantizeRequest
from hls_synthesis.fixed_point_quantizer import apply_fixed_point_ptq
from hls_synthesis.hls_converter import convert_to_hls


def quantize_model_for_hardware(request: QuantizeRequest) -> dict:
    stats = apply_fixed_point_ptq(
        model_name=request.model_name,
        total_bits=request.total_bits,
        int_bits=request.int_bits,
    )
    return {"success": True, **stats}


def convert_model_to_hls(request: HLSConvertRequest, current_user: dict) -> dict:
    user_id = current_user.get("uid", "unknown")
    config = {
        "backend": request.backend,
        "precision": request.precision,
        "reuse_factor": request.reuse_factor,
        "clock_period": request.clock_period,
        "io_type": request.io_type,
        "strategy": request.strategy,
        "board": request.board,
        "part": request.part,
    }
    result = convert_to_hls(model_name=request.model_name, user_id=user_id, config=config)
    return {"success": True, **result}


def download_hls_project(session_id: str, current_user: dict) -> FileResponse:
    zip_path = os.path.join(HLS_OUTPUTS_DIR, session_id, "hls_project.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Proyecto HLS no encontrado: {session_id}")
    return FileResponse(path=zip_path, filename=f"hls_project_{session_id}.zip", media_type="application/zip")
