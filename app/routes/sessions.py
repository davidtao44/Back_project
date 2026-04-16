from fastapi import APIRouter, Depends, HTTPException

from app.services.auth_service import get_current_user
from app.services.session_service import (
    build_download_response,
    cleanup_old_sessions as cleanup_old_sessions_service,
    list_active_sessions as list_active_sessions_service,
)

router = APIRouter(tags=["sessions"])


@router.delete("/cleanup_sessions/")
def cleanup_sessions(
    max_age_hours: int = 24, current_user: dict = Depends(get_current_user)
):
    """Limpia las carpetas de sesión más antiguas que max_age_hours."""
    try:
        return cleanup_old_sessions_service(max_age_hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar sesiones: {str(e)}")


@router.get("/list_sessions/")
def list_sessions(current_user: dict = Depends(get_current_user)):
    """Lista todas las sesiones activas con información básica."""
    try:
        return list_active_sessions_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar sesiones: {str(e)}")


@router.get("/download_file/")
def download_file(file_path: str, t: str = None):
    """Descarga un archivo dentro de las carpetas de sesión (o ruta absoluta)."""
    try:
        return build_download_response(file_path, session_id=t)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
