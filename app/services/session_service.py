import os
import shutil
import time

from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.core.config import LAYER_OUTPUTS_DIR, MODEL_WEIGHTS_OUTPUTS_DIR, VHDL_OUTPUTS_DIR


SESSION_DIRS = [LAYER_OUTPUTS_DIR, VHDL_OUTPUTS_DIR, MODEL_WEIGHTS_OUTPUTS_DIR]


def cleanup_old_sessions(max_age_hours: int = 24):
    """Limpia las carpetas de sesión más antiguas que max_age_hours."""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_sessions = []
    total_cleaned = 0

    for base_dir in SESSION_DIRS:
        if os.path.exists(base_dir):
            for session_folder in os.listdir(base_dir):
                session_path = os.path.join(base_dir, session_folder)
                if os.path.isdir(session_path):
                    folder_age = current_time - os.path.getctime(session_path)
                    if folder_age > max_age_seconds:
                        shutil.rmtree(session_path)
                        cleaned_sessions.append(f"{os.path.basename(base_dir)}/{session_folder}")
                        total_cleaned += 1

    if total_cleaned == 0:
        return {"message": "No hay carpetas de sesión antiguas para limpiar"}

    return {
        "message": f"Se limpiaron {total_cleaned} sesiones antiguas",
        "cleaned_sessions": cleaned_sessions,
    }


def list_active_sessions():
    """Lista todas las sesiones activas con información básica."""
    labeled_dirs = [
        ("layer_outputs", LAYER_OUTPUTS_DIR),
        ("vhdl_outputs", VHDL_OUTPUTS_DIR),
        ("model_weights_outputs", MODEL_WEIGHTS_OUTPUTS_DIR),
    ]

    sessions = []
    current_time = time.time()

    for dir_type, base_dir in labeled_dirs:
        if os.path.exists(base_dir):
            for session_folder in os.listdir(base_dir):
                session_path = os.path.join(base_dir, session_folder)
                if os.path.isdir(session_path):
                    creation_time = os.path.getctime(session_path)
                    age_hours = (current_time - creation_time) / 3600

                    file_count = 0
                    for _, _, files in os.walk(session_path):
                        file_count += len(files)

                    sessions.append(
                        {
                            "session_id": session_folder,
                            "session_type": dir_type,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time)),
                            "age_hours": round(age_hours, 2),
                            "file_count": file_count,
                        }
                    )

    sessions.sort(key=lambda x: x["created_at"], reverse=True)

    return {"total_sessions": len(sessions), "sessions": sessions}


def _locate_file(file_path: str, session_id: str = None):
    """Localiza un archivo en los directorios de sesión siguiendo la misma estrategia que el endpoint original."""
    if os.path.isabs(file_path):
        return file_path if os.path.exists(file_path) else None

    found_file = None

    if session_id:
        for base_dir in SESSION_DIRS:
            if os.path.exists(base_dir):
                session_path = os.path.join(base_dir, session_id)
                if os.path.isdir(session_path):
                    potential_file = os.path.join(session_path, file_path)
                    if os.path.exists(potential_file):
                        found_file = potential_file
                        break

    if found_file is None:
        session_files = []
        for base_dir in SESSION_DIRS:
            if os.path.exists(base_dir):
                for session_folder in os.listdir(base_dir):
                    session_path = os.path.join(base_dir, session_folder)
                    if os.path.isdir(session_path):
                        potential_file = os.path.join(session_path, file_path)
                        if os.path.exists(potential_file):
                            session_files.append((potential_file, os.path.getmtime(potential_file)))

        if session_files:
            session_files.sort(key=lambda x: x[1], reverse=True)
            found_file = session_files[0][0]

    if found_file is None:
        for base_dir in SESSION_DIRS:
            fallback_path = os.path.join(base_dir, file_path)
            if os.path.exists(fallback_path):
                found_file = fallback_path
                break

    if found_file is None:
        from app.core.config import BASE_DIR
        root_fallback = os.path.join(BASE_DIR, file_path)
        if os.path.exists(root_fallback):
            found_file = root_fallback

    return found_file


def build_download_response(file_path: str, session_id: str = None) -> FileResponse:
    """Construye la respuesta de descarga de un archivo ubicado en sesiones."""
    full_path = _locate_file(file_path, session_id)

    if not full_path or not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {os.path.basename(file_path)}")

    filename = os.path.basename(full_path)
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".png":
        media_type = "image/png"
    elif file_extension == ".xlsx":
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_extension in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"

    response = FileResponse(path=full_path, media_type=media_type, filename=filename)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
