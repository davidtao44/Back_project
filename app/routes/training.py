"""Rutas del módulo de construcción y entrenamiento de CNN."""

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.training import ModelBuildRequest, TrainingRequest
from app.services.auth_service import get_current_user
from app.services.cnn_builder_service import (
    build_model,
    get_template,
    list_layer_types,
    list_templates,
    save_built_model,
)
from app.services.dataset_service import DATASETS, get_dataset_preview, list_datasets
from app.services.training_service import (
    get_training_result,
    get_training_status,
    start_training_job,
)

router = APIRouter(tags=["training"])


# ── Datasets ──────────────────────────────────────────────────────────────────
@router.get("/datasets/")
def get_datasets(current_user: dict = Depends(get_current_user)):
    """Lista los datasets de Keras disponibles (no descarga nada)."""
    return list_datasets()


@router.get("/datasets/{name}/preview")
def get_dataset_preview_route(
    name: str, n: int = 8, current_user: dict = Depends(get_current_user)
):
    """Devuelve imágenes de muestra de un dataset (descarga la primera vez)."""
    try:
        return get_dataset_preview(name, n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Constructor de CNN ────────────────────────────────────────────────────────
@router.get("/cnn/layer-types/")
def get_layer_types(current_user: dict = Depends(get_current_user)):
    """Tipos de capa soportados y sus parámetros (para el constructor)."""
    return list_layer_types()


@router.get("/cnn/templates/")
def get_templates(current_user: dict = Depends(get_current_user)):
    """Plantillas de arquitectura disponibles."""
    return list_templates()


@router.get("/cnn/template/{name}")
def get_template_route(
    name: str, dataset: str, current_user: dict = Depends(get_current_user)
):
    """Devuelve una plantilla resuelta para la entrada/clases de un dataset."""
    if dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset desconocido: '{dataset}'")
    meta = DATASETS[dataset]
    try:
        return get_template(name, meta["input_shape"], meta["num_classes"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/cnn/build/")
def build_cnn(
    request: ModelBuildRequest, current_user: dict = Depends(get_current_user)
):
    """Construye y guarda un modelo CNN desde un spec, sin entrenarlo."""
    try:
        spec = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        model = build_model(spec)
        path = save_built_model(model, request.name)
        return {
            "success": True,
            "model_path": path,
            "name": request.name,
            "layers": len(model.layers),
            "parameters": int(model.count_params()),
            "input_shape": list(request.input_shape),
            "output_shape": list(model.output_shape),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al construir el modelo: {e}")


# ── Entrenamiento ─────────────────────────────────────────────────────────────
@router.post("/training/start/")
def start_training(
    request: TrainingRequest, current_user: dict = Depends(get_current_user)
):
    """Inicia un job de entrenamiento en segundo plano y devuelve su job_id."""
    if request.model_spec is None and not request.model_path:
        raise HTTPException(
            status_code=400, detail="Debe indicar model_spec o model_path"
        )
    if request.dataset not in DATASETS:
        raise HTTPException(
            status_code=400, detail=f"Dataset desconocido: '{request.dataset}'"
        )
    job_id = start_training_job(request)
    return {"job_id": job_id, "status": "pending"}


@router.get("/training/status/{job_id}")
def training_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """Progreso del job de entrenamiento (época, loss, accuracy)."""
    status = get_training_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    return status


@router.get("/training/result/{job_id}")
def training_result(job_id: str, current_user: dict = Depends(get_current_user)):
    """Resultado final del entrenamiento (modelo guardado + métricas)."""
    status = get_training_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    if status["status"] != "done":
        raise HTTPException(
            status_code=202, detail=f"Job aún en progreso: {status['status']}"
        )
    return get_training_result(job_id)
