from fastapi import APIRouter, Depends, HTTPException

from app.schemas.fault import FaultCampaignRequest, WeightFaultCampaignRequest
from app.services.auth_service import get_current_user
from app.services.fault_campaign_service import (
    get_job_results,
    get_job_status,
    run_activation_fault_campaign,
    run_weight_fault_campaign_request,
    start_weight_fault_campaign_job,
)
from app.services.model_service import get_available_models_for_campaign

router = APIRouter(tags=["fault-campaign"])


@router.post("/fault_campaign/run/")
async def run_fault_campaign(
    request: FaultCampaignRequest, current_user: dict = Depends(get_current_user)
):
    return run_activation_fault_campaign(request, current_user)


@router.post("/fault_campaign/weight/run/")
async def run_weight_fault_campaign(
    request: WeightFaultCampaignRequest, current_user: dict = Depends(get_current_user)
):
    return run_weight_fault_campaign_request(request, current_user)


# ── Async job endpoints ───────────────────────────────────────────────────────

@router.post("/fault_campaign/weight/start/")
async def start_weight_fault_campaign(
    request: WeightFaultCampaignRequest, current_user: dict = Depends(get_current_user)
):
    """Lanza la campaña en background y devuelve job_id inmediatamente."""
    job_id = start_weight_fault_campaign_job(request, current_user)
    return {"job_id": job_id, "status": "pending"}


@router.get("/fault_campaign/status/{job_id}")
async def get_campaign_status(job_id: str, current_user: dict = Depends(get_current_user)):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    return status


@router.get("/fault_campaign/results/{job_id}")
async def get_campaign_results(job_id: str, current_user: dict = Depends(get_current_user)):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    if status["status"] != "done":
        raise HTTPException(status_code=202, detail=f"Job aún en progreso: {status['status']}")
    results = get_job_results(job_id)
    return results


@router.get("/fault_campaign/models/")
async def get_available_models(current_user: dict = Depends(get_current_user)):
    try:
        return get_available_models_for_campaign()
    except Exception as e:
        print(f"❌ Error obteniendo modelos: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo modelos disponibles: {str(e)}"
        )
