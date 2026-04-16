from fastapi import APIRouter, Depends, HTTPException

from app.schemas.fault import FaultCampaignRequest, WeightFaultCampaignRequest
from app.services.auth_service import get_current_user
from app.services.fault_campaign_service import (
    run_activation_fault_campaign,
    run_weight_fault_campaign_request,
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


@router.get("/fault_campaign/models/")
async def get_available_models(current_user: dict = Depends(get_current_user)):
    try:
        return get_available_models_for_campaign()
    except Exception as e:
        print(f"❌ Error obteniendo modelos: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo modelos disponibles: {str(e)}"
        )
