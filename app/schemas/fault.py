from typing import Dict, List, Optional

from pydantic import BaseModel


class LayerFaultConfig(BaseModel):
    enabled: bool = False
    num_faults: int = 1
    fault_type: str = "random"  # "random" o "specific"
    positions: Optional[List[List[int]]] = None
    bit_positions: Optional[List[int]] = None


class FaultInjectionConfig(BaseModel):
    enabled: bool = False
    layers: Dict[str, LayerFaultConfig] = {}


class FaultInjectorRequest(BaseModel):
    model_path: str
    image_data: Optional[str] = None
    fault_config: Optional[FaultInjectionConfig] = None


class FaultInjectorInferenceRequest(BaseModel):
    model_path: str
    fault_config: Optional[FaultInjectionConfig] = None


class FaultCampaignRequest(BaseModel):
    model_path: str
    num_samples: int = 100
    fault_config: dict
    image_dir: Optional[str] = None


class WeightFaultCampaignRequest(BaseModel):
    model_path: str
    num_samples: int = 100
    weight_fault_config: dict
    image_dir: Optional[str] = None


class StuckAtAsymmetryRequest(BaseModel):
    """Request for the SAI (Stuck-at Asymmetry Index) campaign.

    `base_config` follows the same shape as weight_fault_config; its
    `fault_type` field is overridden to stuck_at_0 / stuck_at_1 by the
    service. `granularity` is "global" (single paired sweep) or
    "per_layer" (one paired sweep per layer in addition to the global).
    """

    model_path: str
    num_samples: int = 100
    base_config: dict
    granularity: str = "global"
    image_dir: Optional[str] = None
