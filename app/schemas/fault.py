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
