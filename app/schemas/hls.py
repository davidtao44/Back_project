from typing import Optional

from pydantic import BaseModel, Field


class QuantizeRequest(BaseModel):
    model_name: str
    total_bits: int = Field(default=16, ge=4, le=32)
    int_bits: int = Field(default=6, ge=1, le=16)


class HLSConvertRequest(BaseModel):
    model_name: str
    backend: str = "Vivado"
    precision: str = "ap_fixed<16,6>"
    reuse_factor: int = Field(default=1, ge=1)
    clock_period: int = Field(default=5, ge=1)
    io_type: str = "io_parallel"
    strategy: str = "Latency"
    board: Optional[str] = None
    part: Optional[str] = None
