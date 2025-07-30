"""
🟢 GREEN PHASE - Implementación mínima Customer model
"""

from pydantic import BaseModel, validator
from typing import Optional


class Customer(BaseModel):
    """Modelo para clientes y segmentación"""
    name: str
    segment: str
    credit_rating: Optional[str] = "BBB"
    historical_margin_avg: Optional[float] = 0.15
    payment_terms_days: Optional[int] = 30
    is_strategic: Optional[bool] = False
    
    @validator('segment')
    def validate_segment(cls, v):
        valid_segments = ["mining", "industrial", "utility", "residential"]
        if v not in valid_segments:
            raise ValueError(f"segment must be one of {valid_segments}")
        return v
    
    def get_margin_multiplier(self) -> float:
        """🟢 GREEN: Multiplier basado en segmento"""
        multipliers = {
            "mining": 1.25,
            "industrial": 1.15,
            "utility": 1.10,
            "residential": 1.05
        }
        return multipliers.get(self.segment, 1.0)