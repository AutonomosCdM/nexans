"""
🟢 GREEN PHASE - Implementación mínima Pricing models
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PricingRequest(BaseModel):
    """Modelo para solicitudes de pricing"""
    cable_ref: str
    quantity: int
    customer_segment: Optional[str] = "general"
    urgency_level: Optional[str] = "normal"
    delivery_location: Optional[str] = None
    
    def is_valid(self) -> bool:
        """🟢 GREEN: Implementación mínima"""
        return True
    
    def get_urgency_multiplier(self) -> float:
        """🟢 GREEN: Multiplier por urgencia"""
        if self.urgency_level == "high":
            return 1.1
        elif self.urgency_level == "urgent":
            return 1.2
        return 1.0


class PricingResponse(BaseModel):
    """Modelo para respuestas de pricing"""
    cable_ref: str
    suggested_price_usd: float
    confidence_score: float
    justification: List[str]
    lme_copper_price: float
    calculation_timestamp: datetime