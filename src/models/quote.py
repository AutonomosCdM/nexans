"""
🟢 GREEN PHASE - Implementación mínima Quote model
"""

from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta


class QuoteRequest(BaseModel):
    """Request model for quote generation"""
    customer_id: str
    customer_segment: str = "mining"
    product_id: str
    quantity: int
    delivery_location: str = "chile_central"
    delivery_deadline: Optional[str] = None
    special_requirements: List[str] = []
    budget_limit: Optional[float] = None


class QuoteResponse(BaseModel):
    """Response model for generated quotes"""
    quote_id: str
    customer_id: str
    product_id: str
    quantity: int
    unit_price_usd: float
    total_price_usd: float
    currency: str = "USD"
    validity_days: int = 30
    delivery_location: str
    estimated_delivery_days: int = 30
    confidence_score: float
    breakdown: Dict[str, float] = {}
    terms_conditions: List[str] = []
    generated_at: datetime = Field(default_factory=datetime.now)


class Quote(BaseModel):
    """Modelo para cotizaciones automáticas"""
    quote_id: str
    customer_name: str
    cable_ref: str
    quantity: int
    unit_price_usd: float
    total_price_usd: float
    validity_days: int = 30
    terms_conditions: List[str]
    generated_by: str
    confidence_score: float
    
    @validator('total_price_usd')
    def validate_total_price(cls, v, values):
        if 'quantity' in values and 'unit_price_usd' in values:
            expected = values['quantity'] * values['unit_price_usd']
            if abs(v - expected) > 0.01:  # Allow small rounding errors
                raise ValueError("total_price_usd must equal quantity * unit_price_usd")
        return v
    
    def is_valid_quote(self) -> bool:
        """🟢 GREEN: Validar cotización"""
        return (
            self.unit_price_usd > 0 and
            self.quantity > 0 and
            self.confidence_score > 0.5
        )
    
    def get_expiry_date(self) -> datetime:
        """🟢 GREEN: Fecha de expiración"""
        return datetime.now() + timedelta(days=self.validity_days)