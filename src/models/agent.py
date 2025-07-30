"""
🟢 GREEN PHASE - Implementación mínima Agent response model
"""

from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime


class AgentResponse(BaseModel):
    """Modelo para respuestas de Agentes IA"""
    agent_type: str
    query: str
    response_text: str
    confidence: float
    data_sources: List[str]
    recommendations: List[str]
    timestamp: datetime
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ["market_intelligence", "forecasting", "quote_generation", "customer_analysis"]
        if v not in valid_types:
            raise ValueError(f"agent_type must be one of {valid_types}")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v