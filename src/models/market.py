"""
🟢 GREEN PHASE - Implementación mínima Market data models
"""

from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class MarketData(BaseModel):
    """Market intelligence data model"""
    timestamp: datetime = Field(default_factory=datetime.now)
    lme_prices: Dict[str, float] = {}
    volatility_index: float = 0.0
    market_sentiment: str = "neutral"
    price_alerts: List[str] = []
    trend_analysis: Dict[str, str] = {}
    competitive_analysis: Optional[Dict[str, float]] = None


class LMEPriceData(BaseModel):
    """Modelo para datos LME real-time"""
    metal: str
    price_usd_per_ton: float
    timestamp: datetime
    exchange: str = "LME"
    currency: str = "USD"
    change_percent: float = 0.0
    
    def is_fresh(self, max_age_minutes: int = 60) -> bool:
        """🟢 GREEN: Check if data is fresh"""
        age = datetime.now() - self.timestamp
        return age.total_seconds() < (max_age_minutes * 60)
    
    def get_price_per_kg(self) -> float:
        """🟢 GREEN: Convert ton to kg"""
        return self.price_usd_per_ton / 1000