"""
🟢 GREEN PHASE - Implementación mínima Forecast model
"""

from pydantic import BaseModel
from typing import List, Tuple, Optional


class DemandForecast(BaseModel):
    """Modelo para forecasting de demanda"""
    cable_ref: str
    forecast_months: int
    predicted_demand: List[int]
    confidence_intervals: List[Tuple[int, int]]
    seasonality_factor: float
    trend_direction: str
    
    def get_peak_month(self) -> Optional[int]:
        """🟢 GREEN: Retorna mes con mayor demanda predicha"""
        if not self.predicted_demand:
            return None
        max_demand = max(self.predicted_demand)
        return self.predicted_demand.index(max_demand) + 1  # 1-indexed