"""
🟢 GREEN PHASE - API Response Models
Sprint 2.2.2: API response models for consistent outputs

PYDANTIC MODELS FOR API RESPONSES:
✅ QuoteGenerationResponse: Complete quote with breakdown
✅ PricingCalculationResponse: Detailed pricing calculation
✅ CableSearchResponse: Cable search results
✅ ErrorResponse: Standardized error format
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal


class PricingBreakdownResponse(BaseModel):
    """🟢 GREEN: Detailed pricing breakdown"""
    base_cost: float = Field(..., description="Base material and manufacturing cost")
    segment_multiplier: float = Field(..., description="Customer segment multiplier")
    volume_discount: float = Field(..., description="Volume discount rate applied")
    regional_factor: float = Field(..., description="Regional pricing factor")
    urgency_multiplier: float = Field(..., description="Urgency delivery multiplier")
    ml_predicted_price: Optional[float] = Field(None, description="ML model predicted price")
    margin_optimized_price: Optional[float] = Field(None, description="Margin optimized price")
    final_price: float = Field(..., description="Final calculated price per meter")


class BusinessRulesAppliedResponse(BaseModel):
    """🟢 GREEN: Business rules applied in calculation"""
    segment_multiplier: float = Field(..., description="Segment multiplier applied")
    volume_discount: float = Field(..., description="Volume discount applied")
    regional_factor: float = Field(..., description="Regional factor applied")
    urgency_multiplier: float = Field(..., description="Urgency multiplier applied")
    tier_bonus: float = Field(default=0.0, description="Customer tier bonus")


class LMEPricesResponse(BaseModel):
    """🟢 GREEN: Current LME prices"""
    copper_usd_per_ton: float = Field(..., description="Copper price USD per ton")
    aluminum_usd_per_ton: float = Field(..., description="Aluminum price USD per ton")
    timestamp: datetime = Field(..., description="Price timestamp")
    source: str = Field(default="LME", description="Price source")


class CostBreakdownResponse(BaseModel):
    """🟢 GREEN: Detailed cost breakdown"""
    copper_cost: float = Field(..., description="Copper material cost")
    aluminum_cost: float = Field(..., description="Aluminum material cost")
    polymer_cost: float = Field(..., description="Polymer/insulation cost")
    manufacturing_cost: float = Field(..., description="Manufacturing cost")
    overhead_cost: float = Field(..., description="Overhead cost")
    total_cost: float = Field(..., description="Total cost per meter")


class QuoteGenerationResponse(BaseModel):
    """🟢 GREEN: Complete quote generation response"""
    quote_id: str = Field(..., description="Unique quote identifier")
    cable_reference: str = Field(..., description="Nexans cable reference")
    customer: Dict[str, Any] = Field(..., description="Customer information")
    pricing_breakdown: PricingBreakdownResponse = Field(..., description="Detailed pricing breakdown")
    total_price: float = Field(..., description="Total price for order quantity")
    quote_timestamp: datetime = Field(..., description="Quote generation timestamp")
    validity_period: int = Field(default=30, description="Quote validity in days")
    terms_conditions: Optional[str] = Field(None, description="Terms and conditions")


class PricingCalculationResponse(BaseModel):
    """🟢 GREEN: Pricing calculation response"""
    calculation_id: str = Field(..., description="Unique calculation identifier")
    cable_specs: Dict[str, Any] = Field(..., description="Cable specifications used")
    cost_breakdown: CostBreakdownResponse = Field(..., description="Detailed cost breakdown")
    business_rules_applied: BusinessRulesAppliedResponse = Field(..., description="Business rules applied")
    final_price: float = Field(..., description="Final calculated price per meter")
    lme_prices: LMEPricesResponse = Field(..., description="Current LME prices used")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")


class CableInfoResponse(BaseModel):
    """🟢 GREEN: Cable information response"""
    nexans_reference: str = Field(..., description="Nexans reference number")
    product_name: str = Field(..., description="Product name")
    voltage_rating: int = Field(..., description="Voltage rating in volts")
    current_rating: Optional[int] = Field(None, description="Current rating in amperes")
    conductor_section_mm2: Optional[float] = Field(None, description="Conductor section")
    weight_kg_per_km: Optional[float] = Field(None, description="Cable weight")
    applications: List[str] = Field(..., description="Cable applications")
    copper_content_kg: Optional[float] = Field(None, description="Copper content per km")
    aluminum_content_kg: Optional[float] = Field(None, description="Aluminum content per km")
    manufacturing_complexity: Optional[str] = Field(None, description="Manufacturing complexity")


class PaginationResponse(BaseModel):
    """🟢 GREEN: Pagination information"""
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Results per page")
    total_pages: int = Field(..., description="Total number of pages")
    total_results: int = Field(..., description="Total number of results")


class CableSearchResponse(BaseModel):
    """🟢 GREEN: Cable search results response"""
    cables: List[CableInfoResponse] = Field(..., description="Found cables")
    search_criteria: Dict[str, Any] = Field(..., description="Search criteria used")
    pagination: PaginationResponse = Field(..., description="Pagination information")
    total_results: int = Field(..., description="Total number of matching cables")


class CurrentPricesResponse(BaseModel):
    """🟢 GREEN: Current prices response"""
    lme_prices: LMEPricesResponse = Field(..., description="Current LME prices")
    timestamp: datetime = Field(..., description="Response timestamp")
    source: str = Field(default="Nexans API", description="Data source")


class CableCurrentPriceResponse(BaseModel):
    """🟢 GREEN: Current price for specific cable"""
    cable_reference: str = Field(..., description="Cable reference")
    current_price: float = Field(..., description="Current price per meter")
    cost_breakdown: CostBreakdownResponse = Field(..., description="Cost breakdown")
    last_updated: datetime = Field(..., description="Last price update")
    lme_prices: LMEPricesResponse = Field(..., description="LME prices used")


class ErrorResponse(BaseModel):
    """🟢 GREEN: Standardized error response"""
    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    path: Optional[str] = Field(None, description="Request path that caused error")
    validation_errors: Optional[List[Dict[str, Any]]] = Field(None, description="Validation errors if applicable")


class HealthResponse(BaseModel):
    """🟢 GREEN: Health check response"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Service status breakdown")


class APIInfoResponse(BaseModel):
    """🟢 GREEN: API information response"""
    title: str = Field(..., description="API title")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    timestamp: datetime = Field(..., description="Response timestamp")