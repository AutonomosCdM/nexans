"""
🟢 GREEN PHASE - API Request Models
Sprint 2.2.2: API request models for validation

PYDANTIC MODELS FOR API REQUESTS:
✅ QuoteGenerationRequest: Complete quote generation
✅ PricingCalculationRequest: Pricing calculations
✅ CableSearchRequest: Cable search filters
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal


class CustomerRequest(BaseModel):
    """🟢 GREEN: Customer information for requests"""
    name: Optional[str] = Field(None, description="Customer name")
    segment: str = Field(..., description="Customer segment: mining, industrial, utility, residential")
    tier: str = Field(default="standard", description="Customer tier: enterprise, government, standard")
    region: str = Field(default="chile_central", description="Customer region")
    
    @validator('segment')
    def validate_segment(cls, v):
        valid_segments = ['mining', 'industrial', 'utility', 'residential']
        if v.lower() not in valid_segments:
            raise ValueError(f'Segment must be one of: {valid_segments}')
        return v.lower()
    
    @validator('tier')
    def validate_tier(cls, v):
        valid_tiers = ['enterprise', 'government', 'standard', 'retail']
        if v.lower() not in valid_tiers:
            raise ValueError(f'Tier must be one of: {valid_tiers}')
        return v.lower()


class OrderRequest(BaseModel):
    """🟢 GREEN: Order information for requests"""
    quantity: int = Field(..., gt=0, description="Order quantity in meters")
    urgency: str = Field(default="standard", description="Delivery urgency: standard, urgent, express")
    delivery_date: Optional[date] = Field(None, description="Requested delivery date")
    
    @validator('urgency')
    def validate_urgency(cls, v):
        valid_urgencies = ['standard', 'urgent', 'express']
        if v.lower() not in valid_urgencies:
            raise ValueError(f'Urgency must be one of: {valid_urgencies}')
        return v.lower()
    
    @validator('delivery_date')
    def validate_delivery_date(cls, v):
        if v and v < date.today():
            raise ValueError('Delivery date cannot be in the past')
        return v


class PricingOptionsRequest(BaseModel):
    """🟢 GREEN: Pricing options for requests"""
    include_volume_discount: bool = Field(default=True, description="Apply volume discounts")
    include_regional_factors: bool = Field(default=True, description="Apply regional pricing")
    include_urgency_premium: bool = Field(default=True, description="Apply urgency premiums")
    use_real_time_lme: bool = Field(default=True, description="Use real-time LME prices")
    apply_business_rules: bool = Field(default=True, description="Apply business rules")
    optimize_margin: bool = Field(default=True, description="Optimize profit margins")


class QuoteGenerationRequest(BaseModel):
    """🟢 GREEN: Complete quote generation request"""
    cable_reference: str = Field(..., description="Nexans cable reference number")
    customer: CustomerRequest = Field(..., description="Customer information")
    order: OrderRequest = Field(..., description="Order details")
    pricing_options: Optional[PricingOptionsRequest] = Field(
        default_factory=PricingOptionsRequest,
        description="Pricing calculation options"
    )
    
    @validator('cable_reference')
    def validate_cable_reference(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Cable reference cannot be empty')
        return v.strip()


class CableSpecsRequest(BaseModel):
    """🟢 GREEN: Cable specifications for pricing"""
    nexans_reference: Optional[str] = Field(None, description="Nexans reference if available")
    voltage_rating: int = Field(..., gt=0, description="Voltage rating in volts")
    current_rating: Optional[int] = Field(None, gt=0, description="Current rating in amperes")
    copper_content_kg: float = Field(default=0.0, ge=0, description="Copper content in kg per km")
    aluminum_content_kg: float = Field(default=0.0, ge=0, description="Aluminum content in kg per km")
    conductor_section_mm2: Optional[float] = Field(None, gt=0, description="Conductor section in mm²")
    weight_kg_per_km: Optional[float] = Field(None, gt=0, description="Cable weight in kg per km")
    applications: List[str] = Field(default_factory=list, description="Cable applications")
    
    @validator('applications')
    def validate_applications(cls, v):
        valid_apps = ['mining', 'industrial', 'utility', 'residential', 'marine', 'construction']
        for app in v:
            if app.lower() not in valid_apps:
                raise ValueError(f'Application {app} not valid. Must be one of: {valid_apps}')
        return [app.lower() for app in v]


class PricingCalculationRequest(BaseModel):
    """🟢 GREEN: Pricing calculation request"""
    cable_specs: CableSpecsRequest = Field(..., description="Cable specifications")
    customer: CustomerRequest = Field(..., description="Customer information")
    order: OrderRequest = Field(..., description="Order details")
    options: Optional[PricingOptionsRequest] = Field(
        default_factory=PricingOptionsRequest,
        description="Calculation options"
    )


class CableSearchRequest(BaseModel):
    """🟢 GREEN: Cable search request"""
    reference: Optional[str] = Field(None, description="Search by Nexans reference")
    voltage_min: Optional[int] = Field(None, ge=0, description="Minimum voltage rating")
    voltage_max: Optional[int] = Field(None, ge=0, description="Maximum voltage rating")
    current_min: Optional[int] = Field(None, ge=0, description="Minimum current rating")
    current_max: Optional[int] = Field(None, ge=0, description="Maximum current rating")
    application: Optional[str] = Field(None, description="Cable application")
    conductor_material: Optional[str] = Field(None, description="Conductor material: copper, aluminum")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    limit: int = Field(default=20, ge=1, le=100, description="Results per page (max 100)")
    
    @validator('voltage_max')
    def validate_voltage_range(cls, v, values):
        if v and 'voltage_min' in values and values['voltage_min']:
            if v < values['voltage_min']:
                raise ValueError('Maximum voltage must be greater than minimum voltage')
        return v
    
    @validator('current_max')
    def validate_current_range(cls, v, values):
        if v and 'current_min' in values and values['current_min']:
            if v < values['current_min']:
                raise ValueError('Maximum current must be greater than minimum current')
        return v
    
    @validator('application')
    def validate_application(cls, v):
        if v:
            valid_apps = ['mining', 'industrial', 'utility', 'residential', 'marine', 'construction']
            if v.lower() not in valid_apps:
                raise ValueError(f'Application must be one of: {valid_apps}')
            return v.lower()
        return v
    
    @validator('conductor_material')
    def validate_conductor_material(cls, v):
        if v:
            valid_materials = ['copper', 'aluminum', 'copper_aluminum']
            if v.lower() not in valid_materials:
                raise ValueError(f'Conductor material must be one of: {valid_materials}')
            return v.lower()
        return v