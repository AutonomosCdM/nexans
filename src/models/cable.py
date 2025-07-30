"""
♻️ REFACTOR PHASE - CableProduct model con features avanzadas
"""

from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from decimal import Decimal
from enum import Enum


class CableType(str, Enum):
    """Tipos de cable soportados"""
    POWER = "power"
    CONTROL = "control" 
    INSTRUMENTATION = "instrumentation"
    TELECOMMUNICATIONS = "telecommunications"


class ApplicationArea(str, Enum):
    """Áreas de aplicación"""
    MINING = "mining"
    INDUSTRIAL = "industrial"
    UTILITY = "utility" 
    RESIDENTIAL = "residential"
    MARINE = "marine"


class CableProduct(BaseModel):
    """♻️ REFACTOR: Modelo completo para productos de cable Nexans"""
    nexans_ref: str = Field(..., description="Referencia única Nexans")
    product_name: Optional[str] = Field(None, description="Nombre comercial del producto")
    
    # Material composition
    copper_content_kg: Optional[Decimal] = Field(None, description="Contenido de cobre en kg/km")
    aluminum_content_kg: Optional[Decimal] = Field(Decimal("0.0"), description="Contenido de aluminio en kg/km")
    polymer_content_kg: Optional[Decimal] = Field(None, description="Contenido de polímeros en kg/km")
    
    # Electrical specifications
    voltage_rating: Optional[int] = Field(None, description="Tensión nominal en V")
    current_rating: Optional[Decimal] = Field(None, description="Corriente nominal en A")
    conductor_section_mm2: Optional[Decimal] = Field(None, description="Sección del conductor en mm²")
    
    # Physical properties
    outer_diameter_mm: Optional[Decimal] = Field(None, description="Diámetro exterior en mm")
    weight_kg_per_km: Optional[Decimal] = Field(None, description="Peso total en kg/km")
    
    # Classification
    cable_type: Optional[CableType] = Field(None, description="Tipo de cable")
    applications: Optional[List[ApplicationArea]] = Field(default_factory=list, description="Aplicaciones recomendadas")
    
    # Cost factors
    manufacturing_complexity: Optional[int] = Field(1, ge=1, le=5, description="Complejidad de fabricación (1-5)")
    minimum_order_quantity: Optional[int] = Field(100, description="Cantidad mínima de pedido")
    
    # Metadata
    datasheet_url: Optional[str] = Field(None, description="URL del datasheet")
    last_updated: Optional[str] = Field(None, description="Última actualización de precios")
    
    @validator('nexans_ref')
    def validate_nexans_ref(cls, v):
        if not v or v.strip() == "":
            raise ValueError("nexans_ref cannot be empty")
        # Validar formato Nexans (ej: 540317340)
        if not v.isdigit() or len(v) != 9:
            raise ValueError("nexans_ref must be 9 digits")
        return v
    
    @validator('copper_content_kg', 'aluminum_content_kg', 'polymer_content_kg')
    def validate_material_content(cls, v):
        if v is not None and v < 0:
            raise ValueError("Material content cannot be negative")
        return v
    
    @validator('voltage_rating')
    def validate_voltage(cls, v):
        if v is not None and v <= 0:
            raise ValueError("voltage_rating must be positive")
        return v
    
    def get_total_material_cost(self, copper_price_usd_kg: float, aluminum_price_usd_kg: float) -> Decimal:
        """♻️ REFACTOR: Calcular costo total de materiales"""
        total_cost = Decimal("0.0")
        
        if self.copper_content_kg:
            total_cost += self.copper_content_kg * Decimal(str(copper_price_usd_kg))
        
        if self.aluminum_content_kg:
            total_cost += self.aluminum_content_kg * Decimal(str(aluminum_price_usd_kg))
            
        return total_cost
    
    def get_complexity_multiplier(self) -> Decimal:
        """♻️ REFACTOR: Multiplier basado en complejidad de fabricación"""
        multipliers = {
            1: Decimal("1.0"),   # Básico
            2: Decimal("1.1"),   # Simple
            3: Decimal("1.25"),  # Medio
            4: Decimal("1.4"),   # Complejo
            5: Decimal("1.6")    # Muy complejo
        }
        return multipliers.get(self.manufacturing_complexity, Decimal("1.0"))
    
    def is_mining_suitable(self) -> bool:
        """♻️ REFACTOR: Check if suitable for mining applications"""
        return ApplicationArea.MINING in (self.applications or [])
    
    def get_power_density(self) -> Optional[Decimal]:
        """♻️ REFACTOR: Calculate power density if specs available"""
        if self.voltage_rating and self.current_rating:
            power_w = self.voltage_rating * self.current_rating
            return Decimal(str(power_w))
        return None
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True