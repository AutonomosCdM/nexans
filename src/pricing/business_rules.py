"""
🟢 GREEN PHASE - Business Rules Implementation
Sprint 2.2.1: Business rules por segmento cliente

IMPLEMENTATION TO MAKE TESTS PASS:
✅ BusinessRulesEngine: Main orchestrator
✅ CustomerSegmentationError: Custom exception
✅ VolumeDiscountCalculator: Volume-based discounts  
✅ RegionalPricingEngine: Regional pricing factors
✅ MarginOptimizer: Margin optimization logic
✅ PriorityOrderProcessor: Urgency-based pricing
✅ CustomerTierValidator: Customer tier validation

All implementations follow TDD methodology - minimal code to pass tests.
"""

from typing import Dict, Optional, Any, Union
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)


class CustomerSegmentationError(Exception):
    """🟢 GREEN: Custom exception for customer segmentation errors"""
    pass


class BusinessRulesEngine:
    """🟢 GREEN: Main business rules orchestrator"""
    
    def __init__(self):
        # Customer segment multipliers (matches cost calculator integration)
        self.customer_segments = {
            "mining": 1.5,      # Highest premium - harsh environments
            "industrial": 1.3,  # Standard industrial premium
            "utility": 1.2,     # Utility grade premium
            "residential": 1.0  # Base pricing
        }
        
        # Initialize sub-components
        self.volume_calculator = VolumeDiscountCalculator()
        self.regional_engine = RegionalPricingEngine()
        self.margin_optimizer = MarginOptimizer()
        self.priority_processor = PriorityOrderProcessor()
        self.tier_validator = CustomerTierValidator()
    
    def get_customer_segment_multiplier(self, segment: str) -> float:
        """🟢 GREEN: Get multiplier for customer segment"""
        if segment not in self.customer_segments:
            raise CustomerSegmentationError(f"Invalid customer segment: {segment}")
        
        return self.customer_segments[segment]
    
    def apply_segment_rules(self, customer, base_price: float) -> Dict[str, float]:
        """🟢 GREEN: Apply customer segment rules to pricing"""
        try:
            segment = customer.segment.lower()
            multiplier = self.get_customer_segment_multiplier(segment)
            adjusted_price = base_price * multiplier
            
            return {
                "segment": segment,
                "multiplier": multiplier,
                "base_price": base_price,
                "adjusted_price": adjusted_price
            }
            
        except AttributeError:
            raise CustomerSegmentationError("Customer object missing required segment attribute")
        except Exception as e:
            raise CustomerSegmentationError(f"Error applying segment rules: {e}")
    
    def apply_complete_pricing_rules(self, customer, order) -> Dict[str, Any]:
        """🟢 GREEN: Apply complete business rules workflow"""
        try:
            # Validate inputs
            if order.quantity <= 0:
                raise ValueError("Order quantity must be positive")
            if order.base_cost <= 0:
                raise ValueError("Base cost must be positive")
            
            results = {}
            current_price = order.base_cost
            
            # 1. Apply segment multiplier
            segment_result = self.apply_segment_rules(customer, current_price)
            results["segment_multiplier"] = segment_result["multiplier"]
            current_price = segment_result["adjusted_price"]
            
            # 2. Apply volume discount
            volume_result = self.volume_calculator.apply_volume_discount(
                base_price=current_price,
                quantity=order.quantity,
                customer_tier=getattr(customer, 'tier', 'standard')
            )
            results["volume_discount"] = volume_result["discount_rate"]
            current_price = volume_result["final_price"]
            
            # 3. Apply regional pricing
            region = getattr(customer, 'region', 'chile_central')
            regional_result = self.regional_engine.apply_regional_pricing(
                base_price=current_price,
                region=region
            )
            results["regional_factor"] = regional_result["regional_factor"]
            current_price = regional_result["adjusted_price"]
            
            # 4. Apply urgency multiplier
            urgency = getattr(order, 'urgency', 'standard')
            priority_result = self.priority_processor.apply_priority_pricing(
                base_price=current_price,
                urgency=urgency
            )
            results["urgency_multiplier"] = priority_result["urgency_multiplier"]
            current_price = priority_result["adjusted_price"]
            
            # 5. Apply margin optimization
            margin_result = self.margin_optimizer.optimize_margin(
                cost_price=order.base_cost,
                segment=customer.segment
            )
            results["margin_optimized_price"] = margin_result["optimized_price"]
            
            # Final price is the higher of business rules price or margin-optimized price
            final_price = max(current_price, margin_result["optimized_price"])
            results["final_price"] = round(final_price, 2)
            
            return results
            
        except Exception as e:
            if isinstance(e, (CustomerSegmentationError, ValueError)):
                raise
            raise CustomerSegmentationError(f"Error in complete pricing workflow: {e}")


class VolumeDiscountCalculator:
    """🟢 GREEN: Volume discount calculations"""
    
    def __init__(self):
        # Volume discount tiers (quantity in meters)
        self.tier_thresholds = [
            (0, 100),      # Tier 1: 1-100m
            (101, 500),    # Tier 2: 101-500m  
            (501, 1000),   # Tier 3: 501-1000m
            (1001, 5000),  # Tier 4: 1001-5000m
            (5001, float('inf'))  # Tier 5: 5000m+
        ]
        
        # Discount rates by tier
        self.discount_rates = [0.0, 0.03, 0.05, 0.08, 0.12]
        
        # Customer tier bonuses
        self.tier_bonuses = {
            "enterprise": 0.01,  # +1% discount
            "government": 0.02,  # +2% discount
            "standard": 0.0,     # No bonus
            "retail": 0.0        # No bonus
        }
    
    def calculate_volume_discount(self, quantity: int, customer_tier: str = "standard") -> float:
        """🟢 GREEN: Calculate volume discount rate"""
        # Find appropriate tier
        discount_rate = 0.0
        for i, (min_qty, max_qty) in enumerate(self.tier_thresholds):
            if min_qty <= quantity <= max_qty:
                discount_rate = self.discount_rates[i]
                break
        
        # Add customer tier bonus
        tier_bonus = self.tier_bonuses.get(customer_tier, 0.0)
        total_discount = discount_rate + tier_bonus
        
        return total_discount
    
    def apply_volume_discount(self, base_price: float, quantity: int, 
                            customer_tier: str = "standard") -> Dict[str, float]:
        """🟢 GREEN: Apply volume discount to price"""
        discount_rate = self.calculate_volume_discount(quantity, customer_tier)
        discount_amount = base_price * discount_rate
        final_price = base_price - discount_amount
        savings = discount_amount
        
        return {
            "discount_rate": discount_rate,
            "discount_amount": round(discount_amount, 2),
            "final_price": round(final_price, 2),
            "savings": round(savings, 2)
        }


class RegionalPricingEngine:
    """🟢 GREEN: Regional pricing adjustments"""
    
    def __init__(self):
        # Regional pricing factors
        self.regional_factors = {
            "chile_central": 1.0,    # Santiago - base pricing
            "chile_north": 1.15,     # Mining regions - 15% premium
            "chile_south": 1.08,     # Logistics premium - 8%
            "international": 1.25    # International - 25% premium
        }
        
        # Transport costs (USD per km)
        self.transport_costs = {
            "chile_central": 0.0,     # Base location
            "chile_north": 0.15,      # Transport to north
            "chile_south": 0.12,      # Transport to south
            "international": 15.0     # International shipping base
        }
        
        # Tax rates by region
        self.tax_rates = {
            "chile_central": 0.19,    # Chilean IVA
            "chile_north": 0.19,
            "chile_south": 0.19,
            "international": 0.0      # Export - no local tax
        }
    
    def get_regional_factor(self, region: str) -> float:
        """🟢 GREEN: Get regional pricing factor"""
        if region not in self.regional_factors:
            raise ValueError(f"Invalid region: {region}")
        
        return self.regional_factors[region]
    
    def apply_regional_pricing(self, base_price: float, region: str) -> Dict[str, float]:
        """🟢 GREEN: Apply regional pricing adjustments"""
        if region not in self.regional_factors:
            raise ValueError(f"Invalid region: {region}")
        
        regional_factor = self.regional_factors[region]
        transport_cost = self.transport_costs[region]
        
        adjusted_price = base_price * regional_factor
        final_price = adjusted_price + transport_cost
        
        return {
            "regional_factor": regional_factor,
            "transport_cost": transport_cost,
            "adjusted_price": round(final_price, 2)
        }


class MarginOptimizer:
    """🟢 GREEN: Margin optimization rules"""
    
    def __init__(self):
        # Target margins by customer segment
        self.target_margins = {
            "mining": 0.45,      # 45% margin - premium segment
            "industrial": 0.35,  # 35% margin - standard
            "utility": 0.30,     # 30% margin - competitive
            "residential": 0.25  # 25% margin - volume play
        }
        
        # Market condition adjustments
        self.market_adjustments = {
            "high_demand": 0.05,     # +5% when demand is high
            "low_volatility": 0.02,  # +2% when LME is stable
            "competitive": -0.03     # -3% in competitive situations
        }
    
    def get_target_margin(self, segment: str) -> float:
        """🟢 GREEN: Get target margin for segment"""
        return self.target_margins.get(segment.lower(), 0.25)
    
    def optimize_margin(self, cost_price: float, segment: str, 
                       market_demand: str = "normal", lme_volatility: str = "normal",
                       competitor_price: Optional[float] = None) -> Dict[str, float]:
        """🟢 GREEN: Optimize margin based on conditions"""
        target_margin = self.get_target_margin(segment)
        
        # Calculate base optimized price
        base_optimized_price = cost_price / (1 - target_margin)
        
        # Apply market adjustments
        market_adjustment = 0.0
        if market_demand == "high" and lme_volatility == "low":
            market_adjustment = self.market_adjustments["high_demand"] + \
                              self.market_adjustments["low_volatility"]
        
        adjusted_margin = target_margin + market_adjustment
        optimized_price = cost_price / (1 - adjusted_margin)
        
        # Check against competitor pricing
        if competitor_price:
            max_price = competitor_price * 1.05  # Max 5% over competitor
            optimized_price = min(optimized_price, max_price)
        
        return {
            "target_margin": target_margin,
            "market_adjustment": market_adjustment,
            "optimized_price": round(optimized_price, 2)
        }


class PriorityOrderProcessor:
    """🟢 GREEN: Priority order processing rules"""
    
    def __init__(self):
        # Urgency multipliers
        self.urgency_multipliers = {
            "standard": 1.0,   # No premium
            "urgent": 1.20,    # 20% premium
            "express": 1.35    # 35% premium
        }
        
        # Priority levels for queue management
        self.priority_levels = {
            "express": 1,      # Highest priority
            "urgent": 2,       # High priority  
            "standard": 3      # Normal priority
        }
    
    def get_urgency_multiplier(self, urgency: str) -> float:
        """🟢 GREEN: Get urgency multiplier"""
        return self.urgency_multipliers.get(urgency.lower(), 1.0)
    
    def apply_priority_pricing(self, base_price: float, urgency: str) -> Dict[str, float]:
        """🟢 GREEN: Apply priority pricing"""
        urgency_multiplier = self.get_urgency_multiplier(urgency)
        adjusted_price = base_price * urgency_multiplier
        
        return {
            "urgency": urgency,
            "urgency_multiplier": urgency_multiplier,
            "adjusted_price": round(adjusted_price, 2)
        }


class CustomerTierValidator:
    """🟢 GREEN: Customer tier validation"""
    
    def __init__(self):
        # Tier requirements
        self.tier_requirements = {
            "enterprise": {
                "min_annual_revenue": 1000000,  # $1M+ annual revenue
                "min_relationship_years": 5,    # 5+ years relationship
                "segments": ["mining", "industrial", "utility"]
            },
            "government": {
                "customer_type": "government",
                "required_certifications": ["government_approved"],
                "segments": ["utility", "infrastructure"]
            },
            "standard": {
                # No specific requirements - default tier
            }
        }
        
        # Tier benefits
        self.tier_benefits = {
            "enterprise": {
                "volume_discount_bonus": 0.01,  # +1% volume discount
                "priority_support": True,
                "dedicated_account_manager": True
            },
            "government": {
                "volume_discount_bonus": 0.02,  # +2% volume discount
                "extended_payment_terms": True,
                "compliance_support": True
            },
            "standard": {
                "volume_discount_bonus": 0.0,   # No bonus
                "priority_support": False
            }
        }
    
    def validate_tier(self, customer, tier: str) -> bool:
        """🟢 GREEN: Validate customer tier eligibility"""
        if tier == "standard":
            return True  # Anyone can be standard tier
        
        requirements = self.tier_requirements.get(tier, {})
        
        if tier == "enterprise":
            annual_revenue = getattr(customer, 'annual_revenue', 0)
            relationship_years = getattr(customer, 'relationship_years', 0)
            segment = getattr(customer, 'segment', '')
            
            return (annual_revenue >= requirements.get("min_annual_revenue", 0) and
                   relationship_years >= requirements.get("min_relationship_years", 0) and
                   segment in requirements.get("segments", []))
        
        elif tier == "government":
            customer_type = getattr(customer, 'customer_type', '')
            certifications = getattr(customer, 'certifications', [])
            
            return (customer_type == requirements.get("customer_type") and
                   any(cert in certifications for cert in requirements.get("required_certifications", [])))
        
        return False
    
    def get_tier_benefits(self, tier: str) -> Dict[str, Any]:
        """🟢 GREEN: Get benefits for customer tier"""
        return self.tier_benefits.get(tier, self.tier_benefits["standard"])


# Export classes for testing
__all__ = [
    "BusinessRulesEngine",
    "CustomerSegmentationError", 
    "VolumeDiscountCalculator",
    "RegionalPricingEngine",
    "MarginOptimizer",
    "PriorityOrderProcessor",
    "CustomerTierValidator"
]