"""
🔴 RED PHASE - Business Rules Tests
Sprint 2.2.1: Business rules por segmento cliente

TESTS TO WRITE FIRST (RED):
- Customer segmentation logic with multipliers
- Volume discount calculations
- Regional pricing adjustments
- Margin optimization rules
- Priority order processing
- Customer tier validation

All tests MUST FAIL initially to follow TDD methodology.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import will fail initially - that's expected in RED phase
from src.pricing.business_rules import (
    BusinessRulesEngine,
    CustomerSegmentationError,
    VolumeDiscountCalculator, 
    RegionalPricingEngine,
    MarginOptimizer,
    PriorityOrderProcessor,
    CustomerTierValidator
)


class TestBusinessRulesEngine:
    """🔴 RED: Test Business Rules Engine functionality"""
    
    def test_business_rules_engine_initialization(self):
        """🔴 RED: Test BusinessRulesEngine can be instantiated with config"""
        # EXPECT: BusinessRulesEngine class doesn't exist yet
        engine = BusinessRulesEngine()
        assert engine is not None
        assert hasattr(engine, 'customer_segments')
        assert hasattr(engine, 'volume_calculator')
        assert hasattr(engine, 'regional_engine')
        assert hasattr(engine, 'margin_optimizer')
    
    def test_customer_segment_mining_multiplier(self):
        """🔴 RED: Test mining segment gets 1.5x multiplier"""
        engine = BusinessRulesEngine()
        
        # Mining segment should have highest multiplier
        multiplier = engine.get_customer_segment_multiplier("mining")
        assert multiplier == 1.5
        
        # Test with customer object
        customer = Mock()
        customer.segment = "mining"
        customer.tier = "enterprise"
        
        result = engine.apply_segment_rules(customer, base_price=100.0)
        assert result["multiplier"] == 1.5
        assert result["adjusted_price"] == 150.0
    
    def test_customer_segment_industrial_multiplier(self):
        """🔴 RED: Test industrial segment gets 1.3x multiplier"""
        engine = BusinessRulesEngine()
        
        multiplier = engine.get_customer_segment_multiplier("industrial")
        assert multiplier == 1.3
        
        customer = Mock()
        customer.segment = "industrial"
        customer.tier = "standard"
        
        result = engine.apply_segment_rules(customer, base_price=100.0)
        assert result["multiplier"] == 1.3
        assert result["adjusted_price"] == 130.0
    
    def test_customer_segment_utility_multiplier(self):
        """🔴 RED: Test utility segment gets 1.2x multiplier"""
        engine = BusinessRulesEngine()
        
        multiplier = engine.get_customer_segment_multiplier("utility")
        assert multiplier == 1.2
        
        customer = Mock()
        customer.segment = "utility"
        customer.tier = "government"
        
        result = engine.apply_segment_rules(customer, base_price=100.0)
        assert result["multiplier"] == 1.2
        assert result["adjusted_price"] == 120.0
    
    def test_customer_segment_residential_multiplier(self):
        """🔴 RED: Test residential segment gets 1.0x multiplier (base)"""
        engine = BusinessRulesEngine()
        
        multiplier = engine.get_customer_segment_multiplier("residential")
        assert multiplier == 1.0
        
        customer = Mock()
        customer.segment = "residential"
        customer.tier = "retail"
        
        result = engine.apply_segment_rules(customer, base_price=100.0)
        assert result["multiplier"] == 1.0
        assert result["adjusted_price"] == 100.0
    
    def test_invalid_customer_segment_raises_error(self):
        """🔴 RED: Test invalid segment raises CustomerSegmentationError"""
        engine = BusinessRulesEngine()
        
        with pytest.raises(CustomerSegmentationError):
            engine.get_customer_segment_multiplier("invalid_segment")
        
        customer = Mock()
        customer.segment = "unknown"
        
        with pytest.raises(CustomerSegmentationError):
            engine.apply_segment_rules(customer, base_price=100.0)


class TestVolumeDiscountCalculator:
    """🔴 RED: Test volume discount calculations"""
    
    def test_volume_discount_calculator_initialization(self):
        """🔴 RED: Test VolumeDiscountCalculator initialization"""
        calculator = VolumeDiscountCalculator()
        assert calculator is not None
        assert hasattr(calculator, 'tier_thresholds')
        assert hasattr(calculator, 'discount_rates')
    
    def test_volume_discount_tier_1_small_orders(self):
        """🔴 RED: Test small orders (1-100m) get no discount"""
        calculator = VolumeDiscountCalculator()
        
        discount = calculator.calculate_volume_discount(50)  # 50 meters
        assert discount == 0.0
        
        discount = calculator.calculate_volume_discount(100)  # 100 meters
        assert discount == 0.0
    
    def test_volume_discount_tier_2_medium_orders(self):
        """🔴 RED: Test medium orders (101-500m) get 3% discount"""
        calculator = VolumeDiscountCalculator()
        
        discount = calculator.calculate_volume_discount(250)  # 250 meters
        assert discount == 0.03
        
        discount = calculator.calculate_volume_discount(500)  # 500 meters
        assert discount == 0.03
    
    def test_volume_discount_tier_3_large_orders(self):
        """🔴 RED: Test large orders (501-1000m) get 5% discount"""
        calculator = VolumeDiscountCalculator()
        
        discount = calculator.calculate_volume_discount(750)  # 750 meters
        assert discount == 0.05
        
        discount = calculator.calculate_volume_discount(1000)  # 1000 meters
        assert discount == 0.05
    
    def test_volume_discount_tier_4_enterprise_orders(self):
        """🔴 RED: Test enterprise orders (1001-5000m) get 8% discount"""
        calculator = VolumeDiscountCalculator()
        
        discount = calculator.calculate_volume_discount(2500)  # 2500 meters
        assert discount == 0.08
        
        discount = calculator.calculate_volume_discount(5000)  # 5000 meters
        assert discount == 0.08
    
    def test_volume_discount_tier_5_mega_orders(self):
        """🔴 RED: Test mega orders (5000m+) get 12% discount"""
        calculator = VolumeDiscountCalculator()
        
        discount = calculator.calculate_volume_discount(10000)  # 10000 meters
        assert discount == 0.12
        
        discount = calculator.calculate_volume_discount(50000)  # 50000 meters
        assert discount == 0.12
    
    def test_volume_discount_with_customer_tier(self):
        """🔴 RED: Test volume discount considers customer tier"""
        calculator = VolumeDiscountCalculator()
        
        # Enterprise customer gets additional 1% discount
        discount = calculator.calculate_volume_discount(
            quantity=1500, 
            customer_tier="enterprise"
        )
        assert discount == 0.09  # 8% + 1% tier bonus
        
        # Government customer gets additional 2% discount
        discount = calculator.calculate_volume_discount(
            quantity=1500, 
            customer_tier="government"
        )
        assert discount == 0.10  # 8% + 2% tier bonus
        
        # Standard customer gets no tier bonus
        discount = calculator.calculate_volume_discount(
            quantity=1500, 
            customer_tier="standard"
        )
        assert discount == 0.08  # 8% only
    
    def test_volume_discount_applies_to_price(self):
        """🔴 RED: Test volume discount application to price"""
        calculator = VolumeDiscountCalculator()
        
        # Test 5% discount on $100 base price
        result = calculator.apply_volume_discount(
            base_price=100.0,
            quantity=750
        )
        
        assert result["discount_rate"] == 0.05
        assert result["discount_amount"] == 5.0
        assert result["final_price"] == 95.0
        assert result["savings"] == 5.0


class TestRegionalPricingEngine:
    """🔴 RED: Test regional pricing adjustments"""
    
    def test_regional_pricing_engine_initialization(self):
        """🔴 RED: Test RegionalPricingEngine initialization"""
        engine = RegionalPricingEngine()
        assert engine is not None
        assert hasattr(engine, 'regional_factors')
        assert hasattr(engine, 'transport_costs')
        assert hasattr(engine, 'tax_rates')
    
    def test_regional_pricing_chile_central(self):
        """🔴 RED: Test Chile Central region (Santiago) - base pricing"""
        engine = RegionalPricingEngine()
        
        factor = engine.get_regional_factor("chile_central")
        assert factor == 1.0  # Base region, no adjustment
        
        result = engine.apply_regional_pricing(
            base_price=100.0,
            region="chile_central"
        )
        assert result["regional_factor"] == 1.0
        assert result["adjusted_price"] == 100.0
        assert result["transport_cost"] == 0.0
    
    def test_regional_pricing_chile_north(self):
        """🔴 RED: Test Chile North region - mining premium"""
        engine = RegionalPricingEngine()
        
        factor = engine.get_regional_factor("chile_north")
        assert factor == 1.15  # 15% premium for mining regions
        
        result = engine.apply_regional_pricing(
            base_price=100.0,
            region="chile_north"
        )
        assert result["regional_factor"] == 1.15
        assert result["adjusted_price"] == 115.0
        assert result["transport_cost"] > 0  # Transport costs to north
    
    def test_regional_pricing_chile_south(self):
        """🔴 RED: Test Chile South region - logistics premium"""
        engine = RegionalPricingEngine()
        
        factor = engine.get_regional_factor("chile_south")
        assert factor == 1.08  # 8% premium for logistics
        
        result = engine.apply_regional_pricing(
            base_price=100.0,
            region="chile_south"
        )
        assert result["regional_factor"] == 1.08
        assert result["adjusted_price"] == 108.0
        assert result["transport_cost"] > 0  # Transport costs to south
    
    def test_regional_pricing_international(self):
        """🔴 RED: Test international shipping premium"""
        engine = RegionalPricingEngine()
        
        factor = engine.get_regional_factor("international")
        assert factor == 1.25  # 25% premium for international
        
        result = engine.apply_regional_pricing(
            base_price=100.0,
            region="international"
        )
        assert result["regional_factor"] == 1.25
        assert result["adjusted_price"] == 125.0
        assert result["transport_cost"] > 10.0  # High international shipping
    
    def test_invalid_region_raises_error(self):
        """🔴 RED: Test invalid region raises error"""
        engine = RegionalPricingEngine()
        
        with pytest.raises(ValueError):
            engine.get_regional_factor("invalid_region")
        
        with pytest.raises(ValueError):
            engine.apply_regional_pricing(base_price=100.0, region="unknown")


class TestMarginOptimizer:
    """🔴 RED: Test margin optimization rules"""
    
    def test_margin_optimizer_initialization(self):
        """🔴 RED: Test MarginOptimizer initialization"""
        optimizer = MarginOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'target_margins')
        assert hasattr(optimizer, 'cost_calculator')
    
    def test_margin_optimization_mining_segment(self):
        """🔴 RED: Test mining segment targets 45% margin"""
        optimizer = MarginOptimizer()
        
        target_margin = optimizer.get_target_margin("mining")
        assert target_margin == 0.45  # 45% margin
        
        # Test margin calculation
        result = optimizer.optimize_margin(
            cost_price=100.0,
            segment="mining"
        )
        assert result["target_margin"] == 0.45
        assert result["optimized_price"] == 181.82  # 100 / (1 - 0.45)
    
    def test_margin_optimization_industrial_segment(self):
        """🔴 RED: Test industrial segment targets 35% margin"""
        optimizer = MarginOptimizer()
        
        target_margin = optimizer.get_target_margin("industrial")
        assert target_margin == 0.35  # 35% margin
        
        result = optimizer.optimize_margin(
            cost_price=100.0,
            segment="industrial"
        )
        assert result["target_margin"] == 0.35
        assert result["optimized_price"] == 153.85  # 100 / (1 - 0.35)
    
    def test_margin_optimization_with_market_conditions(self):
        """🔴 RED: Test margin optimization considers market conditions"""
        optimizer = MarginOptimizer()
        
        # High demand allows higher margins
        result = optimizer.optimize_margin(
            cost_price=100.0,
            segment="mining",
            market_demand="high",
            lme_volatility="low"
        )
        # Should get margin boost for favorable conditions
        assert result["optimized_price"] > 181.82
        assert result["market_adjustment"] > 0
    
    def test_margin_optimization_with_competitor_pricing(self):
        """🔴 RED: Test margin optimization considers competitor prices"""
        optimizer = MarginOptimizer()
        
        result = optimizer.optimize_margin(
            cost_price=100.0,
            segment="industrial",
            competitor_price=140.0
        )
        # Should not exceed competitor price significantly
        assert result["optimized_price"] <= 140.0 * 1.05  # Max 5% over competitor


class TestPriorityOrderProcessor:
    """🔴 RED: Test priority order processing rules"""
    
    def test_priority_order_processor_initialization(self):
        """🔴 RED: Test PriorityOrderProcessor initialization"""
        processor = PriorityOrderProcessor()
        assert processor is not None
        assert hasattr(processor, 'priority_levels')
        assert hasattr(processor, 'urgency_multipliers')
    
    def test_urgent_delivery_premium(self):
        """🔴 RED: Test urgent delivery gets 20% premium"""
        processor = PriorityOrderProcessor()
        
        multiplier = processor.get_urgency_multiplier("urgent")
        assert multiplier == 1.20  # 20% premium
        
        result = processor.apply_priority_pricing(
            base_price=100.0,
            urgency="urgent"
        )
        assert result["urgency_multiplier"] == 1.20
        assert result["adjusted_price"] == 120.0
    
    def test_express_delivery_premium(self):
        """🔴 RED: Test express delivery gets 35% premium"""
        processor = PriorityOrderProcessor()
        
        multiplier = processor.get_urgency_multiplier("express")
        assert multiplier == 1.35  # 35% premium
        
        result = processor.apply_priority_pricing(
            base_price=100.0,
            urgency="express"
        )
        assert result["urgency_multiplier"] == 1.35
        assert result["adjusted_price"] == 135.0
    
    def test_standard_delivery_no_premium(self):
        """🔴 RED: Test standard delivery gets no premium"""
        processor = PriorityOrderProcessor()
        
        multiplier = processor.get_urgency_multiplier("standard")
        assert multiplier == 1.0  # No premium
        
        result = processor.apply_priority_pricing(
            base_price=100.0,
            urgency="standard"
        )
        assert result["urgency_multiplier"] == 1.0
        assert result["adjusted_price"] == 100.0


class TestCustomerTierValidator:
    """🔴 RED: Test customer tier validation"""
    
    def test_customer_tier_validator_initialization(self):
        """🔴 RED: Test CustomerTierValidator initialization"""
        validator = CustomerTierValidator()
        assert validator is not None
        assert hasattr(validator, 'tier_requirements')
        assert hasattr(validator, 'tier_benefits')
    
    def test_enterprise_tier_validation(self):
        """🔴 RED: Test enterprise tier requirements"""
        validator = CustomerTierValidator()
        
        # Enterprise tier: >$1M annual revenue, >5 years relationship
        customer = Mock()
        customer.annual_revenue = 1500000
        customer.relationship_years = 7
        customer.segment = "mining"
        
        is_valid = validator.validate_tier(customer, "enterprise")
        assert is_valid == True
        
        benefits = validator.get_tier_benefits("enterprise")
        assert "volume_discount_bonus" in benefits
        assert benefits["volume_discount_bonus"] == 0.01  # 1% additional
    
    def test_government_tier_validation(self):
        """🔴 RED: Test government tier requirements"""
        validator = CustomerTierValidator()
        
        customer = Mock()
        customer.customer_type = "government"
        customer.segment = "utility"
        customer.certifications = ["iso_9001", "government_approved"]
        
        is_valid = validator.validate_tier(customer, "government")
        assert is_valid == True
        
        benefits = validator.get_tier_benefits("government")
        assert "volume_discount_bonus" in benefits
        assert benefits["volume_discount_bonus"] == 0.02  # 2% additional
    
    def test_standard_tier_validation(self):
        """🔴 RED: Test standard tier is default"""
        validator = CustomerTierValidator()
        
        customer = Mock()
        customer.annual_revenue = 50000
        customer.relationship_years = 1
        
        is_valid = validator.validate_tier(customer, "standard")
        assert is_valid == True
        
        benefits = validator.get_tier_benefits("standard")
        assert benefits["volume_discount_bonus"] == 0.0  # No bonus


class TestIntegratedBusinessRules:
    """🔴 RED: Test integrated business rules workflow"""
    
    def test_complete_pricing_workflow(self):
        """🔴 RED: Test complete business rules workflow"""
        engine = BusinessRulesEngine()
        
        # Mock customer data
        customer = Mock()
        customer.segment = "mining" 
        customer.tier = "enterprise"
        customer.region = "chile_north"
        
        # Mock order data
        order = Mock()
        order.quantity = 1500  # meters
        order.urgency = "urgent"
        order.base_cost = 100.0
        
        # Apply complete business rules
        result = engine.apply_complete_pricing_rules(customer, order)
        
        # Should include all adjustments
        assert "segment_multiplier" in result
        assert "volume_discount" in result  
        assert "regional_factor" in result
        assert "urgency_multiplier" in result
        assert "margin_optimized_price" in result
        assert "final_price" in result
        
        # Verify final price considers all factors
        assert result["final_price"] > order.base_cost
    
    def test_business_rules_edge_cases(self):
        """🔴 RED: Test business rules handle edge cases"""
        engine = BusinessRulesEngine()
        
        # Test with minimal data
        customer = Mock()
        customer.segment = "residential"
        customer.tier = "standard"
        customer.region = "chile_central"
        
        order = Mock()
        order.quantity = 10  # Very small order
        order.urgency = "standard"
        order.base_cost = 50.0
        
        result = engine.apply_complete_pricing_rules(customer, order)
        
        # Should still return valid pricing
        assert result["final_price"] >= order.base_cost
        assert result["volume_discount"] == 0.0  # No discount for small orders
    
    def test_business_rules_validation_errors(self):
        """🔴 RED: Test business rules validation and error handling"""
        engine = BusinessRulesEngine()
        
        # Test with invalid data
        customer = Mock()
        customer.segment = "invalid_segment"
        
        order = Mock()
        order.quantity = -100  # Invalid quantity
        order.base_cost = -50.0  # Invalid cost
        
        with pytest.raises(CustomerSegmentationError):
            engine.apply_complete_pricing_rules(customer, order)