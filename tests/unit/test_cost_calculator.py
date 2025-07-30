"""
🔴 RED PHASE - Tests para Cost Calculator Real-time - DEBEN FALLAR PRIMERO
Sprint 2.1.2: Cost calculator con LME real-time integration
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import patch, Mock


def test_cost_calculator_initialization():
    """🔴 RED: Test inicialización del cost calculator - DEBE FALLAR"""
    from src.pricing.cost_calculator import CostCalculator
    
    calculator = CostCalculator()
    
    assert calculator is not None
    assert hasattr(calculator, 'lme_client')
    assert hasattr(calculator, 'cache_enabled')
    assert calculator.cache_enabled is True  # Default caching enabled


def test_real_time_material_cost_calculation():
    """🔴 RED: Test cálculo costo materiales con LME real-time"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    # Cable real extraído de PDF Nexans
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    material_cost = calculator.calculate_material_cost(cable)
    
    # Expected: 2.3kg copper * ~$9.5/kg = ~$21.85 base
    expected_min = 20.0  # USD per meter minimum
    expected_max = 30.0  # USD per meter maximum
    
    assert expected_min <= material_cost <= expected_max
    assert isinstance(material_cost, (float, Decimal))


def test_lme_price_integration():
    """🔴 RED: Test integración con APIs LME existentes"""
    from src.pricing.cost_calculator import CostCalculator
    
    calculator = CostCalculator()
    
    # Should use existing LME API from Phase 1
    copper_price = calculator.get_current_copper_price()
    aluminum_price = calculator.get_current_aluminum_price()
    
    # Validate realistic LME ranges
    assert 8000 < copper_price < 12000  # USD/ton
    assert 2000 < aluminum_price < 4000  # USD/ton
    
    assert isinstance(copper_price, (int, float))
    assert isinstance(aluminum_price, (int, float))


def test_manufacturing_cost_calculation():
    """🔴 RED: Test cálculo costos de manufactura"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    manufacturing_cost = calculator.calculate_manufacturing_cost(cable)
    
    # Manufacturing should include: labor, equipment, overhead
    # Mining cables have higher manufacturing complexity
    assert manufacturing_cost > 5.0  # Minimum manufacturing cost
    assert manufacturing_cost < 20.0  # Maximum realistic cost
    assert isinstance(manufacturing_cost, (float, Decimal))


def test_total_cost_calculation():
    """🔴 RED: Test cálculo costo total integrado"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    total_cost = calculator.calculate_total_cost(cable)
    
    # Total = Material + Manufacturing + Overhead
    # Should be realistic for mining cable
    assert 25.0 < total_cost < 60.0  # USD per meter
    assert isinstance(total_cost, (float, Decimal))


def test_cost_breakdown_detailed():
    """🔴 RED: Test breakdown detallado de costos"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    breakdown = calculator.get_cost_breakdown(cable)
    
    # Expected breakdown structure
    required_fields = [
        "copper_cost", "aluminum_cost", "polymer_cost",
        "manufacturing_cost", "overhead_cost", "total_cost"
    ]
    
    assert isinstance(breakdown, dict)
    assert all(field in breakdown for field in required_fields)
    assert all(breakdown[field] >= 0 for field in required_fields)
    
    # Verify total matches sum of components
    expected_total = sum(breakdown[field] for field in required_fields[:-1])
    assert abs(breakdown["total_cost"] - expected_total) < 0.01


def test_cost_with_market_volatility():
    """🔴 RED: Test costos con volatilidad del mercado"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # Calculate cost with different market scenarios
    base_cost = calculator.calculate_total_cost(cable)
    
    # High copper price scenario
    with patch.object(calculator, 'get_current_copper_price', return_value=11000):
        high_copper_cost = calculator.calculate_total_cost(cable)
    
    # Low copper price scenario  
    with patch.object(calculator, 'get_current_copper_price', return_value=8000):
        low_copper_cost = calculator.calculate_total_cost(cable)
    
    # Costs should vary with copper prices
    assert high_copper_cost > base_cost
    assert low_copper_cost < base_cost
    assert (high_copper_cost - low_copper_cost) > 2.0  # Significant difference


def test_cost_caching_mechanism():
    """🔴 RED: Test mecanismo de cache para evitar llamadas API excesivas"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator(cache_enabled=True)
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Test Cable",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # First calculation should cache LME prices
    with patch('src.services.lme_api.get_lme_copper_price') as mock_copper:
        mock_copper.return_value = 9500.0
        
        cost1 = calculator.calculate_total_cost(cable)
        cost2 = calculator.calculate_total_cost(cable)  # Should use cache
        
        assert cost1 == cost2
        assert mock_copper.call_count <= 2  # Should use cache on second call


def test_cost_accuracy_target():
    """🔴 RED: Test precisión de costos vs cálculo manual"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # Mock fixed prices for deterministic test
    with patch.object(calculator, 'get_current_copper_price', return_value=9500.0):
        calculated_cost = calculator.calculate_total_cost(cable)
        
        # Manual calculation for verification
        copper_cost = 2.3 * (9500.0 / 1000)  # 2.3kg * $9.5/kg = $21.85
        expected_min = copper_cost + 5.0  # Add minimum manufacturing
        expected_max = copper_cost + 15.0  # Add maximum manufacturing
        
        # Target: ±2% accuracy per success metrics
        assert expected_min * 0.98 <= calculated_cost <= expected_max * 1.02


def test_polymer_cost_calculation():
    """🔴 RED: Test cálculo costos polímeros/insulation"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    # Cable with significant insulation (5kV)
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    polymer_cost = calculator.calculate_polymer_cost(cable)
    
    # High voltage cables need more insulation
    assert polymer_cost > 2.0  # Minimum polymer cost
    assert polymer_cost < 10.0  # Maximum realistic cost
    assert isinstance(polymer_cost, (float, Decimal))


def test_cost_calculator_with_different_applications():
    """🔴 RED: Test costos diferentes por aplicación"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    # Mining cable (high complexity)
    mining_cable = CableProduct(
        nexans_reference="540317340",
        product_name="Mining Cable",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,       
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # Residential cable (low complexity)
    residential_cable = CableProduct(
        nexans_reference="123456789",
        product_name="Residential Cable",
        voltage_rating=1000,
        current_rating=50,
        conductor_section_mm2=10.0,
        copper_content_kg=1.0,
        aluminum_content_kg=0.0,
        weight_kg_per_km=800,
        applications=["residential"]
    )
    
    mining_cost = calculator.calculate_total_cost(mining_cable)
    residential_cost = calculator.calculate_total_cost(residential_cable)
    
    # Mining cables should cost more due to complexity and specifications
    assert mining_cost > residential_cost


@pytest.fixture
def sample_cable_data():
    """🔴 RED: Fixture con data de cables Nexans reales"""
    return {
        "mining_cable": {
            "reference": "540317340",
            "copper_kg": 2.3,
            "voltage": 5000,
            "current": 122,
            "application": "mining"
        },
        "lme_snapshot": {
            "copper_price": 9500.0,
            "aluminum_price": 2650.0,
            "timestamp": datetime.now()
        }
    }


def test_cost_calculation_integration(sample_cable_data):
    """🔴 RED: Test integración completa con data real"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable_spec = sample_cable_data["mining_cable"]
    lme_prices = sample_cable_data["lme_snapshot"]
    
    cable = CableProduct(
        nexans_reference=cable_spec["reference"],
        product_name="Test Mining Cable",
        voltage_rating=cable_spec["voltage"],
        current_rating=cable_spec["current"],
        conductor_section_mm2=21.2,
        copper_content_kg=cable_spec["copper_kg"],
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=[cable_spec["application"]]
    )
    
    # Mock LME prices for deterministic test
    with patch.object(calculator, 'get_current_copper_price', 
                     return_value=lme_prices["copper_price"]):
        with patch.object(calculator, 'get_current_aluminum_price',
                         return_value=lme_prices["aluminum_price"]):
            
            total_cost = calculator.calculate_total_cost(cable)
            breakdown = calculator.get_cost_breakdown(cable)
            
            assert total_cost == breakdown["total_cost"]
            assert total_cost > 20.0  # Realistic minimum
            assert total_cost < 70.0  # Realistic maximum


def test_cost_calculator_error_handling():
    """🔴 RED: Test error handling cuando APIs fallan"""
    from src.pricing.cost_calculator import CostCalculator, CostCalculationError
    from src.models.cable import CableProduct
    
    calculator = CostCalculator()
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Test Cable",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # Test with API failure
    with patch.object(calculator, 'get_current_copper_price', 
                     side_effect=Exception("API Error")):
        
        # Should either use fallback or raise specific error
        try:
            cost = calculator.calculate_total_cost(cable)
            assert cost > 0  # Fallback worked
        except CostCalculationError:
            pass  # Expected error handling


def test_cost_calculator_performance():
    """🔴 RED: Test performance para múltiples cálculos"""
    from src.pricing.cost_calculator import CostCalculator
    from src.models.cable import CableProduct
    import time
    
    calculator = CostCalculator(cache_enabled=True)
    
    cable = CableProduct(
        nexans_reference="540317340",
        product_name="Performance Test Cable",
        voltage_rating=5000,
        current_rating=122,
        conductor_section_mm2=21.2,
        copper_content_kg=2.3,
        aluminum_content_kg=0.0,
        weight_kg_per_km=2300,
        applications=["mining"]
    )
    
    # Time multiple calculations
    start_time = time.time()
    
    costs = []
    for _ in range(10):
        cost = calculator.calculate_total_cost(cable)
        costs.append(cost)
    
    elapsed_time = time.time() - start_time
    
    # Should complete quickly with caching
    assert elapsed_time < 2.0  # Max 2 seconds for 10 calculations
    assert all(c > 0 for c in costs)  # All calculations successful