"""
🔴 RED PHASE - Tests para ML Pricing Model - DEBEN FALLAR PRIMERO
Sprint 2.1.1: ML Model training con data extraída de PDFs Nexans + LME real-time
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from decimal import Decimal


def test_pricing_model_initialization():
    """🔴 RED: Test inicialización del modelo XGBoost - DEBE FALLAR"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'is_trained')
    assert not model.is_trained  # Initially not trained


def test_pricing_model_feature_engineering():
    """🔴 RED: Test feature engineering con data real extraída"""
    from src.pricing.ml_model import PricingModel
    from src.models.cable import CableProduct
    from src.models.market import LMEPriceData
    
    model = PricingModel()
    
    # Data realista basada en PDFs Nexans extraídos
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
    
    lme_data = LMEPriceData(
        metal="copper",
        price_usd_per_ton=9500.0,
        timestamp=datetime.now(),
        exchange="LME",
        currency="USD"
    )
    
    features = model.engineer_features(
        cable=cable,
        copper_price=lme_data.price_usd_per_ton,
        aluminum_price=2650.0,
        customer_segment="mining",
        order_quantity=1000,
        delivery_urgency="standard"
    )
    
    # Features expected based on Phase 1 analysis
    expected_features = [
        "lme_copper_price", "lme_aluminum_price", "copper_content_kg",
        "aluminum_content_kg", "voltage_rating", "current_rating", 
        "cable_complexity", "customer_segment", "order_quantity", "delivery_urgency"
    ]
    
    assert isinstance(features, np.ndarray)
    assert len(features) == len(expected_features)
    assert features[0] == 9500.0  # copper price
    assert features[2] == 2.3     # copper content


def test_pricing_model_training_with_synthetic_data():
    """🔴 RED: Test training con synthetic data basada en PDFs reales"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    # Generate realistic training data based on extracted PDF specs
    n_samples = 100
    X_train = np.random.rand(n_samples, 10)  # 10 features engineered
    
    # Target prices based on realistic cable pricing (USD/meter)
    # Mining cables: $15-50/meter based on specifications
    y_train = np.random.uniform(15.0, 50.0, n_samples)
    
    model.train(X_train, y_train)
    
    assert model.is_trained
    assert hasattr(model.model, 'predict')


def test_pricing_model_prediction():
    """🔴 RED: Test predicción de precios con modelo entrenado"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    # Mock training data
    X_train = np.random.rand(50, 10)
    y_train = np.random.uniform(15.0, 50.0, 50)
    model.train(X_train, y_train)
    
    # Test prediction with realistic features
    test_features = np.array([
        9500.0,  # copper_price
        2650.0,  # aluminum_price
        2.3,     # copper_content_kg (from PDF)
        0.0,     # aluminum_content_kg
        5000,    # voltage_rating (from PDF)
        122,     # current_rating (from PDF)
        1.25,    # complexity_multiplier
        2.0,     # segment_multiplier (mining)
        1000,    # order_quantity
        1.0      # urgency_multiplier
    ]).reshape(1, -1)
    
    predicted_price = model.predict(test_features)
    
    assert isinstance(predicted_price, (float, np.ndarray))
    assert 10.0 < predicted_price < 100.0  # Realistic cable price range


def test_pricing_model_validation_metrics():
    """🔴 RED: Test métricas de validación del modelo"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    # Training data
    X_train = np.random.rand(100, 10)
    y_train = np.random.uniform(15.0, 50.0, 100)
    
    # Validation data
    X_val = np.random.rand(20, 10)
    y_val = np.random.uniform(15.0, 50.0, 20)
    
    model.train(X_train, y_train)
    metrics = model.validate(X_val, y_val)
    
    assert 'mae' in metrics  # Mean Absolute Error
    assert 'rmse' in metrics  # Root Mean Square Error
    assert 'r2' in metrics   # R-squared
    
    # Target: MAE < 5% según success metrics
    assert metrics['mae'] < 2.5  # 5% of average price ~50USD


def test_cable_price_calculation_integration():
    """🔴 RED: Test integración con CableProduct.get_total_material_cost()"""
    from src.pricing.ml_model import calculate_cable_base_cost
    from src.models.cable import CableProduct
    
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
    
    # LME prices reales
    copper_price_per_ton = 9500.0
    aluminum_price_per_ton = 2650.0
    
    base_cost = calculate_cable_base_cost(
        cable=cable,
        copper_price_usd_per_ton=copper_price_per_ton,
        aluminum_price_usd_per_ton=aluminum_price_per_ton
    )
    
    # Expected: 2.3kg copper * $9.5/kg = $21.85 + manufacturing
    expected_copper_cost = cable.copper_content_kg * (copper_price_per_ton / 1000)
    
    assert base_cost > expected_copper_cost
    assert isinstance(base_cost, (float, Decimal))


def test_model_persistence():
    """🔴 RED: Test guardar y cargar modelo entrenado"""
    from src.pricing.ml_model import PricingModel
    import tempfile
    import os
    
    model = PricingModel()
    
    # Train model
    X_train = np.random.rand(50, 10)
    y_train = np.random.uniform(15.0, 50.0, 50)
    model.train(X_train, y_train)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        model.save_model(tmp.name)
        
        # Load new model instance
        new_model = PricingModel()
        new_model.load_model(tmp.name)
        
        assert new_model.is_trained
        
        # Cleanup
        os.unlink(tmp.name)


def test_feature_importance_analysis():
    """🔴 RED: Test análisis de importancia de features"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    X_train = np.random.rand(100, 10)
    y_train = np.random.uniform(15.0, 50.0, 100)
    model.train(X_train, y_train)
    
    feature_importance = model.get_feature_importance()
    
    assert len(feature_importance) == 10  # 10 features
    assert all(importance >= 0 for importance in feature_importance)
    assert abs(sum(feature_importance) - 1.0) < 0.01  # Should sum to ~1.0


def test_pricing_with_market_volatility():
    """🔴 RED: Test pricing considerando volatilidad del mercado LME"""
    from src.pricing.ml_model import PricingModel
    
    model = PricingModel()
    
    # Train with market conditions
    X_train = np.random.rand(100, 10)
    y_train = np.random.uniform(15.0, 50.0, 100)
    model.train(X_train, y_train)
    
    # Test with different market scenarios
    base_features = np.array([9500, 2650, 2.3, 0.0, 5000, 122, 1.25, 2.0, 1000, 1.0])
    
    # High volatility scenario
    volatile_features = base_features.copy()
    volatile_features[0] = 11000  # Copper spike +15%
    
    base_price = model.predict(base_features.reshape(1, -1))
    volatile_price = model.predict(volatile_features.reshape(1, -1))
    
    # Price should increase with copper price
    assert volatile_price > base_price


@pytest.fixture
def sample_nexans_data():
    """🔴 RED: Fixture con data extraída de PDFs Nexans reales"""
    return {
        "cables": [
            {
                "reference": "540317340",
                "voltage": 5000,
                "current": 122,
                "copper_kg": 2.3,
                "application": "mining"
            },
            # More realistic samples based on PDF extraction
        ],
        "lme_prices": {
            "copper": 9500.0,
            "aluminum": 2650.0
        }
    }


def test_end_to_end_pricing_workflow(sample_nexans_data):
    """🔴 RED: Test workflow completo de pricing"""
    from src.pricing.ml_model import PricingModel, end_to_end_price_calculation
    
    model = PricingModel()
    
    # Train with sample data
    X_train = np.random.rand(50, 10)
    y_train = np.random.uniform(15.0, 50.0, 50)
    model.train(X_train, y_train)
    
    cable_data = sample_nexans_data["cables"][0]
    lme_prices = sample_nexans_data["lme_prices"]
    
    final_price = end_to_end_price_calculation(
        model=model,
        cable_reference=cable_data["reference"],
        customer_segment="mining",
        order_quantity=1000,
        delivery_urgency="standard",
        copper_price=lme_prices["copper"],
        aluminum_price=lme_prices["aluminum"]
    )
    
    assert isinstance(final_price, float)
    assert 15.0 < final_price < 100.0  # Realistic range for mining cables