"""
🔴 RED PHASE - Tests para Data Models - DEBEN FALLAR PRIMERO
Tarea 1.1.2: Data Models con TDD - Escribiendo tests PRIMERO según plan
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Optional


def test_cable_product_model():
    """🔴 RED: Test CableProduct model - DEBE FALLAR PRIMERO"""
    from src.models.cable import CableProduct
    
    # Test construcción básica
    cable = CableProduct(
        nexans_ref="540317340",
        product_name="NEXANS Mining Cable 15kV",
        copper_content_kg=100.5,
        aluminum_content_kg=25.2,
        voltage_rating=15000
    )
    
    assert cable.nexans_ref == "540317340"
    assert cable.copper_content_kg > 0
    assert cable.aluminum_content_kg >= 0
    assert cable.voltage_rating == 15000
    assert cable.product_name is not None


def test_cable_product_validation():
    """🔴 RED: Test validaciones del modelo CableProduct"""
    from src.models.cable import CableProduct
    
    # Test validación nexans_ref
    with pytest.raises(ValueError):
        CableProduct(nexans_ref="")  # Empty ref should fail
    
    # Test validación copper content
    with pytest.raises(ValueError):
        CableProduct(
            nexans_ref="540317340",
            copper_content_kg=-10  # Negative should fail
        )


def test_pricing_request_model():
    """🔴 RED: Test PricingRequest model - DEBE FALLAR"""
    from src.models.pricing import PricingRequest
    
    request = PricingRequest(
        cable_ref="540317340",
        quantity=500,
        customer_segment="mining",
        urgency_level="high",
        delivery_location="Santiago"
    )
    
    assert request.cable_ref == "540317340"
    assert request.quantity == 500
    assert request.is_valid() == True
    assert request.get_urgency_multiplier() > 1.0  # High urgency = higher price


def test_pricing_response_model():
    """🔴 RED: Test PricingResponse model con ML output"""
    from src.models.pricing import PricingResponse
    
    response = PricingResponse(
        cable_ref="540317340",
        suggested_price_usd=2.55,
        confidence_score=0.87,
        justification=[
            "LME copper price: $9,500/ton",
            "Customer segment: mining (+15% margin)",
            "High urgency (+10%)"
        ],
        lme_copper_price=9500.0,
        calculation_timestamp=datetime.now()
    )
    
    assert response.suggested_price_usd > 0
    assert 0 <= response.confidence_score <= 1.0
    assert len(response.justification) > 0
    assert response.lme_copper_price > 0


def test_lme_price_data_model():
    """🔴 RED: Test modelo para data LME real-time"""
    from src.models.market import LMEPriceData
    
    lme_data = LMEPriceData(
        metal="copper",
        price_usd_per_ton=9500.50,
        timestamp=datetime.now(),
        exchange="LME",
        currency="USD",
        change_percent=2.5
    )
    
    assert lme_data.metal == "copper"
    assert lme_data.price_usd_per_ton > 0
    assert lme_data.is_fresh()  # Should be fresh data
    assert lme_data.get_price_per_kg() > 0  # Convert to kg


def test_customer_model():
    """🔴 RED: Test Customer model para segmentación"""
    from src.models.customer import Customer
    
    customer = Customer(
        name="Codelco Norte",
        segment="mining",
        credit_rating="AAA",
        historical_margin_avg=0.18,
        payment_terms_days=30,
        is_strategic=True
    )
    
    assert customer.segment in ["mining", "industrial", "utility", "residential"]
    assert customer.get_margin_multiplier() != 1.0  # Should have multiplier
    assert customer.is_strategic == True


def test_forecast_model():
    """🔴 RED: Test modelo para demand forecasting"""
    from src.models.forecast import DemandForecast
    
    forecast = DemandForecast(
        cable_ref="540317340",
        forecast_months=3,
        predicted_demand=[450, 520, 380],  # 3 months
        confidence_intervals=[(400, 500), (470, 570), (330, 430)],
        seasonality_factor=1.15,
        trend_direction="increasing"
    )
    
    assert len(forecast.predicted_demand) == forecast.forecast_months
    assert forecast.seasonality_factor > 0
    assert forecast.trend_direction in ["increasing", "decreasing", "stable"]
    assert forecast.get_peak_month() is not None


def test_quote_model():
    """🔴 RED: Test modelo para cotizaciones automáticas"""
    from src.models.quote import Quote
    
    quote = Quote(
        quote_id="Q-2025-001",
        customer_name="Codelco Norte",
        cable_ref="540317340",
        quantity=500,
        unit_price_usd=2.55,
        total_price_usd=1275.0,
        validity_days=30,
        terms_conditions=["FOB Santiago", "Payment NET 30", "Price subject to LME"],
        generated_by="pricing_agent",
        confidence_score=0.89
    )
    
    assert quote.total_price_usd == quote.quantity * quote.unit_price_usd
    assert quote.is_valid_quote()
    assert quote.get_expiry_date() is not None


def test_agent_response_model():
    """🔴 RED: Test modelo para respuestas de Agentes IA"""
    from src.models.agent import AgentResponse
    
    response = AgentResponse(
        agent_type="market_intelligence",
        query="What's the copper price trend?",
        response_text="Copper shows bullish trend, up +2.5% this week",
        confidence=0.91,
        data_sources=["LME", "trading_economics"],
        recommendations=["Consider price increase", "Monitor volatility"],
        timestamp=datetime.now()
    )
    
    assert response.agent_type in ["market_intelligence", "forecasting", "quote_generation"]
    assert response.confidence > 0.8  # High confidence expected
    assert len(response.data_sources) > 0
    assert len(response.recommendations) > 0