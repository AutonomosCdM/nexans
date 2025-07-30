"""
🔴 RED PHASE - Tests para LME Price API - DEBEN FALLAR PRIMERO
Tarea 1.2.1: LME Price API con tests - Usando APIs reales
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal


def test_get_copper_price_lme():
    """🔴 RED: Test obtener precio LME copper real-time - DEBE FALLAR PRIMERO"""
    from src.services.lme_api import get_lme_copper_price
    
    price = get_lme_copper_price()
    
    # Validar rango realista (USD/ton)
    assert 8000 < price < 12000, f"Copper price {price} outside realistic range"
    assert isinstance(price, (int, float, Decimal))


def test_get_aluminum_price_lme():
    """🔴 RED: Test obtener precio LME aluminum"""
    from src.services.lme_api import get_lme_aluminum_price
    
    price = get_lme_aluminum_price()
    
    # Validar rango aluminio (USD/ton)
    assert 2000 < price < 4000, f"Aluminum price {price} outside realistic range"
    assert isinstance(price, (int, float, Decimal))


@patch('requests.get')
def test_lme_api_with_mock(mock_get):
    """🔴 RED: Test con mock para evitar llamadas reales en tests"""
    from src.services.lme_api import get_lme_copper_price
    
    # Mock response structure based on Metals-API
    mock_response = Mock()
    mock_response.json.return_value = {
        "success": True,
        "rates": {
            "LME-XCU": 9500.75
        },
        "timestamp": 1643723400,
        "base": "USD"
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    price = get_lme_copper_price()
    
    assert price == 9500.75
    mock_get.assert_called_once()


def test_lme_price_caching():
    """🔴 RED: Test sistema de cache para evitar API calls excesivos"""
    from src.services.lme_api import get_lme_copper_price, clear_price_cache
    
    clear_price_cache()
    
    # Primera llamada - debe hacer API request
    with patch('src.services.lme_api._fetch_from_api') as mock_fetch:
        mock_fetch.return_value = 9500.0
        
        price1 = get_lme_copper_price()
        price2 = get_lme_copper_price()  # Segunda llamada debe usar cache
        
        assert price1 == price2
        assert mock_fetch.call_count == 1  # Solo una llamada real


def test_lme_price_with_error_handling():
    """🔴 RED: Test manejo de errores API"""
    from src.services.lme_api import get_lme_copper_price, LMEAPIError
    
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")
        
        # Debe usar fallback price o lanzar exception específica
        with pytest.raises(LMEAPIError):
            get_lme_copper_price(use_fallback=False)


def test_lme_price_with_fallback():
    """🔴 RED: Test sistema de fallback cuando API falla"""
    from src.services.lme_api import get_lme_copper_price
    
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("API down")
        
        # Con fallback debe retornar precio por defecto
        price = get_lme_copper_price(use_fallback=True)
        
        assert price > 0
        assert 8000 < price < 12000  # Fallback en rango realista


def test_lme_historical_prices():
    """🔴 RED: Test obtener precios históricos"""
    from src.services.lme_api import get_lme_historical_prices
    
    # Obtener precios últimos 7 días
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    historical = get_lme_historical_prices(
        metal="copper",
        start_date=start_date,
        end_date=end_date
    )
    
    assert len(historical) > 0
    assert len(historical) <= 7  # Max 7 days
    assert all(isinstance(price, dict) for price in historical)
    assert all('date' in price and 'price' in price for price in historical)


def test_lme_price_change_calculation():
    """🔴 RED: Test calcular cambio porcentual de precios"""
    from src.services.lme_api import calculate_price_change
    
    current_price = 9500.0
    previous_price = 9000.0
    
    change_pct = calculate_price_change(current_price, previous_price)
    
    expected_change = ((9500 - 9000) / 9000) * 100  # ~5.56%
    assert abs(change_pct - expected_change) < 0.01


def test_lme_price_to_model_conversion():
    """🔴 RED: Test conversión a modelo LMEPriceData"""
    from src.services.lme_api import fetch_and_convert_lme_data
    from src.models.market import LMEPriceData
    
    lme_data = fetch_and_convert_lme_data("copper")
    
    assert isinstance(lme_data, LMEPriceData)
    assert lme_data.metal == "copper"
    assert lme_data.price_usd_per_ton > 0
    assert lme_data.is_fresh()  # Should be recent data
    assert lme_data.get_price_per_kg() > 0


def test_lme_api_rate_limiting():
    """🔴 RED: Test rate limiting para evitar exceder límites API"""
    from src.services.lme_api import get_lme_copper_price, APIRateLimitError
    
    # Simular muchas llamadas rápidas
    with patch('time.time') as mock_time:
        mock_time.return_value = 1643723400  # Fixed timestamp
        
        # Primera llamada OK
        price1 = get_lme_copper_price()
        
        # Llamadas inmediatas deben usar cache o rate limit
        for i in range(10):
            price = get_lme_copper_price()
            assert price == price1  # Should be cached


def test_lme_api_authentication():
    """🔴 RED: Test autenticación con API key"""
    from src.services.lme_api import LMEAPIClient
    
    # Test con API key válida
    client = LMEAPIClient(api_key="test_key_123")
    assert client.api_key == "test_key_123"
    assert client.is_authenticated()
    
    # Test sin API key
    client_no_auth = LMEAPIClient()
    assert not client_no_auth.is_authenticated()


def test_multiple_metals_pricing():
    """🔴 RED: Test obtener precios de múltiples metales"""
    from src.services.lme_api import get_multiple_metal_prices
    
    metals = ["copper", "aluminum", "nickel"]
    prices = get_multiple_metal_prices(metals)
    
    assert len(prices) == len(metals)
    assert all(metal in prices for metal in metals)
    assert all(prices[metal] > 0 for metal in metals)


@pytest.fixture
def mock_lme_response():
    """🔴 RED: Fixture con respuesta LME realista"""
    return {
        "success": True,
        "timestamp": 1643723400,
        "base": "USD",
        "rates": {
            "LME-XCU": 9500.75,  # Copper
            "LME-XAL": 2650.25,  # Aluminum
            "LME-XNI": 21000.50  # Nickel
        }
    }


def test_parse_metals_api_response(mock_lme_response):
    """🔴 RED: Test parsing respuesta de Metals-API"""
    from src.services.lme_api import parse_metals_api_response
    
    parsed = parse_metals_api_response(mock_lme_response, "copper")
    
    assert parsed["metal"] == "copper"
    assert parsed["price"] == 9500.75
    assert parsed["timestamp"] == 1643723400
    assert parsed["currency"] == "USD"