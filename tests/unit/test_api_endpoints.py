"""
🔴 RED PHASE - API Endpoints Tests
Sprint 2.2.2: API endpoints para cotizaciones automáticas

TESTS TO WRITE FIRST (RED):
- FastAPI application initialization
- Quote generation endpoints (POST /api/quote/generate)
- Current prices endpoints (GET /api/prices/current)
- Cable search endpoints (GET /api/cables/search)
- Pricing calculation endpoints (POST /api/pricing/calculate)
- Authentication and rate limiting
- Error handling and validation
- Response formatting

All tests MUST FAIL initially to follow TDD methodology.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

# Import will fail initially - that's expected in RED phase
from src.api.main import app
from src.api.endpoints.quotes import router as quotes_router
from src.api.endpoints.pricing import router as pricing_router
from src.api.endpoints.cables import router as cables_router
from src.api.models.requests import (
    QuoteGenerationRequest,
    PricingCalculationRequest,
    CableSearchRequest
)
from src.api.models.responses import (
    QuoteGenerationResponse,
    PricingCalculationResponse,
    CableSearchResponse,
    ErrorResponse
)


class TestFastAPIApplication:
    """🔴 RED: Test FastAPI application setup"""
    
    def test_fastapi_app_creation(self):
        """🔴 RED: Test FastAPI app can be created"""
        # EXPECT: app doesn't exist yet
        assert app is not None
        assert hasattr(app, 'title')
        assert app.title == "Nexans Pricing Intelligence API"
        assert hasattr(app, 'version')
        assert app.version == "2.0.0"
    
    def test_api_routers_included(self):
        """🔴 RED: Test all routers are included in app"""
        client = TestClient(app)
        
        # Test that routers are properly included
        routes = [route.path for route in app.routes]
        
        assert "/api/quotes" in [route for route in routes if route.startswith("/api/quotes")]
        assert "/api/pricing" in [route for route in routes if route.startswith("/api/pricing")]
        assert "/api/cables" in [route for route in routes if route.startswith("/api/cables")]
        assert "/health" in routes
        assert "/docs" in routes
    
    def test_health_endpoint(self):
        """🔴 RED: Test health check endpoint"""
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
        assert "lme_api" in data["services"]
        assert "ml_model" in data["services"]


class TestQuoteGenerationEndpoints:
    """🔴 RED: Test quote generation API endpoints"""
    
    def test_quote_generation_endpoint_exists(self):
        """🔴 RED: Test POST /api/quotes/generate endpoint exists"""
        client = TestClient(app)
        
        # Test endpoint exists (should return method not allowed or validation error, not 404)
        response = client.post("/api/quotes/generate")
        assert response.status_code != 404
    
    def test_quote_generation_with_valid_data(self):
        """🔴 RED: Test quote generation with valid request data"""
        client = TestClient(app)
        
        quote_request = {
            "cable_reference": "540317340",
            "customer": {
                "name": "Minera Los Pelambres",
                "segment": "mining",
                "tier": "enterprise",
                "region": "chile_north"
            },
            "order": {
                "quantity": 1500,
                "urgency": "urgent",
                "delivery_date": "2024-03-15"
            },
            "pricing_options": {
                "include_volume_discount": True,
                "include_regional_factors": True,
                "include_urgency_premium": True
            }
        }
        
        response = client.post("/api/quotes/generate", json=quote_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "quote_id" in data
        assert "cable_reference" in data
        assert "customer" in data
        assert "pricing_breakdown" in data
        assert "total_price" in data
        assert "quote_timestamp" in data
        assert "validity_period" in data
        
        # Verify pricing breakdown structure
        breakdown = data["pricing_breakdown"]
        assert "base_cost" in breakdown
        assert "segment_multiplier" in breakdown
        assert "volume_discount" in breakdown
        assert "regional_factor" in breakdown
        assert "urgency_multiplier" in breakdown
        assert "final_price" in breakdown
    
    def test_quote_generation_with_invalid_cable_reference(self):
        """🔴 RED: Test quote generation with invalid cable reference"""
        client = TestClient(app)
        
        quote_request = {
            "cable_reference": "INVALID_REF",
            "customer": {"segment": "mining"},
            "order": {"quantity": 100}
        }
        
        response = client.post("/api/quotes/generate", json=quote_request)
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "Cable not found" in data["error"]
    
    def test_quote_generation_validation_errors(self):
        """🔴 RED: Test quote generation input validation"""
        client = TestClient(app)
        
        # Test missing required fields
        invalid_request = {"cable_reference": "540317340"}
        
        response = client.post("/api/quotes/generate", json=invalid_request)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert "validation_errors" in data or "detail" in data
    
    def test_quote_generation_with_mock_ml_model(self):
        """🔴 RED: Test quote generation integrates with ML model"""
        client = TestClient(app)
        
        with patch('src.pricing.ml_model.PricingModel.predict') as mock_predict:
            mock_predict.return_value = 145.50
            
            quote_request = {
                "cable_reference": "540317340",
                "customer": {"segment": "industrial", "tier": "standard", "region": "chile_central"},
                "order": {"quantity": 500, "urgency": "standard"}
            }
            
            response = client.post("/api/quotes/generate", json=quote_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "ml_predicted_price" in data["pricing_breakdown"]
            mock_predict.assert_called_once()


class TestPricingCalculationEndpoints:
    """🔴 RED: Test pricing calculation API endpoints"""
    
    def test_pricing_calculation_endpoint_exists(self):
        """🔴 RED: Test POST /api/pricing/calculate endpoint exists"""
        client = TestClient(app)
        
        response = client.post("/api/pricing/calculate")
        assert response.status_code != 404
    
    def test_pricing_calculation_with_business_rules(self):
        """🔴 RED: Test pricing calculation applies business rules"""
        client = TestClient(app)
        
        calculation_request = {
            "cable_specs": {
                "nexans_reference": "540317340",
                "voltage_rating": 5000,
                "copper_content_kg": 2.3,
                "applications": ["mining"]
            },
            "customer": {
                "segment": "mining",
                "tier": "enterprise",
                "region": "chile_north"
            },
            "order": {
                "quantity": 2000,
                "urgency": "express"
            },
            "options": {
                "use_real_time_lme": True,
                "apply_business_rules": True,
                "optimize_margin": True
            }
        }
        
        response = client.post("/api/pricing/calculate", json=calculation_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "calculation_id" in data
        assert "cable_specs" in data
        assert "cost_breakdown" in data
        assert "business_rules_applied" in data
        assert "final_price" in data
        assert "lme_prices" in data
        
        # Verify business rules were applied
        rules_applied = data["business_rules_applied"]
        assert "segment_multiplier" in rules_applied
        assert "volume_discount" in rules_applied
        assert "regional_factor" in rules_applied
        assert "urgency_multiplier" in rules_applied
    
    def test_pricing_calculation_real_time_lme(self):
        """🔴 RED: Test pricing calculation uses real-time LME prices"""
        client = TestClient(app)
        
        with patch('src.services.lme_api.get_lme_copper_price') as mock_copper:
            with patch('src.services.lme_api.get_lme_aluminum_price') as mock_aluminum:
                mock_copper.return_value = 9500.0
                mock_aluminum.return_value = 2650.0
                
                calculation_request = {
                    "cable_specs": {
                        "copper_content_kg": 1.5,
                        "aluminum_content_kg": 0.8
                    },
                    "customer": {"segment": "industrial"},
                    "order": {"quantity": 100},
                    "options": {"use_real_time_lme": True}
                }
                
                response = client.post("/api/pricing/calculate", json=calculation_request)
                assert response.status_code == 200
                
                data = response.json()
                lme_prices = data["lme_prices"]
                assert lme_prices["copper_usd_per_ton"] == 9500.0
                assert lme_prices["aluminum_usd_per_ton"] == 2650.0
                
                mock_copper.assert_called_once()
                mock_aluminum.assert_called_once()


class TestCableSearchEndpoints:
    """🔴 RED: Test cable search API endpoints"""
    
    def test_cable_search_endpoint_exists(self):
        """🔴 RED: Test GET /api/cables/search endpoint exists"""
        client = TestClient(app)
        
        response = client.get("/api/cables/search")
        assert response.status_code != 404
    
    def test_cable_search_by_reference(self):
        """🔴 RED: Test cable search by Nexans reference"""
        client = TestClient(app)
        
        response = client.get("/api/cables/search?reference=540317340")
        assert response.status_code == 200
        
        data = response.json()
        assert "cables" in data
        assert len(data["cables"]) > 0
        
        cable = data["cables"][0]
        assert "nexans_reference" in cable
        assert cable["nexans_reference"] == "540317340"
        assert "product_name" in cable
        assert "voltage_rating" in cable
        assert "applications" in cable
    
    def test_cable_search_by_specifications(self):
        """🔴 RED: Test cable search by technical specifications"""
        client = TestClient(app)
        
        search_params = {
            "voltage_min": 1000,
            "voltage_max": 10000,
            "application": "mining",
            "current_min": 100
        }
        
        response = client.get("/api/cables/search", params=search_params)
        assert response.status_code == 200
        
        data = response.json()
        assert "cables" in data
        assert "search_criteria" in data
        assert "total_results" in data
        
        # Verify filtering worked
        for cable in data["cables"]:
            assert cable["voltage_rating"] >= 1000
            assert cable["voltage_rating"] <= 10000
            assert "mining" in cable["applications"]
            assert cable["current_rating"] >= 100
    
    def test_cable_search_pagination(self):
        """🔴 RED: Test cable search supports pagination"""
        client = TestClient(app)
        
        response = client.get("/api/cables/search?page=1&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert "cables" in data
        assert "pagination" in data
        assert "page" in data["pagination"]
        assert "limit" in data["pagination"]
        assert "total_pages" in data["pagination"]
        assert "total_results" in data["pagination"]
        
        assert len(data["cables"]) <= 10
    
    def test_cable_search_no_results(self):
        """🔴 RED: Test cable search handles no results"""
        client = TestClient(app)
        
        response = client.get("/api/cables/search?voltage_min=999999")
        assert response.status_code == 200
        
        data = response.json()
        assert "cables" in data
        assert len(data["cables"]) == 0
        assert data["total_results"] == 0


class TestCurrentPricesEndpoints:
    """🔴 RED: Test current prices API endpoints"""
    
    def test_current_prices_endpoint_exists(self):
        """🔴 RED: Test GET /api/prices/current endpoint exists"""
        client = TestClient(app)
        
        response = client.get("/api/prices/current")
        assert response.status_code == 200
    
    def test_current_lme_prices(self):
        """🔴 RED: Test current LME prices endpoint"""
        client = TestClient(app)
        
        with patch('src.services.lme_api.get_lme_copper_price') as mock_copper:
            with patch('src.services.lme_api.get_lme_aluminum_price') as mock_aluminum:
                mock_copper.return_value = 9500.0
                mock_aluminum.return_value = 2650.0
                
                response = client.get("/api/prices/current/lme")
                assert response.status_code == 200
                
                data = response.json()
                assert "lme_prices" in data
                assert "timestamp" in data
                assert "source" in data
                
                lme_prices = data["lme_prices"]
                assert "copper_usd_per_ton" in lme_prices
                assert "aluminum_usd_per_ton" in lme_prices
                assert lme_prices["copper_usd_per_ton"] == 9500.0
                assert lme_prices["aluminum_usd_per_ton"] == 2650.0
    
    def test_cable_current_price(self):
        """🔴 RED: Test current price for specific cable"""
        client = TestClient(app)
        
        response = client.get("/api/prices/current/cable/540317340")
        assert response.status_code == 200
        
        data = response.json()
        assert "cable_reference" in data
        assert "current_price" in data
        assert "cost_breakdown" in data
        assert "last_updated" in data
        assert "lme_prices" in data
        
        # Verify cost breakdown structure
        breakdown = data["cost_breakdown"]
        assert "copper_cost" in breakdown
        assert "aluminum_cost" in breakdown
        assert "polymer_cost" in breakdown
        assert "manufacturing_cost" in breakdown
        assert "total_cost" in breakdown


class TestAPIValidationAndErrorHandling:
    """🔴 RED: Test API validation and error handling"""
    
    def test_api_authentication_required(self):
        """🔴 RED: Test API requires authentication for protected endpoints"""
        client = TestClient(app)
        
        # Test without API key
        response = client.post("/api/quotes/generate", json={})
        # Should require authentication or return 401/403
        assert response.status_code in [401, 403, 422]
    
    def test_api_rate_limiting(self):
        """🔴 RED: Test API rate limiting is enforced"""
        client = TestClient(app)
        
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.get("/api/prices/current")
            responses.append(response.status_code)
        
        # Should eventually hit rate limit
        assert 429 in responses or all(r == 200 for r in responses[:5])
    
    def test_api_input_validation(self):
        """🔴 RED: Test comprehensive input validation"""
        client = TestClient(app)
        
        # Test invalid JSON
        response = client.post(
            "/api/pricing/calculate",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test invalid data types
        invalid_request = {
            "cable_specs": {
                "voltage_rating": "not_a_number",
                "copper_content_kg": -1.0  # Negative value
            },
            "order": {
                "quantity": 0  # Zero quantity
            }
        }
        
        response = client.post("/api/pricing/calculate", json=invalid_request)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data or "validation_errors" in data
    
    def test_api_error_response_format(self):
        """🔴 RED: Test consistent error response format"""
        client = TestClient(app)
        
        # Trigger a 404 error
        response = client.get("/api/cables/search?reference=NONEXISTENT")
        
        if response.status_code == 404:
            data = response.json()
            assert "error" in data
            assert "code" in data
            assert "message" in data
            assert "timestamp" in data
    
    def test_api_cors_headers(self):
        """🔴 RED: Test CORS headers are properly set"""
        client = TestClient(app)
        
        # Test preflight request
        response = client.options("/api/prices/current")
        headers = response.headers
        
        assert "access-control-allow-origin" in headers or response.status_code == 405
    
    def test_api_response_time_limits(self):
        """🔴 RED: Test API response time limits"""
        client = TestClient(app)
        
        import time
        start_time = time.time()
        
        response = client.get("/api/prices/current")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should respond within 5 seconds
        assert response_time < 5.0
        assert response.status_code == 200


class TestAPIDocumentationAndMetadata:
    """🔴 RED: Test API documentation and metadata"""
    
    def test_openapi_schema_available(self):
        """🔴 RED: Test OpenAPI schema is available"""
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
    
    def test_swagger_ui_available(self):
        """🔴 RED: Test Swagger UI is available"""
        client = TestClient(app)
        
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_api_metadata_endpoints(self):
        """🔴 RED: Test API metadata endpoints"""
        client = TestClient(app)
        
        response = client.get("/api/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "version" in data
            assert "title" in data
            assert "description" in data
            assert "endpoints" in data