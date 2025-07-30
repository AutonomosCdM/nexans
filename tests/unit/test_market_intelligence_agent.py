"""
🔴 RED PHASE - Market Intelligence Agent Tests
Sprint 3.1: Market Intelligence Agent para LME monitoring y price alerts

TESTS TO WRITE FIRST (RED):
- MarketIntelligenceAgent core functionality
- LME price monitoring and alerts
- Market volatility detection
- Competitor price tracking 
- Automated pricing recommendations
- Market trend analysis
- Price alert notifications
- Historical price data analysis

All tests MUST FAIL initially to follow TDD methodology.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import asyncio

# Import will fail initially - that's expected in RED phase
from src.agents.market_intelligence import (
    MarketIntelligenceAgent,
    MarketAlert,
    PriceVolatilityDetector,
    CompetitorPriceTracker,
    PricingRecommendationEngine,
    MarketTrendAnalyzer,
    AlertNotificationService,
    MarketIntelligenceError
)


class TestMarketIntelligenceAgent:
    """🔴 RED: Test Market Intelligence Agent core functionality"""
    
    def test_market_intelligence_agent_initialization(self):
        """🔴 RED: Test MarketIntelligenceAgent can be instantiated"""
        # EXPECT: MarketIntelligenceAgent class doesn't exist yet
        agent = MarketIntelligenceAgent()
        assert agent is not None
        assert hasattr(agent, 'price_monitor')
        assert hasattr(agent, 'volatility_detector')
        assert hasattr(agent, 'competitor_tracker')
        assert hasattr(agent, 'recommendation_engine')
        assert hasattr(agent, 'trend_analyzer')
        assert hasattr(agent, 'notification_service')
    
    def test_agent_start_monitoring(self):
        """🔴 RED: Test agent can start price monitoring"""
        agent = MarketIntelligenceAgent()
        
        # Should be able to start monitoring
        result = agent.start_monitoring()
        assert result == True
        assert agent.is_monitoring == True
        assert agent.monitoring_start_time is not None
    
    def test_agent_stop_monitoring(self):
        """🔴 RED: Test agent can stop price monitoring"""
        agent = MarketIntelligenceAgent()
        agent.start_monitoring()
        
        # Should be able to stop monitoring
        result = agent.stop_monitoring()
        assert result == True
        assert agent.is_monitoring == False
        assert agent.monitoring_end_time is not None
    
    def test_agent_get_market_status(self):
        """🔴 RED: Test agent can provide market status"""
        agent = MarketIntelligenceAgent()
        
        status = agent.get_market_status()
        assert isinstance(status, dict)
        assert "lme_prices" in status
        assert "volatility_level" in status
        assert "market_trend" in status
        assert "alerts_count" in status
        assert "last_update" in status
    
    @pytest.mark.asyncio
    async def test_agent_process_market_data_async(self):
        """🔴 RED: Test agent can process market data asynchronously"""
        agent = MarketIntelligenceAgent()
        
        # Mock market data
        market_data = {
            "copper_price": 9500.0,
            "aluminum_price": 2650.0,
            "timestamp": datetime.now(),
            "source": "LME"
        }
        
        result = await agent.process_market_data(market_data)
        assert result is not None
        assert "alerts" in result
        assert "recommendations" in result
        assert "volatility_analysis" in result


class TestMarketAlert:
    """🔴 RED: Test Market Alert data structure"""
    
    def test_market_alert_creation(self):
        """🔴 RED: Test MarketAlert can be created with required fields"""
        alert = MarketAlert(
            alert_id="ALERT_001",
            alert_type="PRICE_SPIKE",
            metal="copper",
            current_price=9800.0,
            previous_price=9500.0,
            threshold_exceeded=3.0,  # 3% threshold
            severity="HIGH",
            message="Copper price spiked 3.16% in last hour",
            timestamp=datetime.now()
        )
        
        assert alert.alert_id == "ALERT_001"
        assert alert.alert_type == "PRICE_SPIKE"
        assert alert.metal == "copper"
        assert alert.current_price == 9800.0
        assert alert.severity == "HIGH"
        assert "spiked" in alert.message
    
    def test_market_alert_validation(self):
        """🔴 RED: Test MarketAlert validates required fields"""
        # Should raise error for missing required fields
        with pytest.raises(ValueError):
            MarketAlert()
        
        # Should raise error for invalid severity
        with pytest.raises(ValueError):
            MarketAlert(
                alert_id="ALERT_002",
                alert_type="PRICE_DROP",
                metal="aluminum",
                current_price=2500.0,
                previous_price=2650.0,
                severity="INVALID_SEVERITY"  # Invalid severity
            )
    
    def test_market_alert_price_change_calculation(self):
        """🔴 RED: Test MarketAlert calculates price change correctly"""
        alert = MarketAlert(
            alert_id="ALERT_003",
            alert_type="PRICE_DROP",
            metal="aluminum",
            current_price=2500.0,
            previous_price=2650.0,
            severity="MEDIUM"
        )
        
        price_change = alert.calculate_price_change()
        assert price_change == -150.0  # 2500 - 2650
        
        price_change_pct = alert.calculate_price_change_percentage()
        assert abs(price_change_pct - (-5.66)) < 0.01  # -5.66%


class TestPriceVolatilityDetector:
    """🔴 RED: Test Price Volatility Detection"""
    
    def test_volatility_detector_initialization(self):
        """🔴 RED: Test PriceVolatilityDetector initialization"""
        detector = PriceVolatilityDetector()
        assert detector is not None
        assert hasattr(detector, 'volatility_thresholds')
        assert hasattr(detector, 'price_history')
        assert hasattr(detector, 'detection_window')
    
    def test_detect_price_volatility_low(self):
        """🔴 RED: Test detection of low volatility"""
        detector = PriceVolatilityDetector()
        
        # Stable prices - low volatility
        price_data = [
            {"price": 9500.0, "timestamp": datetime.now() - timedelta(hours=5)},
            {"price": 9510.0, "timestamp": datetime.now() - timedelta(hours=4)},
            {"price": 9505.0, "timestamp": datetime.now() - timedelta(hours=3)},
            {"price": 9515.0, "timestamp": datetime.now() - timedelta(hours=2)},
            {"price": 9508.0, "timestamp": datetime.now() - timedelta(hours=1)},
        ]
        
        volatility_level = detector.detect_volatility("copper", price_data)
        assert volatility_level == "LOW"
    
    def test_detect_price_volatility_high(self):
        """🔴 RED: Test detection of high volatility"""
        detector = PriceVolatilityDetector()
        
        # Highly volatile prices
        price_data = [
            {"price": 9500.0, "timestamp": datetime.now() - timedelta(hours=5)},
            {"price": 9800.0, "timestamp": datetime.now() - timedelta(hours=4)},
            {"price": 9200.0, "timestamp": datetime.now() - timedelta(hours=3)},
            {"price": 9900.0, "timestamp": datetime.now() - timedelta(hours=2)},
            {"price": 9300.0, "timestamp": datetime.now() - timedelta(hours=1)},
        ]
        
        volatility_level = detector.detect_volatility("copper", price_data)
        assert volatility_level == "HIGH"
    
    def test_calculate_volatility_metrics(self):
        """🔴 RED: Test volatility metrics calculation"""
        detector = PriceVolatilityDetector()
        
        price_data = [9500.0, 9550.0, 9480.0, 9620.0, 9510.0]
        
        metrics = detector.calculate_volatility_metrics(price_data)
        assert "standard_deviation" in metrics
        assert "coefficient_of_variation" in metrics
        assert "price_range" in metrics
        assert "average_price" in metrics
        
        assert metrics["standard_deviation"] > 0
        assert metrics["price_range"] == 9620.0 - 9480.0  # 140.0


class TestCompetitorPriceTracker:
    """🔴 RED: Test Competitor Price Tracking"""
    
    def test_competitor_tracker_initialization(self):
        """🔴 RED: Test CompetitorPriceTracker initialization"""
        tracker = CompetitorPriceTracker()
        assert tracker is not None
        assert hasattr(tracker, 'competitors')
        assert hasattr(tracker, 'price_sources')
        assert hasattr(tracker, 'tracking_intervals')
    
    def test_add_competitor(self):
        """🔴 RED: Test adding competitor for tracking"""
        tracker = CompetitorPriceTracker()
        
        competitor_info = {
            "name": "Southwire",
            "market_segment": "industrial",
            "price_source": "public_quotes",
            "tracking_enabled": True,
            "price_adjustment_factor": 0.95  # Typically 5% lower than Nexans
        }
        
        result = tracker.add_competitor("southwire", competitor_info)
        assert result == True
        assert "southwire" in tracker.competitors
        assert tracker.competitors["southwire"]["name"] == "Southwire"
    
    def test_track_competitor_prices(self):
        """🔴 RED: Test tracking competitor prices"""
        tracker = CompetitorPriceTracker()
        
        # Add competitor first
        tracker.add_competitor("prysmian", {
            "name": "Prysmian Group",
            "market_segment": "mining",
            "price_source": "market_quotes"
        })
        
        # Mock price data
        with patch.object(tracker, '_fetch_competitor_price') as mock_fetch:
            mock_fetch.return_value = 42.50  # USD per meter
            
            prices = tracker.track_competitor_prices("mining_cable_5kv")
            assert isinstance(prices, dict)
            assert "prysmian" in prices
            assert prices["prysmian"] == 42.50
    
    def test_compare_with_nexans_pricing(self):
        """🔴 RED: Test comparison with Nexans pricing"""
        tracker = CompetitorPriceTracker()
        
        # Add competitors
        tracker.add_competitor("general_cable", {
            "name": "General Cable",
            "market_segment": "utility"
        })
        
        nexans_price = 45.83  # From our cost calculator
        competitor_prices = {
            "general_cable": 43.20,
            "prysmian": 47.10
        }
        
        comparison = tracker.compare_with_nexans_pricing(nexans_price, competitor_prices)
        
        assert "price_position" in comparison
        assert "competitive_advantage" in comparison
        assert "recommendations" in comparison
        
        # Nexans should be in middle position
        assert comparison["price_position"] == "COMPETITIVE"


class TestPricingRecommendationEngine:
    """🔴 RED: Test Pricing Recommendation Engine"""
    
    def test_recommendation_engine_initialization(self):
        """🔴 RED: Test PricingRecommendationEngine initialization"""
        engine = PricingRecommendationEngine()
        assert engine is not None
        assert hasattr(engine, 'recommendation_rules')
        assert hasattr(engine, 'market_conditions')
        assert hasattr(engine, 'pricing_strategies')
    
    def test_generate_pricing_recommendations_volatile_market(self):
        """🔴 RED: Test recommendations for volatile market conditions"""
        engine = PricingRecommendationEngine()
        
        market_context = {
            "volatility_level": "HIGH",
            "price_trend": "INCREASING",
            "competitor_position": "BELOW_MARKET",
            "demand_level": "HIGH",
            "lme_copper_price": 9800.0,  # High copper price
            "current_nexans_price": 45.83
        }
        
        recommendations = engine.generate_recommendations(market_context)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend price increase due to high volatility and rising trend
        price_recommendations = [r for r in recommendations if r["type"] == "PRICE_ADJUSTMENT"]
        assert len(price_recommendations) > 0
        
        # Should suggest increasing prices
        price_rec = price_recommendations[0]
        assert price_rec["action"] == "INCREASE"
        assert price_rec["magnitude"] > 0
    
    def test_generate_pricing_recommendations_stable_market(self):
        """🔴 RED: Test recommendations for stable market conditions"""
        engine = PricingRecommendationEngine()
        
        market_context = {
            "volatility_level": "LOW",
            "price_trend": "STABLE",
            "competitor_position": "COMPETITIVE",
            "demand_level": "NORMAL",
            "lme_copper_price": 9500.0,  # Normal copper price
            "current_nexans_price": 45.83
        }
        
        recommendations = engine.generate_recommendations(market_context)
        
        # Should recommend maintaining current prices
        price_recommendations = [r for r in recommendations if r["type"] == "PRICE_ADJUSTMENT"]
        
        if price_recommendations:
            price_rec = price_recommendations[0]
            assert price_rec["action"] in ["MAINTAIN", "SLIGHT_ADJUSTMENT"]
    
    def test_calculate_optimal_price_adjustment(self):
        """🔴 RED: Test optimal price adjustment calculation"""
        engine = PricingRecommendationEngine()
        
        current_price = 45.83
        market_factors = {
            "lme_price_change": 0.05,  # 5% increase in LME
            "volatility_adjustment": 0.02,  # 2% volatility premium
            "competitive_adjustment": -0.01,  # 1% competitive discount
            "demand_adjustment": 0.03  # 3% demand premium
        }
        
        optimal_price = engine.calculate_optimal_price_adjustment(current_price, market_factors)
        
        # Should increase price due to positive factors
        assert optimal_price > current_price
        
        # Should be reasonable adjustment (not more than 15%)
        price_change_pct = (optimal_price - current_price) / current_price
        assert 0 < price_change_pct < 0.15


class TestMarketTrendAnalyzer:
    """🔴 RED: Test Market Trend Analysis"""
    
    def test_trend_analyzer_initialization(self):
        """🔴 RED: Test MarketTrendAnalyzer initialization"""
        analyzer = MarketTrendAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'trend_detection_methods')
        assert hasattr(analyzer, 'historical_data_window')
        assert hasattr(analyzer, 'trend_confidence_threshold')
    
    def test_analyze_price_trend_increasing(self):
        """🔴 RED: Test detection of increasing price trend"""
        analyzer = MarketTrendAnalyzer()
        
        # Increasing price data over time
        price_history = [
            {"price": 9000.0, "timestamp": datetime.now() - timedelta(days=30)},
            {"price": 9100.0, "timestamp": datetime.now() - timedelta(days=25)},
            {"price": 9250.0, "timestamp": datetime.now() - timedelta(days=20)},
            {"price": 9400.0, "timestamp": datetime.now() - timedelta(days=15)},
            {"price": 9500.0, "timestamp": datetime.now() - timedelta(days=10)},
            {"price": 9650.0, "timestamp": datetime.now() - timedelta(days=5)},
            {"price": 9800.0, "timestamp": datetime.now()}
        ]
        
        trend_analysis = analyzer.analyze_price_trend("copper", price_history)
        
        assert trend_analysis["trend_direction"] == "INCREASING"
        assert trend_analysis["trend_strength"] in ["STRONG", "MODERATE"]
        assert trend_analysis["confidence_level"] > 0.7
    
    def test_analyze_price_trend_decreasing(self):
        """🔴 RED: Test detection of decreasing price trend"""
        analyzer = MarketTrendAnalyzer()
        
        # Decreasing price data over time
        price_history = [
            {"price": 9800.0, "timestamp": datetime.now() - timedelta(days=30)},
            {"price": 9650.0, "timestamp": datetime.now() - timedelta(days=25)},
            {"price": 9500.0, "timestamp": datetime.now() - timedelta(days=20)},
            {"price": 9350.0, "timestamp": datetime.now() - timedelta(days=15)},
            {"price": 9200.0, "timestamp": datetime.now() - timedelta(days=10)},
            {"price": 9100.0, "timestamp": datetime.now() - timedelta(days=5)},
            {"price": 9000.0, "timestamp": datetime.now()}
        ]
        
        trend_analysis = analyzer.analyze_price_trend("copper", price_history)
        
        assert trend_analysis["trend_direction"] == "DECREASING"
        assert trend_analysis["trend_strength"] in ["STRONG", "MODERATE"]
        assert trend_analysis["confidence_level"] > 0.7
    
    def test_predict_future_prices(self):
        """🔴 RED: Test future price prediction"""
        analyzer = MarketTrendAnalyzer()
        
        # Historical price data
        price_history = [9000.0, 9050.0, 9100.0, 9200.0, 9300.0, 9400.0, 9500.0]
        
        # Predict next 5 days
        predictions = analyzer.predict_future_prices("copper", price_history, days_ahead=5)
        
        assert len(predictions) == 5
        assert all(isinstance(p, dict) for p in predictions)
        assert all("predicted_price" in p and "confidence" in p for p in predictions)
        
        # Predictions should follow the increasing trend
        assert predictions[0]["predicted_price"] > 9500.0
        assert predictions[-1]["predicted_price"] > predictions[0]["predicted_price"]


class TestAlertNotificationService:
    """🔴 RED: Test Alert Notification Service"""
    
    def test_notification_service_initialization(self):
        """🔴 RED: Test AlertNotificationService initialization"""
        service = AlertNotificationService()
        assert service is not None
        assert hasattr(service, 'notification_channels')
        assert hasattr(service, 'alert_templates')
        assert hasattr(service, 'delivery_methods')
    
    def test_send_price_alert_notification(self):
        """🔴 RED: Test sending price alert notifications"""
        service = AlertNotificationService()
        
        alert = MarketAlert(
            alert_id="ALERT_004",
            alert_type="PRICE_SPIKE",
            metal="copper",
            current_price=9800.0,
            previous_price=9500.0,
            severity="HIGH",
            message="Copper price spiked 3.16% in last hour"
        )
        
        recipients = ["pricing_team@nexans.com", "sales_manager@nexans.com"]
        
        with patch.object(service, '_send_email') as mock_email:
            mock_email.return_value = True
            
            result = service.send_alert_notification(alert, recipients, method="email")
            assert result == True
            mock_email.assert_called_once()
    
    def test_format_alert_message(self):
        """🔴 RED: Test alert message formatting"""
        service = AlertNotificationService()
        
        alert = MarketAlert(
            alert_id="ALERT_005",
            alert_type="VOLATILITY_WARNING",
            metal="aluminum",
            current_price=2750.0,
            previous_price=2650.0,
            severity="MEDIUM",
            message="Aluminum volatility increased significantly"
        )
        
        formatted_message = service.format_alert_message(alert, template="email")
        
        assert "ALERT_005" in formatted_message
        assert "aluminum" in formatted_message.lower()
        assert "2750.0" in formatted_message
        assert "MEDIUM" in formatted_message
    
    def test_batch_send_notifications(self):
        """🔴 RED: Test sending multiple notifications in batch"""
        service = AlertNotificationService()
        
        alerts = [
            MarketAlert(
                alert_id="BATCH_001",
                alert_type="PRICE_SPIKE",
                metal="copper",
                current_price=9800.0,
                severity="HIGH"
            ),
            MarketAlert(
                alert_id="BATCH_002", 
                alert_type="TREND_CHANGE",
                metal="aluminum",
                current_price=2700.0,
                severity="MEDIUM"
            )
        ]
        
        recipients = ["team@nexans.com"]
        
        with patch.object(service, 'send_alert_notification') as mock_send:
            mock_send.return_value = True
            
            results = service.batch_send_notifications(alerts, recipients)
            
            assert len(results) == 2
            assert all(r == True for r in results)
            assert mock_send.call_count == 2


class TestMarketIntelligenceIntegration:
    """🔴 RED: Test Market Intelligence Agent integration"""
    
    @pytest.mark.asyncio
    async def test_full_market_intelligence_workflow(self):
        """🔴 RED: Test complete market intelligence workflow"""
        agent = MarketIntelligenceAgent()
        
        # Start monitoring
        agent.start_monitoring()
        
        # Simulate market data update
        market_update = {
            "copper_price": 9850.0,  # Significant increase
            "aluminum_price": 2680.0,
            "timestamp": datetime.now(),
            "source": "LME_REAL_TIME"
        }
        
        # Process the market update
        result = await agent.process_market_data(market_update)
        
        # Should detect volatility and generate alerts
        assert "alerts" in result
        assert len(result["alerts"]) > 0
        
        # Should provide recommendations  
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0
        
        # Should analyze volatility
        assert "volatility_analysis" in result
        assert result["volatility_analysis"]["level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_market_intelligence_error_handling(self):
        """🔴 RED: Test error handling in market intelligence"""
        agent = MarketIntelligenceAgent()
        
        # Test with invalid market data
        invalid_data = {
            "copper_price": "invalid_price",  # Should be float
            "timestamp": "invalid_timestamp"  # Should be datetime
        }
        
        with pytest.raises(MarketIntelligenceError):
            asyncio.run(agent.process_market_data(invalid_data))
    
    def test_market_intelligence_performance(self):
        """🔴 RED: Test market intelligence performance requirements"""
        agent = MarketIntelligenceAgent()
        
        # Should process market data quickly (< 1 second)
        start_time = datetime.now()
        
        market_data = {
            "copper_price": 9500.0,
            "aluminum_price": 2650.0,
            "timestamp": datetime.now()
        }
        
        result = asyncio.run(agent.process_market_data(market_data))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert processing_time < 1.0  # Should process within 1 second
        assert result is not None