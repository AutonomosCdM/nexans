"""
🟢 GREEN PHASE - Market Intelligence Agent Implementation
Sprint 3.1: Market Intelligence Agent para LME monitoring y price alerts

IMPLEMENTATION TO MAKE TESTS PASS:
✅ MarketIntelligenceAgent: Main orchestrator for market monitoring
✅ MarketAlert: Alert data structure with validation
✅ PriceVolatilityDetector: Volatility detection and analysis  
✅ CompetitorPriceTracker: Competitor pricing monitoring
✅ PricingRecommendationEngine: Automated pricing recommendations
✅ MarketTrendAnalyzer: Trend analysis and prediction
✅ AlertNotificationService: Alert delivery system

All implementations follow TDD methodology - minimal code to pass tests.
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
import logging
import numpy as np
from enum import Enum

# Import existing components for integration
from src.services.lme_api import get_lme_copper_price, get_lme_aluminum_price

# Configure logging
logger = logging.getLogger(__name__)


class MarketIntelligenceError(Exception):
    """🟢 GREEN: Custom exception for market intelligence errors"""
    pass


class AlertSeverity(str, Enum):
    """🟢 GREEN: Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    """🟢 GREEN: Alert types"""
    PRICE_SPIKE = "PRICE_SPIKE"
    PRICE_DROP = "PRICE_DROP"
    VOLATILITY_WARNING = "VOLATILITY_WARNING"
    TREND_CHANGE = "TREND_CHANGE"
    COMPETITOR_ALERT = "COMPETITOR_ALERT"


@dataclass
class MarketAlert:
    """🟢 GREEN: Market alert data structure"""
    alert_id: str
    alert_type: str
    metal: str
    current_price: float
    severity: str
    previous_price: Optional[float] = None
    threshold_exceeded: Optional[float] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """🟢 GREEN: Validate alert data after initialization"""
        if not self.alert_id:
            raise ValueError("Alert ID is required")
        
        if self.severity not in [s.value for s in AlertSeverity]:
            raise ValueError(f"Invalid severity: {self.severity}")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        if self.message is None:
            self.message = f"{self.metal.title()} {self.alert_type.lower()} detected"
    
    def calculate_price_change(self) -> float:
        """🟢 GREEN: Calculate absolute price change"""
        if self.previous_price is None:
            return 0.0
        return self.current_price - self.previous_price
    
    def calculate_price_change_percentage(self) -> float:
        """🟢 GREEN: Calculate percentage price change"""
        if self.previous_price is None or self.previous_price == 0:
            return 0.0
        change = (self.current_price - self.previous_price) / self.previous_price * 100
        return round(change, 2)


class PriceVolatilityDetector:
    """🟢 GREEN: Price volatility detection and analysis"""
    
    def __init__(self):
        self.volatility_thresholds = {
            "LOW": 0.02,     # < 2% standard deviation  
            "MEDIUM": 0.05,  # 2-5% standard deviation
            "HIGH": 0.10     # > 5% standard deviation
        }
        self.price_history = {}
        self.detection_window = 24  # hours
    
    def detect_volatility(self, metal: str, price_data: List[Dict]) -> str:
        """🟢 GREEN: Detect volatility level from price data"""
        if len(price_data) < 3:
            return "LOW"  # Not enough data
        
        prices = [p["price"] for p in price_data]
        metrics = self.calculate_volatility_metrics(prices)
        
        cv = metrics["coefficient_of_variation"]
        
        if cv < self.volatility_thresholds["LOW"]:
            return "LOW"
        elif cv < self.volatility_thresholds["MEDIUM"]:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def calculate_volatility_metrics(self, prices: List[float]) -> Dict[str, float]:
        """🟢 GREEN: Calculate volatility metrics"""
        if len(prices) < 2:
            return {
                "standard_deviation": 0.0,
                "coefficient_of_variation": 0.0,
                "price_range": 0.0,
                "average_price": prices[0] if prices else 0.0
            }
        
        avg_price = statistics.mean(prices)
        std_dev = statistics.stdev(prices)
        price_range = max(prices) - min(prices)
        cv = std_dev / avg_price if avg_price > 0 else 0.0
        
        return {
            "standard_deviation": round(std_dev, 2),
            "coefficient_of_variation": round(cv, 4),
            "price_range": round(price_range, 2),
            "average_price": round(avg_price, 2)
        }


class CompetitorPriceTracker:
    """🟢 GREEN: Competitor price tracking and analysis"""
    
    def __init__(self):
        self.competitors = {}
        self.price_sources = {}
        self.tracking_intervals = {}
    
    def add_competitor(self, competitor_id: str, competitor_info: Dict) -> bool:
        """🟢 GREEN: Add competitor for tracking"""
        try:
            self.competitors[competitor_id] = competitor_info
            return True
        except Exception as e:
            logger.error(f"Failed to add competitor: {e}")
            return False
    
    def track_competitor_prices(self, product_category: str) -> Dict[str, float]:
        """🟢 GREEN: Track competitor prices for product category"""
        competitor_prices = {}
        
        for competitor_id, info in self.competitors.items():
            try:
                # Mock price fetching - in production would call real APIs
                price = self._fetch_competitor_price(competitor_id, product_category)
                competitor_prices[competitor_id] = price
            except Exception as e:
                logger.warning(f"Failed to fetch price for {competitor_id}: {e}")
                continue
        
        return competitor_prices
    
    def _fetch_competitor_price(self, competitor_id: str, product_category: str) -> float:
        """🟢 GREEN: Mock competitor price fetching"""
        # In production, this would integrate with competitor price APIs
        # For now, return mock data based on competitor
        base_price = 45.0  # Base cable price
        
        competitor_adjustments = {
            "southwire": 0.95,    # 5% lower
            "prysmian": 1.03,     # 3% higher  
            "general_cable": 0.94, # 6% lower
        }
        
        adjustment = competitor_adjustments.get(competitor_id, 1.0)
        return round(base_price * adjustment, 2)
    
    def compare_with_nexans_pricing(self, nexans_price: float, competitor_prices: Dict[str, float]) -> Dict:
        """🟢 GREEN: Compare Nexans pricing with competitors"""
        if not competitor_prices:
            return {
                "price_position": "NO_DATA",
                "competitive_advantage": 0.0,
                "recommendations": ["Gather competitor pricing data"]
            }
        
        competitor_avg = statistics.mean(competitor_prices.values())
        price_difference = nexans_price - competitor_avg
        price_difference_pct = (price_difference / competitor_avg) * 100
        
        # Determine competitive position
        if price_difference_pct > 10:
            position = "PREMIUM"
        elif price_difference_pct > 5:
            position = "ABOVE_MARKET"
        elif price_difference_pct > -5:
            position = "COMPETITIVE"
        elif price_difference_pct > -10:
            position = "BELOW_MARKET"
        else:
            position = "DISCOUNT"
        
        recommendations = []
        if position == "PREMIUM":
            recommendations.append("Consider price reduction or value justification")
        elif position == "BELOW_MARKET":
            recommendations.append("Opportunity for price increase")
        else:
            recommendations.append("Maintain competitive positioning")
        
        return {
            "price_position": position,
            "competitive_advantage": round(price_difference_pct, 2),
            "recommendations": recommendations,
            "competitor_average": round(competitor_avg, 2),
            "nexans_price": nexans_price
        }


class PricingRecommendationEngine:
    """🟢 GREEN: Automated pricing recommendations"""
    
    def __init__(self):
        self.recommendation_rules = {}
        self.market_conditions = {}
        self.pricing_strategies = {}
    
    def generate_recommendations(self, market_context: Dict) -> List[Dict]:
        """🟢 GREEN: Generate pricing recommendations based on market context"""
        recommendations = []
        
        # Price adjustment recommendations
        price_rec = self._generate_price_adjustment_recommendation(market_context)
        if price_rec:
            recommendations.append(price_rec)
        
        # Strategy recommendations
        strategy_recs = self._generate_strategy_recommendations(market_context)
        recommendations.extend(strategy_recs)
        
        return recommendations
    
    def _generate_price_adjustment_recommendation(self, context: Dict) -> Optional[Dict]:
        """🟢 GREEN: Generate price adjustment recommendation"""
        volatility = context.get("volatility_level", "LOW")
        trend = context.get("price_trend", "STABLE")
        demand = context.get("demand_level", "NORMAL")
        competitor_position = context.get("competitor_position", "COMPETITIVE")
        
        # Decision logic for price adjustments
        if volatility == "HIGH" and trend == "INCREASING":
            return {
                "type": "PRICE_ADJUSTMENT",
                "action": "INCREASE",
                "magnitude": 5.0,  # 5% increase
                "reasoning": "High volatility with increasing trend - capitalize on market conditions",
                "urgency": "HIGH"
            }
        elif volatility == "LOW" and trend == "STABLE":
            return {
                "type": "PRICE_ADJUSTMENT", 
                "action": "MAINTAIN",
                "magnitude": 0.0,
                "reasoning": "Stable conditions - maintain current pricing",
                "urgency": "LOW"
            }
        elif competitor_position == "BELOW_MARKET":
            return {
                "type": "PRICE_ADJUSTMENT",
                "action": "INCREASE",
                "magnitude": 3.0,  # 3% increase
                "reasoning": "Below market positioning - opportunity for price increase",
                "urgency": "MEDIUM"
            }
        
        return None
    
    def _generate_strategy_recommendations(self, context: Dict) -> List[Dict]:
        """🟢 GREEN: Generate strategic recommendations"""
        recommendations = []
        
        volatility = context.get("volatility_level", "LOW")
        
        if volatility == "HIGH":
            recommendations.append({
                "type": "STRATEGY",
                "action": "IMPLEMENT_DYNAMIC_PRICING",
                "reasoning": "High volatility requires more responsive pricing",
                "implementation": "Review prices twice daily during volatile periods"
            })
        
        return recommendations
    
    def calculate_optimal_price_adjustment(self, current_price: float, market_factors: Dict) -> float:
        """🟢 GREEN: Calculate optimal price adjustment"""
        adjustment_factor = 1.0
        
        # Apply market factors
        for factor, impact in market_factors.items():
            adjustment_factor += impact
        
        # Cap adjustment at ±15%
        adjustment_factor = max(0.85, min(1.15, adjustment_factor))
        
        optimal_price = current_price * adjustment_factor
        return round(optimal_price, 2)


class MarketTrendAnalyzer:
    """🟢 GREEN: Market trend analysis and prediction"""
    
    def __init__(self):
        self.trend_detection_methods = ["linear_regression", "moving_average"]
        self.historical_data_window = 30  # days
        self.trend_confidence_threshold = 0.7
    
    def analyze_price_trend(self, metal: str, price_history: List[Dict]) -> Dict:
        """🟢 GREEN: Analyze price trend from historical data"""
        if len(price_history) < 3:
            return {
                "trend_direction": "INSUFFICIENT_DATA",
                "trend_strength": "UNKNOWN",
                "confidence_level": 0.0
            }
        
        prices = [p["price"] for p in price_history]
        
        # Simple linear trend analysis
        n = len(prices)
        x = list(range(n))
        
        # Calculate linear regression slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(prices)
        
        numerator = sum((x[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction and strength
        price_range = max(prices) - min(prices)
        relative_slope = abs(slope) / (price_range / n) if price_range > 0 else 0
        
        if slope > 0.01:
            direction = "INCREASING"
        elif slope < -0.01:
            direction = "DECREASING"
        else:
            direction = "STABLE"
        
        if relative_slope > 0.5:
            strength = "STRONG"
        elif relative_slope > 0.2:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Calculate confidence based on data consistency
        price_changes = [prices[i+1] - prices[i] for i in range(n-1)]
        positive_changes = sum(1 for change in price_changes if change > 0)
        
        if direction == "INCREASING":
            confidence = positive_changes / len(price_changes)
        elif direction == "DECREASING":
            confidence = (len(price_changes) - positive_changes) / len(price_changes)
        else:
            confidence = 0.5  # Neutral for stable trend
        
        return {
            "trend_direction": direction,
            "trend_strength": strength,
            "confidence_level": round(confidence, 2),
            "slope": round(slope, 4)
        }
    
    def predict_future_prices(self, metal: str, price_history: List[float], days_ahead: int = 5) -> List[Dict]:
        """🟢 GREEN: Predict future prices based on historical data"""
        if len(price_history) < 3:
            return []
        
        # Simple linear extrapolation
        n = len(price_history)
        x = list(range(n))
        
        # Calculate linear regression parameters
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(price_history)
        
        numerator = sum((x[i] - x_mean) * (price_history[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
            intercept = y_mean
        else:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        
        # Generate predictions
        predictions = []
        for day in range(1, days_ahead + 1):
            future_x = n + day - 1
            predicted_price = intercept + slope * future_x
            
            # Calculate confidence (decreases with distance)
            confidence = max(0.1, 0.9 - (day * 0.1))
            
            predictions.append({
                "day": day,
                "predicted_price": round(max(0, predicted_price), 2),
                "confidence": round(confidence, 2),
                "prediction_date": datetime.now() + timedelta(days=day)
            })
        
        return predictions


class AlertNotificationService:
    """🟢 GREEN: Alert notification and delivery service"""
    
    def __init__(self):
        self.notification_channels = ["email", "slack", "webhook"]
        self.alert_templates = {}
        self.delivery_methods = {}
    
    def send_alert_notification(self, alert: MarketAlert, recipients: List[str], method: str = "email") -> bool:
        """🟢 GREEN: Send alert notification to recipients"""
        try:
            if method == "email":
                return self._send_email(alert, recipients)
            elif method == "slack":
                return self._send_slack(alert, recipients)
            else:
                logger.warning(f"Unsupported notification method: {method}")
                return False
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    def _send_email(self, alert: MarketAlert, recipients: List[str]) -> bool:
        """🟢 GREEN: Send email notification (mock implementation)"""
        # In production, this would integrate with email service
        logger.info(f"Email sent to {recipients} for alert {alert.alert_id}")
        return True
    
    def _send_slack(self, alert: MarketAlert, channels: List[str]) -> bool:
        """🟢 GREEN: Send Slack notification (mock implementation)"""
        # In production, this would integrate with Slack API
        logger.info(f"Slack message sent to {channels} for alert {alert.alert_id}")
        return True
    
    def format_alert_message(self, alert: MarketAlert, template: str = "email") -> str:
        """🟢 GREEN: Format alert message using template"""
        if template == "email":
            return f"""
Market Intelligence Alert - {alert.alert_id}

Alert Type: {alert.alert_type}
Metal: {alert.metal.upper()}  
Current Price: ${alert.current_price}
Severity: {alert.severity}
Timestamp: {alert.timestamp}

Message: {alert.message}

Previous Price: ${alert.previous_price or 'N/A'}
Price Change: {alert.calculate_price_change_percentage():.2f}%

This is an automated alert from Nexans Pricing Intelligence System.
"""
        else:
            return f"[{alert.severity}] {alert.metal.upper()}: {alert.message} (${alert.current_price})"
    
    def batch_send_notifications(self, alerts: List[MarketAlert], recipients: List[str]) -> List[bool]:
        """🟢 GREEN: Send multiple notifications in batch"""
        results = []
        for alert in alerts:
            result = self.send_alert_notification(alert, recipients)
            results.append(result)
        return results


class MarketIntelligenceAgent:
    """🟢 GREEN: Main Market Intelligence Agent orchestrator"""
    
    def __init__(self):
        self.price_monitor = None
        self.volatility_detector = PriceVolatilityDetector()
        self.competitor_tracker = CompetitorPriceTracker()
        self.recommendation_engine = PricingRecommendationEngine()
        self.trend_analyzer = MarketTrendAnalyzer()
        self.notification_service = AlertNotificationService()
        
        # Agent state
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.monitoring_end_time = None
        self.current_alerts = []
        self.price_history = {}
    
    def start_monitoring(self) -> bool:
        """🟢 GREEN: Start market monitoring"""
        try:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.monitoring_end_time = None
            logger.info("Market intelligence monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """🟢 GREEN: Stop market monitoring"""
        try:
            self.is_monitoring = False
            self.monitoring_end_time = datetime.now()
            logger.info("Market intelligence monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def get_market_status(self) -> Dict[str, Any]:
        """🟢 GREEN: Get current market status"""
        try:
            # Get current LME prices
            copper_price = get_lme_copper_price(use_fallback=True)
            aluminum_price = get_lme_aluminum_price(use_fallback=True)
            
            # Analyze volatility (mock data for now)
            recent_copper_data = [
                {"price": copper_price * 0.98, "timestamp": datetime.now() - timedelta(hours=2)},
                {"price": copper_price * 1.01, "timestamp": datetime.now() - timedelta(hours=1)},
                {"price": copper_price, "timestamp": datetime.now()}
            ]
            
            volatility_level = self.volatility_detector.detect_volatility("copper", recent_copper_data)
            
            return {
                "lme_prices": {
                    "copper": copper_price,
                    "aluminum": aluminum_price
                },
                "volatility_level": volatility_level,
                "market_trend": "STABLE",  # Would be calculated from historical data
                "alerts_count": len(self.current_alerts),
                "last_update": datetime.now().isoformat(),
                "monitoring_active": self.is_monitoring
            }
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            raise MarketIntelligenceError(f"Market status error: {e}")
    
    async def process_market_data(self, market_data: Dict) -> Dict[str, Any]:
        """🟢 GREEN: Process market data and generate insights"""
        try:
            # Validate input data
            if not isinstance(market_data.get("copper_price"), (int, float)):
                raise MarketIntelligenceError("Invalid copper price data")
            
            if not isinstance(market_data.get("timestamp"), datetime):
                if "timestamp" not in market_data:
                    market_data["timestamp"] = datetime.now()
                else:
                    raise MarketIntelligenceError("Invalid timestamp data")
            
            # Store price history
            metal = "copper"
            if metal not in self.price_history:
                self.price_history[metal] = []
            
            self.price_history[metal].append({
                "price": market_data["copper_price"],
                "timestamp": market_data["timestamp"]
            })
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.price_history[metal] = [
                p for p in self.price_history[metal] 
                if p["timestamp"] > cutoff_time
            ]
            
            # Analyze volatility
            volatility_analysis = {
                "level": self.volatility_detector.detect_volatility(metal, self.price_history[metal]),
                "metrics": self.volatility_detector.calculate_volatility_metrics(
                    [p["price"] for p in self.price_history[metal]]
                )
            }
            
            # Generate alerts
            alerts = []
            if len(self.price_history[metal]) >= 2:
                current_price = market_data["copper_price"]
                previous_price = self.price_history[metal][-2]["price"]
                
                price_change_pct = abs((current_price - previous_price) / previous_price * 100)
                
                if price_change_pct > 3.0:  # 3% threshold
                    alert_type = "PRICE_SPIKE" if current_price > previous_price else "PRICE_DROP"
                    severity = "HIGH" if price_change_pct > 5.0 else "MEDIUM"
                    
                    alert = MarketAlert(
                        alert_id=f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type=alert_type,
                        metal=metal,
                        current_price=current_price,
                        previous_price=previous_price,
                        severity=severity,
                        message=f"{metal.title()} price {'increased' if current_price > previous_price else 'decreased'} by {price_change_pct:.2f}%"
                    )
                    alerts.append(alert)
                    self.current_alerts.append(alert)
            
            # Generate recommendations
            market_context = {
                "volatility_level": volatility_analysis["level"],
                "price_trend": "INCREASING" if len(self.price_history[metal]) >= 2 and 
                             self.price_history[metal][-1]["price"] > self.price_history[metal][-2]["price"] else "STABLE",
                "competitor_position": "COMPETITIVE",
                "demand_level": "NORMAL",
                "lme_copper_price": market_data["copper_price"],
                "current_nexans_price": 45.83
            }
            
            recommendations = self.recommendation_engine.generate_recommendations(market_context)
            
            return {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "type": alert.alert_type,
                        "metal": alert.metal,
                        "severity": alert.severity,
                        "message": alert.message,
                        "price_change_pct": alert.calculate_price_change_percentage()
                    } for alert in alerts
                ],
                "recommendations": recommendations,
                "volatility_analysis": volatility_analysis,
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market data processing error: {e}")
            raise MarketIntelligenceError(f"Processing error: {e}")


# Export main classes
__all__ = [
    "MarketIntelligenceAgent",
    "MarketAlert",
    "PriceVolatilityDetector",
    "CompetitorPriceTracker", 
    "PricingRecommendationEngine",
    "MarketTrendAnalyzer",
    "AlertNotificationService",
    "MarketIntelligenceError"
]