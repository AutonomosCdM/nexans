"""
🟢 GREEN PHASE - Agents Package Init
Sprint 3.1-3.3: Intelligent Agents package initialization

AGENTS STRUCTURE:
✅ MarketIntelligenceAgent: LME monitoring y price alerts
✅ DemandForecastingAgent: ML prediction y seasonal analysis  
✅ QuoteGenerationAgent: Automated quote generation
"""

from .market_intelligence import (
    MarketIntelligenceAgent,
    MarketAlert,
    PriceVolatilityDetector,
    CompetitorPriceTracker,
    PricingRecommendationEngine,
    MarketTrendAnalyzer,
    AlertNotificationService,
    MarketIntelligenceError
)

from .demand_forecasting import (
    DemandForecastingAgent,
    DemandForecast,
    SeasonalPatternAnalyzer,
    InventoryOptimizer,
    DemandAnomalyDetector,
    ForecastAccuracyValidator,
    DemandTrendCorrelator,
    ForecastingModel,
    DemandForecastingError
)

from .quote_generation import (
    QuoteGenerationAgent,
    AutomatedQuote,
    CustomerPreferenceLearner,
    QuoteOptimizer,
    BundleQuoteGenerator,
    QuoteTemplateManager,
    CustomerInteractionAnalyzer,
    DynamicPricingIntegrator,
    QuoteGenerationError
)

__all__ = [
    # Market Intelligence Agent
    "MarketIntelligenceAgent",
    "MarketAlert", 
    "PriceVolatilityDetector",
    "CompetitorPriceTracker",
    "PricingRecommendationEngine",
    "MarketTrendAnalyzer",
    "AlertNotificationService",
    "MarketIntelligenceError",
    
    # Demand Forecasting Agent
    "DemandForecastingAgent",
    "DemandForecast",
    "SeasonalPatternAnalyzer",
    "InventoryOptimizer", 
    "DemandAnomalyDetector",
    "ForecastAccuracyValidator",
    "DemandTrendCorrelator",
    "ForecastingModel",
    "DemandForecastingError",
    
    # Quote Generation Agent
    "QuoteGenerationAgent",
    "AutomatedQuote",
    "CustomerPreferenceLearner",
    "QuoteOptimizer",
    "BundleQuoteGenerator",
    "QuoteTemplateManager",
    "CustomerInteractionAnalyzer",
    "DynamicPricingIntegrator",
    "QuoteGenerationError"
]