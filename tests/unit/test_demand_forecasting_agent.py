"""
🔴 RED PHASE - Demand Forecasting Agent Tests
Sprint 3.2: Demand Forecasting Agent para ML prediction y seasonal analysis

TESTS TO WRITE FIRST (RED):
- DemandForecastingAgent core functionality
- ML demand prediction models (ARIMA, Prophet, LSTM)
- Seasonal pattern recognition and analysis
- Inventory optimization alerts
- Market trend correlation analysis
- Demand anomaly detection
- Forecast accuracy validation
- Historical demand data processing

All tests MUST FAIL initially to follow TDD methodology.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import asyncio
import numpy as np
import pandas as pd

# Import will fail initially - that's expected in RED phase
from src.agents.demand_forecasting import (
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


class TestDemandForecastingAgent:
    """🔴 RED: Test Demand Forecasting Agent core functionality"""
    
    def test_demand_forecasting_agent_initialization(self):
        """🔴 RED: Test DemandForecastingAgent can be instantiated"""
        # EXPECT: DemandForecastingAgent class doesn't exist yet
        agent = DemandForecastingAgent()
        assert agent is not None
        assert hasattr(agent, 'forecasting_models')
        assert hasattr(agent, 'seasonal_analyzer')
        assert hasattr(agent, 'inventory_optimizer')
        assert hasattr(agent, 'anomaly_detector')
        assert hasattr(agent, 'accuracy_validator')
        assert hasattr(agent, 'trend_correlator')
    
    def test_agent_load_historical_data(self):
        """🔴 RED: Test agent can load historical demand data"""
        agent = DemandForecastingAgent()
        
        # Mock historical data
        historical_data = [
            {"date": "2024-01-01", "product_id": "540317340", "quantity_sold": 1500, "revenue": 68745.0, "customer_segment": "mining"},
            {"date": "2024-01-02", "product_id": "540317340", "quantity_sold": 1200, "revenue": 54996.0, "customer_segment": "industrial"},
            {"date": "2024-01-03", "product_id": "540317340", "quantity_sold": 1800, "revenue": 82494.0, "customer_segment": "mining"}
        ]
        
        result = agent.load_historical_data(historical_data)
        assert result == True
        assert len(agent.historical_data) == 3
        assert agent.data_loaded == True
    
    def test_agent_train_forecasting_models(self):
        """🔴 RED: Test agent can train multiple forecasting models"""
        agent = DemandForecastingAgent()
        
        # Load sample data first
        sample_data = self._generate_sample_demand_data(days=90)
        agent.load_historical_data(sample_data)
        
        # Should be able to train multiple models
        training_result = agent.train_forecasting_models(
            models=["arima", "prophet", "lstm"],
            validation_split=0.2
        )
        
        assert training_result["success"] == True
        assert "arima" in training_result["trained_models"]
        assert "prophet" in training_result["trained_models"]
        assert "lstm" in training_result["trained_models"]
        assert all(acc > 0.7 for acc in training_result["model_accuracies"].values())
    
    def test_agent_generate_demand_forecast(self):
        """🔴 RED: Test agent can generate demand forecasts"""
        agent = DemandForecastingAgent()
        
        # Setup trained agent
        sample_data = self._generate_sample_demand_data(days=90)
        agent.load_historical_data(sample_data)
        agent.train_forecasting_models(models=["arima"])
        
        # Generate forecast for next 30 days
        forecast = agent.generate_demand_forecast(
            product_id="540317340",
            days_ahead=30,
            confidence_level=0.95
        )
        
        assert isinstance(forecast, dict)
        assert "forecasted_demand" in forecast
        assert "confidence_intervals" in forecast
        assert "seasonal_adjustments" in forecast
        assert len(forecast["forecasted_demand"]) == 30
    
    def test_agent_detect_seasonal_patterns(self):
        """🔴 RED: Test agent can detect seasonal demand patterns"""
        agent = DemandForecastingAgent()
        
        # Load data with seasonal patterns
        seasonal_data = self._generate_seasonal_demand_data(months=12)
        agent.load_historical_data(seasonal_data)
        
        patterns = agent.detect_seasonal_patterns(product_id="540317340")
        
        assert "weekly_pattern" in patterns
        assert "monthly_pattern" in patterns
        assert "quarterly_pattern" in patterns
        assert patterns["seasonality_strength"] > 0.3
        assert len(patterns["seasonal_peaks"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_real_time_monitoring(self):
        """🔴 RED: Test agent can monitor demand in real-time"""
        agent = DemandForecastingAgent()
        
        # Setup monitoring
        monitoring_config = {
            "update_interval_minutes": 60,
            "alert_threshold_deviation": 0.25,
            "products_to_monitor": ["540317340", "540317341"]
        }
        
        result = await agent.start_real_time_monitoring(monitoring_config)
        assert result["monitoring_active"] == True
        assert len(result["monitored_products"]) == 2
        assert result["next_update"] is not None
    
    def _generate_sample_demand_data(self, days: int) -> list:
        """Helper to generate sample demand data"""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Add some realistic demand variation
            base_demand = 1500
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
            random_noise = 1 + np.random.normal(0, 0.1)
            
            demand = int(base_demand * seasonal_factor * random_noise)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "540317340",
                "quantity_sold": demand,
                "revenue": demand * 45.83,
                "customer_segment": np.random.choice(["mining", "industrial", "utility"])
            })
        
        return data
    
    def _generate_seasonal_demand_data(self, months: int) -> list:
        """Helper to generate data with clear seasonal patterns"""
        data = []
        base_date = datetime.now() - timedelta(days=30*months)
        
        for i in range(30 * months):
            date = base_date + timedelta(days=i)
            day_of_year = date.timetuple().tm_yday
            
            # Strong seasonal pattern - higher demand in mining season (Nov-Mar)
            seasonal_multiplier = 1.5 if day_of_year < 90 or day_of_year > 300 else 1.0
            
            # Weekly pattern - lower demand on weekends
            weekly_multiplier = 0.7 if date.weekday() >= 5 else 1.0
            
            base_demand = 1500
            final_demand = int(base_demand * seasonal_multiplier * weekly_multiplier)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "540317340",
                "quantity_sold": final_demand,
                "revenue": final_demand * 45.83,
                "customer_segment": "mining" if seasonal_multiplier > 1.0 else "industrial"
            })
        
        return data


class TestDemandForecast:
    """🔴 RED: Test Demand Forecast data structure"""
    
    def test_demand_forecast_creation(self):
        """🔴 RED: Test DemandForecast can be created with required fields"""
        forecast = DemandForecast(
            forecast_id="FORECAST_001",
            product_id="540317340",
            forecast_date=datetime.now(),
            forecast_horizon_days=30,
            predicted_quantities=[1500, 1520, 1480, 1600, 1550],
            confidence_intervals=[(1400, 1600), (1420, 1620), (1380, 1580), (1500, 1700), (1450, 1650)],
            model_used="arima",
            accuracy_score=0.85,
            seasonal_adjustments={"weekly_factor": 1.2, "monthly_factor": 1.1}
        )
        
        assert forecast.forecast_id == "FORECAST_001"
        assert forecast.product_id == "540317340"
        assert forecast.forecast_horizon_days == 30
        assert len(forecast.predicted_quantities) == 5
        assert forecast.model_used == "arima"
        assert forecast.accuracy_score == 0.85
    
    def test_demand_forecast_validation(self):
        """🔴 RED: Test DemandForecast validates input data"""
        # Should raise error for negative quantities
        with pytest.raises(ValueError):
            DemandForecast(
                forecast_id="FORECAST_002",
                product_id="540317340",
                forecast_date=datetime.now(),
                forecast_horizon_days=30,
                predicted_quantities=[-100, 1500, 1600],  # Negative quantity
                confidence_intervals=[(0, 200), (1400, 1600), (1500, 1700)],
                model_used="prophet",
                accuracy_score=0.75
            )
        
        # Should raise error for invalid accuracy score
        with pytest.raises(ValueError):
            DemandForecast(
                forecast_id="FORECAST_003",
                product_id="540317340",
                forecast_date=datetime.now(),
                forecast_horizon_days=30,
                predicted_quantities=[1500, 1600],
                confidence_intervals=[(1400, 1600), (1500, 1700)],
                model_used="lstm",
                accuracy_score=1.5  # Invalid score > 1.0
            )
    
    def test_demand_forecast_statistics(self):
        """🔴 RED: Test DemandForecast calculates statistics correctly"""
        forecast = DemandForecast(
            forecast_id="FORECAST_004",
            product_id="540317340",
            forecast_date=datetime.now(),
            forecast_horizon_days=30,
            predicted_quantities=[1500, 1600, 1400, 1700, 1550],
            confidence_intervals=[(1400, 1600), (1500, 1700), (1300, 1500), (1600, 1800), (1450, 1650)],
            model_used="arima",
            accuracy_score=0.88
        )
        
        mean_demand = forecast.calculate_mean_demand()
        assert mean_demand == 1550.0  # (1500+1600+1400+1700+1550)/5
        
        total_demand = forecast.calculate_total_demand()
        assert total_demand == 7750  # Sum of all quantities
        
        demand_variance = forecast.calculate_demand_variance()
        assert demand_variance > 0  # Should have some variance


class TestSeasonalPatternAnalyzer:
    """🔴 RED: Test Seasonal Pattern Analysis"""
    
    def test_seasonal_analyzer_initialization(self):
        """🔴 RED: Test SeasonalPatternAnalyzer initialization"""
        analyzer = SeasonalPatternAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'seasonal_methods')
        assert hasattr(analyzer, 'pattern_cache')
        assert hasattr(analyzer, 'min_data_points')
    
    def test_detect_weekly_seasonality(self):
        """🔴 RED: Test detection of weekly seasonal patterns"""
        analyzer = SeasonalPatternAnalyzer()
        
        # Generate data with clear weekly pattern
        dates = pd.date_range(start='2024-01-01', periods=84, freq='D')  # 12 weeks
        demand_data = []
        
        for i, date in enumerate(dates):
            # Higher demand on weekdays, lower on weekends
            if date.weekday() < 5:  # Monday-Friday
                base_demand = 1500
            else:  # Weekend
                base_demand = 800
            
            demand_data.append({
                "date": date,
                "quantity": base_demand + np.random.normal(0, 50)
            })
        
        pattern = analyzer.detect_weekly_seasonality(demand_data)
        
        assert pattern["seasonality_detected"] == True
        assert pattern["pattern_strength"] > 0.6
        assert len(pattern["weekly_factors"]) == 7
        # Weekdays should have higher factors than weekends
        assert max(pattern["weekly_factors"][:5]) > max(pattern["weekly_factors"][5:])
    
    def test_detect_monthly_seasonality(self):
        """🔴 RED: Test detection of monthly seasonal patterns"""
        analyzer = SeasonalPatternAnalyzer()
        
        # Generate 24 months of data with seasonal pattern
        dates = pd.date_range(start='2022-01-01', periods=730, freq='D')  # 2 years
        demand_data = []
        
        for date in enumerate(dates):
            # Higher demand in Q4 (October-December)
            month = date[1].month
            if month in [10, 11, 12]:
                base_demand = 2000  # High season
            elif month in [6, 7, 8]:
                base_demand = 1000  # Low season
            else:
                base_demand = 1500  # Normal season
            
            demand_data.append({
                "date": date[1],
                "quantity": base_demand + np.random.normal(0, 100)
            })
        
        pattern = analyzer.detect_monthly_seasonality(demand_data)
        
        assert pattern["seasonality_detected"] == True
        assert pattern["pattern_strength"] > 0.5
        assert len(pattern["monthly_factors"]) == 12
        # Q4 months should have higher factors
        assert pattern["monthly_factors"][9] > pattern["monthly_factors"][5]  # Oct > Jun
    
    def test_identify_seasonal_peaks(self):
        """🔴 RED: Test identification of seasonal demand peaks"""
        analyzer = SeasonalPatternAnalyzer()
        
        # Data with clear peaks in March and September
        demand_data = self._generate_bimodal_seasonal_data()
        
        peaks = analyzer.identify_seasonal_peaks(demand_data, threshold=0.2)
        
        assert len(peaks) >= 2
        assert any(peak["month"] == 3 for peak in peaks)  # March peak
        assert any(peak["month"] == 9 for peak in peaks)  # September peak
        assert all(peak["intensity"] > 0.2 for peak in peaks)
    
    def _generate_bimodal_seasonal_data(self) -> list:
        """Helper to generate data with peaks in March and September"""
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        demand_data = []
        
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            
            # Peaks around day 82 (March) and day 244 (September)
            march_peak = 1 + 0.5 * np.exp(-((day_of_year - 82) ** 2) / (2 * 15 ** 2))
            sept_peak = 1 + 0.4 * np.exp(-((day_of_year - 244) ** 2) / (2 * 20 ** 2))
            
            base_demand = 1500
            seasonal_demand = int(base_demand * march_peak * sept_peak)
            
            demand_data.append({
                "date": date,
                "quantity": seasonal_demand
            })
        
        return demand_data


class TestForecastingModel:
    """🔴 RED: Test Forecasting Model implementations"""
    
    def test_arima_forecasting_model(self):
        """🔴 RED: Test ARIMA forecasting model"""
        model = ForecastingModel(model_type="arima")
        
        # Generate training data
        training_data = self._generate_time_series_data(100)
        
        # Train the model
        training_result = model.train(training_data, validation_split=0.2)
        assert training_result["success"] == True
        assert training_result["model_params"]["p"] >= 0
        assert training_result["model_params"]["d"] >= 0
        assert training_result["model_params"]["q"] >= 0
        
        # Make predictions
        predictions = model.predict(steps_ahead=10)
        assert len(predictions) == 10
        assert all(pred > 0 for pred in predictions)
    
    def test_prophet_forecasting_model(self):
        """🔴 RED: Test Prophet forecasting model"""
        model = ForecastingModel(model_type="prophet")
        
        # Generate training data with trend and seasonality
        training_data = self._generate_time_series_data(200, with_trend=True)
        
        # Train the model
        training_result = model.train(training_data)
        assert training_result["success"] == True
        assert "changepoints" in training_result["model_params"]
        assert "seasonality_components" in training_result["model_params"]
        
        # Make predictions with confidence intervals
        predictions = model.predict(steps_ahead=30, include_confidence=True)
        assert len(predictions["forecast"]) == 30
        assert len(predictions["lower_bound"]) == 30
        assert len(predictions["upper_bound"]) == 30
    
    def test_lstm_forecasting_model(self):
        """🔴 RED: Test LSTM neural network forecasting model"""
        model = ForecastingModel(model_type="lstm")
        
        # Generate training data
        training_data = self._generate_time_series_data(300)
        
        # Configure LSTM parameters
        lstm_config = {
            "sequence_length": 30,
            "hidden_units": 50,
            "epochs": 10,
            "batch_size": 32
        }
        
        # Train the model
        training_result = model.train(training_data, model_config=lstm_config)
        assert training_result["success"] == True
        assert training_result["training_loss"] < 1.0
        assert training_result["validation_loss"] < 1.0
        
        # Make predictions
        predictions = model.predict(steps_ahead=15)
        assert len(predictions) == 15
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_model_accuracy_validation(self):
        """🔴 RED: Test model accuracy validation"""
        model = ForecastingModel(model_type="arima")
        
        # Generate test data
        full_data = self._generate_time_series_data(120)
        train_data = full_data[:100]
        test_data = full_data[100:]
        
        # Train model
        model.train(train_data)
        
        # Validate accuracy
        validation_result = model.validate_accuracy(test_data)
        
        assert "mae" in validation_result
        assert "rmse" in validation_result
        assert "mape" in validation_result
        assert validation_result["mae"] > 0
        assert validation_result["rmse"] > 0
        assert 0 <= validation_result["mape"] <= 100
    
    def _generate_time_series_data(self, length: int, with_trend: bool = False) -> list:
        """Helper to generate time series data for testing"""
        data = []
        start_date = datetime(2023, 1, 1)
        
        for i in range(length):
            date = start_date + timedelta(days=i)
            
            # Base demand
            base_value = 1500
            
            # Add trend if requested
            if with_trend:
                base_value += i * 2  # Linear growth
            
            # Add seasonality (weekly pattern)
            seasonal_component = 200 * np.sin(2 * np.pi * i / 7)
            
            # Add noise
            noise = np.random.normal(0, 50)
            
            final_value = max(0, base_value + seasonal_component + noise)
            
            data.append({
                "date": date,
                "value": final_value
            })
        
        return data


class TestInventoryOptimizer:
    """🔴 RED: Test Inventory Optimization"""
    
    def test_inventory_optimizer_initialization(self):
        """🔴 RED: Test InventoryOptimizer initialization"""
        optimizer = InventoryOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'optimization_algorithms')
        assert hasattr(optimizer, 'inventory_constraints')
        assert hasattr(optimizer, 'cost_parameters')
    
    def test_calculate_optimal_inventory_levels(self):
        """🔴 RED: Test optimal inventory level calculation"""
        optimizer = InventoryOptimizer()
        
        # Setup inventory parameters
        demand_forecast = {
            "daily_demand": [1500, 1520, 1480, 1600, 1550, 1520, 1490],
            "demand_variance": 2500,
            "seasonal_adjustments": {"peak_factor": 1.3, "trough_factor": 0.7}
        }
        
        inventory_params = {
            "holding_cost_per_unit": 2.5,  # USD per meter per month
            "ordering_cost": 500.0,  # USD per order
            "stockout_penalty": 25.0,  # USD per meter
            "lead_time_days": 14,
            "service_level": 0.95
        }
        
        optimal_levels = optimizer.calculate_optimal_inventory_levels(
            demand_forecast, inventory_params
        )
        
        assert "reorder_point" in optimal_levels
        assert "economic_order_quantity" in optimal_levels
        assert "safety_stock" in optimal_levels
        assert "max_inventory_level" in optimal_levels
        
        assert optimal_levels["reorder_point"] > 0
        assert optimal_levels["economic_order_quantity"] > 0
        assert optimal_levels["safety_stock"] > 0
    
    def test_generate_inventory_alerts(self):
        """🔴 RED: Test inventory alert generation"""
        optimizer = InventoryOptimizer()
        
        # Current inventory status
        current_inventory = {
            "product_id": "540317340",
            "current_stock": 800,  # Low stock
            "reorder_point": 1200,
            "max_level": 5000,
            "daily_usage_rate": 150
        }
        
        # Demand forecast showing increasing demand
        upcoming_demand = {
            "next_7_days": [160, 170, 165, 180, 175, 160, 155],
            "next_30_days_total": 4800,
            "confidence_level": 0.92
        }
        
        alerts = optimizer.generate_inventory_alerts(current_inventory, upcoming_demand)
        
        assert len(alerts) > 0
        assert any(alert["alert_type"] == "LOW_STOCK" for alert in alerts)
        assert any(alert["urgency"] == "HIGH" for alert in alerts)
        assert all("recommended_action" in alert for alert in alerts)
    
    def test_inventory_cost_optimization(self):
        """🔴 RED: Test inventory cost optimization"""
        optimizer = InventoryOptimizer()
        
        # Cost structure
        cost_params = {
            "holding_cost_rate": 0.15,  # 15% annual holding cost
            "unit_cost": 45.83,  # USD per meter
            "ordering_cost": 750.0,  # USD per order
            "stockout_cost_multiplier": 3.0  # 3x unit cost for stockouts
        }
        
        # Demand parameters
        demand_params = {
            "annual_demand": 150000,  # meters per year
            "demand_variability": 0.25,  # 25% coefficient of variation
            "lead_time_days": 21
        }
        
        optimization_result = optimizer.optimize_inventory_costs(cost_params, demand_params)
        
        assert "optimal_order_quantity" in optimization_result
        assert "total_annual_cost" in optimization_result
        assert "cost_breakdown" in optimization_result
        
        cost_breakdown = optimization_result["cost_breakdown"]
        assert "holding_costs" in cost_breakdown
        assert "ordering_costs" in cost_breakdown
        assert "stockout_costs" in cost_breakdown
        
        # Total cost should be sum of components
        expected_total = sum(cost_breakdown.values())
        assert abs(optimization_result["total_annual_cost"] - expected_total) < 0.01


class TestDemandAnomalyDetector:
    """🔴 RED: Test Demand Anomaly Detection"""
    
    def test_anomaly_detector_initialization(self):
        """🔴 RED: Test DemandAnomalyDetector initialization"""
        detector = DemandAnomalyDetector()
        assert detector is not None
        assert hasattr(detector, 'detection_methods')
        assert hasattr(detector, 'anomaly_thresholds')
        assert hasattr(detector, 'historical_baselines')
    
    def test_detect_demand_spikes(self):
        """🔴 RED: Test detection of demand spikes"""
        detector = DemandAnomalyDetector()
        
        # Generate normal demand data with a spike
        normal_demand = [1500] * 20  # 20 days of normal demand
        spike_data = normal_demand + [3500] + normal_demand  # One spike in the middle
        
        demand_data = []
        base_date = datetime.now() - timedelta(days=len(spike_data))
        
        for i, demand in enumerate(spike_data):
            demand_data.append({
                "date": base_date + timedelta(days=i),
                "quantity": demand,
                "product_id": "540317340"
            })
        
        anomalies = detector.detect_demand_spikes(demand_data, threshold_multiplier=2.0)
        
        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "DEMAND_SPIKE"
        assert anomalies[0]["severity"] in ["HIGH", "CRITICAL"]
        assert anomalies[0]["value"] == 3500
    
    def test_detect_demand_drops(self):
        """🔴 RED: Test detection of demand drops"""
        detector = DemandAnomalyDetector()
        
        # Generate normal demand data with a drop
        normal_demand = [1500] * 15
        drop_data = normal_demand + [300] + normal_demand  # Significant drop
        
        demand_data = []
        base_date = datetime.now() - timedelta(days=len(drop_data))
        
        for i, demand in enumerate(drop_data):
            demand_data.append({
                "date": base_date + timedelta(days=i),
                "quantity": demand,
                "product_id": "540317340"
            })
        
        anomalies = detector.detect_demand_drops(demand_data, threshold_multiplier=2.5)
        
        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "DEMAND_DROP"
        assert anomalies[0]["severity"] in ["MEDIUM", "HIGH"]
        assert anomalies[0]["value"] == 300
    
    def test_analyze_demand_patterns(self):
        """🔴 RED: Test demand pattern analysis for anomalies"""
        detector = DemandAnomalyDetector()
        
        # Generate data with pattern change
        old_pattern = [1500 + 200 * np.sin(2 * np.pi * i / 7) for i in range(30)]  # Weekly pattern
        new_pattern = [2000 + 100 * np.sin(2 * np.pi * i / 3) for i in range(20)]  # Different pattern
        
        combined_data = old_pattern + new_pattern
        
        demand_data = []
        base_date = datetime.now() - timedelta(days=len(combined_data))
        
        for i, demand in enumerate(combined_data):
            demand_data.append({
                "date": base_date + timedelta(days=i),
                "quantity": max(0, int(demand)),
                "product_id": "540317340"
            })
        
        pattern_analysis = detector.analyze_demand_patterns(demand_data)
        
        assert "pattern_changes" in pattern_analysis
        assert "change_points" in pattern_analysis
        assert "pattern_stability" in pattern_analysis
        
        # Should detect a significant pattern change
        assert len(pattern_analysis["change_points"]) > 0
        assert pattern_analysis["pattern_stability"] < 0.8


class TestForecastAccuracyValidator:
    """🔴 RED: Test Forecast Accuracy Validation"""
    
    def test_accuracy_validator_initialization(self):
        """🔴 RED: Test ForecastAccuracyValidator initialization"""
        validator = ForecastAccuracyValidator()
        assert validator is not None
        assert hasattr(validator, 'accuracy_metrics')
        assert hasattr(validator, 'validation_methods')
        assert hasattr(validator, 'benchmark_models')
    
    def test_validate_forecast_accuracy(self):
        """🔴 RED: Test forecast accuracy validation"""
        validator = ForecastAccuracyValidator()
        
        # Mock forecast vs actual data
        forecasted_values = [1500, 1520, 1480, 1600, 1550, 1520, 1490]
        actual_values = [1480, 1535, 1465, 1580, 1570, 1505, 1475]
        
        accuracy_metrics = validator.validate_forecast_accuracy(forecasted_values, actual_values)
        
        assert "mean_absolute_error" in accuracy_metrics
        assert "root_mean_square_error" in accuracy_metrics
        assert "mean_absolute_percentage_error" in accuracy_metrics
        assert "symmetric_mean_absolute_percentage_error" in accuracy_metrics
        
        assert accuracy_metrics["mean_absolute_error"] > 0
        assert accuracy_metrics["root_mean_square_error"] > 0
        assert 0 <= accuracy_metrics["mean_absolute_percentage_error"] <= 100
    
    def test_cross_validation_accuracy(self):
        """🔴 RED: Test cross-validation accuracy assessment"""
        validator = ForecastAccuracyValidator()
        
        # Generate time series data for cross-validation
        time_series_data = []
        base_date = datetime.now() - timedelta(days=100)
        
        for i in range(100):
            value = 1500 + 200 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 50)
            time_series_data.append({
                "date": base_date + timedelta(days=i),
                "value": max(0, value)
            })
        
        # Cross-validation with different fold configurations
        cv_results = validator.cross_validate_accuracy(
            time_series_data,
            model_type="arima",
            cv_folds=5,
            forecast_horizon=7
        )
        
        assert "cv_scores" in cv_results
        assert "mean_accuracy" in cv_results
        assert "accuracy_std" in cv_results
        assert len(cv_results["cv_scores"]) == 5
        assert 0 <= cv_results["mean_accuracy"] <= 1.0
    
    def test_benchmark_model_comparison(self):
        """🔴 RED: Test comparison against benchmark models"""
        validator = ForecastAccuracyValidator()
        
        # Historical data for benchmarking
        historical_data = []
        for i in range(60):
            historical_data.append({
                "date": datetime.now() - timedelta(days=60-i),
                "actual_demand": 1500 + np.random.normal(0, 100)
            })
        
        # Test model predictions
        test_predictions = [1520, 1480, 1600, 1550, 1490, 1530, 1470]
        test_actuals = [1500, 1485, 1580, 1565, 1475, 1515, 1460]
        
        benchmark_comparison = validator.compare_with_benchmarks(
            test_predictions,
            test_actuals,
            historical_data,
            benchmark_models=["naive", "seasonal_naive", "linear_trend"]
        )
        
        assert "model_performance" in benchmark_comparison
        assert "benchmark_performances" in benchmark_comparison
        assert "relative_improvement" in benchmark_comparison
        
        for benchmark in ["naive", "seasonal_naive", "linear_trend"]:
            assert benchmark in benchmark_comparison["benchmark_performances"]


class TestDemandForecastingIntegration:
    """🔴 RED: Test Demand Forecasting Agent integration"""
    
    @pytest.mark.asyncio
    async def test_full_demand_forecasting_workflow(self):
        """🔴 RED: Test complete demand forecasting workflow"""
        agent = DemandForecastingAgent()
        
        # Load historical data
        historical_data = self._generate_comprehensive_demand_data(months=12)
        agent.load_historical_data(historical_data)
        
        # Train multiple models
        training_result = agent.train_forecasting_models(
            models=["arima", "prophet"],
            validation_split=0.2
        )
        assert training_result["success"] == True
        
        # Generate forecast
        forecast = agent.generate_demand_forecast(
            product_id="540317340",
            days_ahead=30,
            confidence_level=0.95
        )
        
        # Detect seasonal patterns
        patterns = agent.detect_seasonal_patterns(product_id="540317340")
        
        # Optimize inventory
        inventory_recommendations = agent.optimize_inventory_levels(
            product_id="540317340",
            current_stock=2000,
            target_service_level=0.95
        )
        
        # Validate all components
        assert len(forecast["forecasted_demand"]) == 30
        assert patterns["seasonality_strength"] >= 0
        assert "reorder_point" in inventory_recommendations
        assert "economic_order_quantity" in inventory_recommendations
    
    def test_demand_forecasting_error_handling(self):
        """🔴 RED: Test error handling in demand forecasting"""
        agent = DemandForecastingAgent()
        
        # Test with insufficient data
        insufficient_data = [{"date": "2024-01-01", "quantity": 1500}]  # Only 1 data point
        
        with pytest.raises(DemandForecastingError):
            agent.load_historical_data(insufficient_data)
        
        # Test with invalid product ID
        agent.load_historical_data(self._generate_comprehensive_demand_data(months=3))
        
        with pytest.raises(DemandForecastingError):
            agent.generate_demand_forecast(
                product_id="INVALID_PRODUCT",
                days_ahead=30
            )
    
    def test_demand_forecasting_performance(self):
        """🔴 RED: Test demand forecasting performance requirements"""
        agent = DemandForecastingAgent()
        
        # Load substantial data
        large_dataset = self._generate_comprehensive_demand_data(months=24)
        
        start_time = datetime.now()
        agent.load_historical_data(large_dataset)
        loading_time = (datetime.now() - start_time).total_seconds()
        
        # Data loading should be fast
        assert loading_time < 5.0  # Should load within 5 seconds
        
        # Training should complete within reasonable time
        start_time = datetime.now()
        training_result = agent.train_forecasting_models(models=["arima"])
        training_time = (datetime.now() - start_time).total_seconds()
        
        assert training_time < 30.0  # Should train within 30 seconds
        assert training_result["success"] == True
    
    def _generate_comprehensive_demand_data(self, months: int) -> list:
        """Helper to generate comprehensive demand data for testing"""
        data = []
        base_date = datetime.now() - timedelta(days=30*months)
        
        for i in range(30 * months):
            date = base_date + timedelta(days=i)
            
            # Multiple seasonality patterns
            day_of_year = date.timetuple().tm_yday
            weekly_pattern = 1 + 0.3 * np.sin(2 * np.pi * i / 7)
            monthly_pattern = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            # Mining season effect (higher demand Nov-Mar)
            if day_of_year < 90 or day_of_year > 300:
                seasonal_multiplier = 1.4
                segment = "mining"
            else:
                seasonal_multiplier = 1.0
                segment = "industrial"
            
            # Base demand with growth trend
            base_demand = 1500 + (i * 0.5)  # Slight growth over time
            
            # Apply all patterns
            final_demand = int(base_demand * weekly_pattern * monthly_pattern * seasonal_multiplier)
            final_demand = max(100, final_demand + np.random.normal(0, 75))  # Add noise
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": "540317340",
                "quantity_sold": final_demand,
                "revenue": final_demand * 45.83,
                "customer_segment": segment,
                "region": "chile_north" if segment == "mining" else "chile_central"
            })
        
        return data