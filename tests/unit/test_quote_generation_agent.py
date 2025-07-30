"""
🔴 RED PHASE - Quote Generation Agent Tests
Sprint 3.3: Quote Generation Agent para automated quote generation

TESTS TO WRITE FIRST (RED):
- QuoteGenerationAgent core functionality
- Automated quote generation from customer requirements
- Customer preference learning and adaptation
- Quote optimization strategies
- Multi-product bundle quoting
- Dynamic pricing integration
- Quote template management
- Customer interaction history analysis

All tests MUST FAIL initially to follow TDD methodology.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import asyncio
import uuid

# Import will fail initially - that's expected in RED phase
from src.agents.quote_generation import (
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


class TestQuoteGenerationAgent:
    """🔴 RED: Test Quote Generation Agent core functionality"""
    
    def test_quote_generation_agent_initialization(self):
        """🔴 RED: Test QuoteGenerationAgent can be instantiated"""
        # EXPECT: QuoteGenerationAgent class doesn't exist yet
        agent = QuoteGenerationAgent()
        assert agent is not None
        assert hasattr(agent, 'preference_learner')
        assert hasattr(agent, 'quote_optimizer')
        assert hasattr(agent, 'bundle_generator')
        assert hasattr(agent, 'template_manager')
        assert hasattr(agent, 'interaction_analyzer')
        assert hasattr(agent, 'pricing_integrator')
    
    def test_agent_generate_automated_quote(self):
        """🔴 RED: Test agent can generate automated quotes"""
        agent = QuoteGenerationAgent()
        
        # Customer requirements
        quote_request = {
            "customer_id": "CODELCO_001",
            "customer_segment": "mining",
            "products_requested": [
                {
                    "product_id": "540317340",
                    "quantity_meters": 2500,
                    "delivery_location": "chile_north",
                    "delivery_deadline": "2024-12-15",
                    "special_requirements": ["fire_resistant", "low_smoke"]
                }
            ],
            "budget_range": {"min": 100000, "max": 150000},
            "request_date": datetime.now(),
            "urgency": "normal"
        }
        
        # Generate quote
        quote = agent.generate_automated_quote(quote_request)
        
        assert isinstance(quote, dict)
        assert "quote_id" in quote
        assert "customer_id" in quote
        assert "total_price" in quote
        assert "line_items" in quote
        assert "delivery_estimate" in quote
        assert "validity_period" in quote
        assert len(quote["line_items"]) > 0
    
    def test_agent_learn_customer_preferences(self):
        """🔴 RED: Test agent can learn from customer behavior"""
        agent = QuoteGenerationAgent()
        
        # Historical customer data
        customer_history = [
            {
                "quote_id": "Q001",
                "customer_id": "CODELCO_001",
                "products": ["540317340"],
                "total_value": 125000,
                "accepted": True,
                "acceptance_time_days": 3,
                "feedback": "Good quality, competitive price"
            },
            {
                "quote_id": "Q002", 
                "customer_id": "CODELCO_001",
                "products": ["540317341"],
                "total_value": 180000,
                "accepted": False,
                "feedback": "Price too high"
            }
        ]
        
        # Learn preferences
        preferences = agent.learn_customer_preferences("CODELCO_001", customer_history)
        
        assert "price_sensitivity" in preferences
        assert "preferred_products" in preferences
        assert "acceptance_patterns" in preferences
        assert "value_thresholds" in preferences
        assert preferences["price_sensitivity"] >= 0
    
    def test_agent_optimize_quote_strategy(self):
        """🔴 RED: Test agent can optimize quote strategies"""
        agent = QuoteGenerationAgent()
        
        # Quote context
        quote_context = {
            "customer_segment": "mining",
            "historical_win_rate": 0.65,
            "competitive_pressure": "high",
            "market_conditions": "volatile",
            "customer_relationship": "long_term",
            "quote_complexity": "standard"
        }
        
        # Current quote details
        base_quote = {
            "total_price": 125000,
            "margin_percentage": 25.0,
            "delivery_days": 30,
            "payment_terms": "net_45",
            "warranty_months": 24
        }
        
        # Optimize
        optimized_quote = agent.optimize_quote_strategy(base_quote, quote_context)
        
        assert "optimized_price" in optimized_quote
        assert "recommended_margin" in optimized_quote
        assert "strategy_adjustments" in optimized_quote
        assert "win_probability" in optimized_quote
        assert 0 <= optimized_quote["win_probability"] <= 1
    
    def test_agent_generate_bundle_quote(self):
        """🔴 RED: Test agent can generate multi-product bundle quotes"""
        agent = QuoteGenerationAgent()
        
        # Bundle request
        bundle_request = {
            "customer_id": "ENGIE_CHILE",
            "project_name": "Solar Plant Expansion",
            "products": [
                {"product_id": "540317340", "quantity": 1500, "application": "distribution"},
                {"product_id": "540317341", "quantity": 800, "application": "transmission"},  
                {"product_id": "540317342", "quantity": 2000, "application": "grounding"}
            ],
            "delivery_schedule": {
                "phase_1": "2024-10-15",
                "phase_2": "2024-11-30", 
                "phase_3": "2024-12-31"
            },
            "bundle_discount_expected": True
        }
        
        # Generate bundle quote
        bundle_quote = agent.generate_bundle_quote(bundle_request)
        
        assert "bundle_id" in bundle_quote
        assert "individual_prices" in bundle_quote
        assert "bundle_discount" in bundle_quote
        assert "total_bundle_price" in bundle_quote
        assert "savings_amount" in bundle_quote
        assert bundle_quote["bundle_discount"] > 0
        assert bundle_quote["total_bundle_price"] < sum(bundle_quote["individual_prices"])
    
    @pytest.mark.asyncio
    async def test_agent_real_time_quote_adjustment(self):
        """🔴 RED: Test agent can adjust quotes in real-time"""
        agent = QuoteGenerationAgent()
        
        # Base quote
        base_quote_id = "Q12345"
        
        # Market changes requiring adjustment
        market_updates = {
            "lme_copper_change": 0.08,  # 8% increase
            "competitor_pricing": {"average_change": -0.02},  # 2% decrease
            "demand_surge": {"product_540317340": 1.15},  # 15% demand increase
            "supply_constraint": {"lead_time_extension": 7}  # 7 extra days
        }
        
        # Adjust quote
        adjusted_quote = await agent.adjust_quote_real_time(base_quote_id, market_updates)
        
        assert "original_quote_id" in adjusted_quote
        assert "adjusted_price" in adjusted_quote
        assert "price_change_explanation" in adjusted_quote
        assert "new_delivery_estimate" in adjusted_quote
        assert "adjustment_reason" in adjusted_quote
    
    def test_agent_customer_interaction_analysis(self):
        """🔴 RED: Test agent can analyze customer interactions"""
        agent = QuoteGenerationAgent()
        
        # Customer interaction data
        interactions = [
            {
                "interaction_id": "INT001",
                "customer_id": "CODELCO_001", 
                "interaction_type": "quote_request",
                "timestamp": datetime.now() - timedelta(days=5),
                "details": {"products_interest": ["540317340"], "budget_mentioned": 120000}
            },
            {
                "interaction_id": "INT002",
                "customer_id": "CODELCO_001",
                "interaction_type": "price_negotiation", 
                "timestamp": datetime.now() - timedelta(days=3),
                "details": {"requested_discount": 0.10, "accepted_terms": "extended_warranty"}
            },
            {
                "interaction_id": "INT003",
                "customer_id": "CODELCO_001",
                "interaction_type": "quote_acceptance",
                "timestamp": datetime.now() - timedelta(days=1),
                "details": {"final_price": 108000, "negotiation_rounds": 2}
            }
        ]
        
        # Analyze interactions
        analysis = agent.analyze_customer_interactions("CODELCO_001", interactions)
        
        assert "interaction_patterns" in analysis
        assert "negotiation_behavior" in analysis
        assert "decision_timeline" in analysis
        assert "price_sensitivity_score" in analysis
        assert "preferred_communication_style" in analysis


class TestAutomatedQuote:
    """🔴 RED: Test Automated Quote data structure"""
    
    def test_automated_quote_creation(self):
        """🔴 RED: Test AutomatedQuote can be created with required fields"""
        quote = AutomatedQuote(
            quote_id="AQ_001",
            customer_id="CODELCO_001",
            customer_segment="mining",
            products=[
                {
                    "product_id": "540317340",
                    "quantity": 2500,
                    "unit_price": 45.83,
                    "line_total": 114575.0
                }
            ],
            subtotal=114575.0,
            taxes=20623.50,
            total_price=135198.50,
            delivery_estimate="2024-12-15",
            validity_days=30,
            terms_conditions="Standard Nexans terms",
            generation_timestamp=datetime.now()
        )
        
        assert quote.quote_id == "AQ_001"
        assert quote.customer_id == "CODELCO_001"
        assert quote.total_price == 135198.50
        assert len(quote.products) == 1
        assert quote.validity_days == 30
    
    def test_automated_quote_validation(self):
        """🔴 RED: Test AutomatedQuote validates input data"""
        # Should raise error for negative prices
        with pytest.raises(ValueError):
            AutomatedQuote(
                quote_id="AQ_002",
                customer_id="TEST_CUSTOMER",
                customer_segment="industrial",
                products=[{"product_id": "test", "quantity": 100, "unit_price": -10, "line_total": -1000}],
                subtotal=-1000,
                taxes=0,
                total_price=-1000,
                delivery_estimate="2024-12-01",
                validity_days=30
            )
        
        # Should raise error for empty product list
        with pytest.raises(ValueError):
            AutomatedQuote(
                quote_id="AQ_003",
                customer_id="TEST_CUSTOMER",
                customer_segment="industrial", 
                products=[],  # Empty products
                subtotal=0,
                taxes=0,
                total_price=0,
                delivery_estimate="2024-12-01",
                validity_days=30
            )
    
    def test_automated_quote_calculations(self):
        """🔴 RED: Test AutomatedQuote performs calculations correctly"""
        quote = AutomatedQuote(
            quote_id="AQ_004",
            customer_id="TEST_CUSTOMER",
            customer_segment="utility",
            products=[
                {"product_id": "540317340", "quantity": 1000, "unit_price": 45.83, "line_total": 45830},
                {"product_id": "540317341", "quantity": 500, "unit_price": 52.15, "line_total": 26075}
            ],
            subtotal=71905,
            taxes=12943.90,
            total_price=84848.90,
            delivery_estimate="2024-11-30",
            validity_days=45
        )
        
        calculated_subtotal = quote.calculate_subtotal()
        assert calculated_subtotal == 71905
        
        margin_analysis = quote.analyze_margin()
        assert "total_cost" in margin_analysis
        assert "margin_amount" in margin_analysis
        assert "margin_percentage" in margin_analysis
        
        discount_price = quote.apply_discount(0.10)  # 10% discount
        assert discount_price < quote.total_price
        assert abs(discount_price - (quote.total_price * 0.9)) < 0.01


class TestCustomerPreferenceLearner:
    """🔴 RED: Test Customer Preference Learning"""
    
    def test_preference_learner_initialization(self):
        """🔴 RED: Test CustomerPreferenceLearner initialization"""
        learner = CustomerPreferenceLearner()
        assert learner is not None
        assert hasattr(learner, 'learning_algorithms')
        assert hasattr(learner, 'preference_models')
        assert hasattr(learner, 'customer_profiles')
    
    def test_analyze_quote_acceptance_patterns(self):
        """🔴 RED: Test analysis of quote acceptance patterns"""
        learner = CustomerPreferenceLearner()
        
        # Historical quote data
        quote_history = [
            {"quote_id": "Q001", "price": 100000, "margin": 0.25, "delivery_days": 30, "accepted": True, "acceptance_days": 2},
            {"quote_id": "Q002", "price": 150000, "margin": 0.35, "delivery_days": 45, "accepted": False, "rejection_reason": "price"},
            {"quote_id": "Q003", "price": 120000, "margin": 0.28, "delivery_days": 25, "accepted": True, "acceptance_days": 5},
            {"quote_id": "Q004", "price": 180000, "margin": 0.40, "delivery_days": 60, "accepted": False, "rejection_reason": "delivery"},
            {"quote_id": "Q005", "price": 110000, "margin": 0.26, "delivery_days": 35, "accepted": True, "acceptance_days": 1}
        ]
        
        patterns = learner.analyze_quote_acceptance_patterns("CUSTOMER_001", quote_history)
        
        assert "acceptance_rate" in patterns
        assert "price_sensitivity" in patterns  
        assert "delivery_sensitivity" in patterns
        assert "margin_tolerance" in patterns
        assert "decision_speed" in patterns
        assert 0 <= patterns["acceptance_rate"] <= 1
    
    def test_learn_product_preferences(self):
        """🔴 RED: Test learning of product preferences"""
        learner = CustomerPreferenceLearner()
        
        # Customer purchase history
        purchase_history = [
            {"product_id": "540317340", "quantity": 2000, "frequency": 4, "satisfaction_score": 9},
            {"product_id": "540317341", "quantity": 800, "frequency": 2, "satisfaction_score": 7},
            {"product_id": "540317342", "quantity": 1500, "frequency": 3, "satisfaction_score": 8},
            {"product_id": "540317340", "quantity": 1200, "frequency": 2, "satisfaction_score": 9}  # Repeat purchase
        ]
        
        preferences = learner.learn_product_preferences("CUSTOMER_001", purchase_history)
        
        assert "preferred_products" in preferences
        assert "product_loyalty_scores" in preferences
        assert "quantity_patterns" in preferences
        assert "satisfaction_correlation" in preferences
        
        # Product 540317340 should be highly preferred (appears twice with high satisfaction)
        assert "540317340" in preferences["preferred_products"]
    
    def test_predict_quote_acceptance_probability(self):
        """🔴 RED: Test prediction of quote acceptance probability"""
        learner = CustomerPreferenceLearner()
        
        # Train with historical data first
        training_data = [
            {"price": 100000, "margin": 0.25, "delivery_days": 30, "competitor_count": 2, "accepted": True},
            {"price": 150000, "margin": 0.35, "delivery_days": 45, "competitor_count": 3, "accepted": False},
            {"price": 120000, "margin": 0.28, "delivery_days": 25, "competitor_count": 1, "accepted": True},
            {"price": 180000, "margin": 0.40, "delivery_days": 60, "competitor_count": 4, "accepted": False}
        ]
        
        learner.train_acceptance_model("CUSTOMER_001", training_data)
        
        # Predict for new quote
        new_quote_features = {
            "price": 125000,
            "margin": 0.27,
            "delivery_days": 32,
            "competitor_count": 2
        }
        
        probability = learner.predict_quote_acceptance_probability("CUSTOMER_001", new_quote_features)
        
        assert 0 <= probability <= 1
        assert isinstance(probability, float)


class TestQuoteOptimizer:
    """🔴 RED: Test Quote Optimization"""
    
    def test_quote_optimizer_initialization(self):
        """🔴 RED: Test QuoteOptimizer initialization"""
        optimizer = QuoteOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'optimization_strategies')
        assert hasattr(optimizer, 'market_factors')
        assert hasattr(optimizer, 'competitive_intelligence')
    
    def test_optimize_price_for_win_probability(self):
        """🔴 RED: Test price optimization for target win probability"""
        optimizer = QuoteOptimizer()
        
        # Quote parameters
        base_quote = {
            "base_price": 125000,
            "cost": 95000,
            "margin": 30000,
            "customer_segment": "mining"
        }
        
        # Market context
        market_context = {
            "competitor_prices": [118000, 132000, 128000],
            "market_demand": "high",
            "customer_urgency": "normal",
            "relationship_strength": "strong"
        }
        
        # Optimize for 75% win probability
        optimized_quote = optimizer.optimize_price_for_win_probability(
            base_quote, market_context, target_win_probability=0.75
        )
        
        assert "optimized_price" in optimized_quote
        assert "predicted_win_probability" in optimized_quote
        assert "price_adjustment" in optimized_quote
        assert "margin_impact" in optimized_quote
        assert abs(optimized_quote["predicted_win_probability"] - 0.75) < 0.1
    
    def test_optimize_delivery_terms(self):
        """🔴 RED: Test delivery terms optimization"""
        optimizer = QuoteOptimizer()
        
        # Current delivery terms
        current_terms = {
            "delivery_days": 45,
            "delivery_cost": 5000,
            "rush_available": True,
            "installation_included": False
        }
        
        # Customer requirements
        customer_requirements = {
            "max_delivery_days": 35,
            "delivery_flexibility": "medium",
            "cost_sensitivity": "high",
            "installation_preference": "included"
        }
        
        # Optimize
        optimized_terms = optimizer.optimize_delivery_terms(current_terms, customer_requirements)
        
        assert "optimized_delivery_days" in optimized_terms
        assert "delivery_cost_adjustment" in optimized_terms
        assert "additional_services" in optimized_terms
        assert "customer_satisfaction_score" in optimized_terms
        assert optimized_terms["optimized_delivery_days"] <= customer_requirements["max_delivery_days"]
    
    def test_optimize_payment_terms(self):
        """🔴 RED: Test payment terms optimization"""
        optimizer = QuoteOptimizer()
        
        # Standard payment terms
        standard_terms = {
            "payment_schedule": "net_30",
            "early_payment_discount": 0.02,
            "late_payment_penalty": 0.015,
            "advance_payment_required": False
        }
        
        # Customer financial profile
        customer_profile = {
            "credit_rating": "A",
            "payment_history": "excellent",
            "cash_flow_preference": "extended_terms",
            "relationship_duration": "5_years"
        }
        
        # Optimize
        optimized_terms = optimizer.optimize_payment_terms(standard_terms, customer_profile)
        
        assert "recommended_payment_schedule" in optimized_terms
        assert "discount_adjustments" in optimized_terms
        assert "risk_assessment" in optimized_terms
        assert "terms_flexibility" in optimized_terms


class TestBundleQuoteGenerator:
    """🔴 RED: Test Bundle Quote Generation"""
    
    def test_bundle_generator_initialization(self):
        """🔴 RED: Test BundleQuoteGenerator initialization"""
        generator = BundleQuoteGenerator()
        assert generator is not None
        assert hasattr(generator, 'bundling_algorithms')
        assert hasattr(generator, 'discount_strategies')
        assert hasattr(generator, 'product_compatibility_matrix')
    
    def test_calculate_bundle_discount(self):
        """🔴 RED: Test bundle discount calculation"""
        generator = BundleQuoteGenerator()
        
        # Products in bundle
        bundle_products = [
            {"product_id": "540317340", "quantity": 1000, "unit_price": 45.83, "category": "distribution"},
            {"product_id": "540317341", "quantity": 500, "unit_price": 52.15, "category": "transmission"},
            {"product_id": "540317342", "quantity": 800, "unit_price": 38.95, "category": "grounding"}
        ]
        
        # Bundle context
        bundle_context = {
            "customer_segment": "utility",
            "total_value": 100000,
            "products_complementary": True,
            "strategic_customer": True
        }
        
        # Calculate discount
        discount_analysis = generator.calculate_bundle_discount(bundle_products, bundle_context)
        
        assert "base_discount_percentage" in discount_analysis
        assert "volume_discount" in discount_analysis
        assert "complementary_discount" in discount_analysis
        assert "strategic_discount" in discount_analysis
        assert "total_discount_percentage" in discount_analysis
        assert 0 <= discount_analysis["total_discount_percentage"] <= 0.25  # Max 25% discount
    
    def test_generate_product_recommendations(self):
        """🔴 RED: Test product recommendation for bundles"""
        generator = BundleQuoteGenerator()
        
        # Customer's current selection
        selected_products = [
            {"product_id": "540317340", "quantity": 1500, "application": "distribution"}
        ]
        
        # Customer profile
        customer_profile = {
            "segment": "mining",
            "typical_applications": ["distribution", "power", "control"],
            "purchase_history": ["540317340", "540317342"],
            "project_scope": "complete_electrical_system"
        }
        
        # Generate recommendations
        recommendations = generator.generate_product_recommendations(selected_products, customer_profile)
        
        assert "recommended_products" in recommendations
        assert "compatibility_scores" in recommendations
        assert "bundle_value_increase" in recommendations
        assert "recommendation_reasons" in recommendations
        assert len(recommendations["recommended_products"]) > 0
    
    def test_optimize_bundle_configuration(self):
        """🔴 RED: Test bundle configuration optimization"""
        generator = BundleQuoteGenerator()
        
        # Available products
        available_products = [
            {"product_id": "540317340", "price": 45.83, "margin": 0.25, "compatibility": ["540317341", "540317342"]},
            {"product_id": "540317341", "price": 52.15, "margin": 0.30, "compatibility": ["540317340"]},
            {"product_id": "540317342", "price": 38.95, "margin": 0.22, "compatibility": ["540317340"]},
            {"product_id": "540317343", "price": 41.20, "margin": 0.28, "compatibility": ["540317341"]}
        ]
        
        # Optimization objectives
        objectives = {
            "maximize_revenue": True,
            "maintain_minimum_margin": 0.25,
            "customer_budget_limit": 150000,
            "prefer_high_margin_products": True
        }
        
        # Optimize
        optimal_bundle = generator.optimize_bundle_configuration(available_products, objectives)
        
        assert "selected_products" in optimal_bundle
        assert "total_bundle_value" in optimal_bundle
        assert "average_margin" in optimal_bundle
        assert "optimization_score" in optimal_bundle
        assert optimal_bundle["total_bundle_value"] <= objectives["customer_budget_limit"]


class TestQuoteTemplateManager:
    """🔴 RED: Test Quote Template Management"""
    
    def test_template_manager_initialization(self):
        """🔴 RED: Test QuoteTemplateManager initialization"""
        manager = QuoteTemplateManager()
        assert manager is not None
        assert hasattr(manager, 'template_library')
        assert hasattr(manager, 'customization_rules')
        assert hasattr(manager, 'approval_workflows')
    
    def test_select_appropriate_template(self):
        """🔴 RED: Test selection of appropriate quote template"""
        manager = QuoteTemplateManager()
        
        # Quote characteristics
        quote_characteristics = {
            "customer_segment": "mining",
            "quote_value": 125000,
            "product_complexity": "standard",
            "delivery_urgency": "normal",
            "customization_level": "low",
            "multi_product": False
        }
        
        # Select template
        selected_template = manager.select_appropriate_template(quote_characteristics)
        
        assert "template_id" in selected_template
        assert "template_name" in selected_template
        assert "applicable_segments" in selected_template
        assert "customization_options" in selected_template
        assert quote_characteristics["customer_segment"] in selected_template["applicable_segments"]
    
    def test_customize_template_for_customer(self):
        """🔴 RED: Test template customization for specific customer"""
        manager = QuoteTemplateManager()
        
        # Base template
        base_template = {
            "template_id": "MINING_STANDARD",
            "sections": ["header", "products", "pricing", "terms", "footer"],
            "customizable_fields": ["payment_terms", "delivery_conditions", "warranty"],
            "branding": "nexans_standard"
        }
        
        # Customer preferences
        customer_preferences = {
            "customer_id": "CODELCO_001",
            "preferred_language": "spanish",
            "custom_payment_terms": "net_60",
            "required_certifications": ["ISO_9001", "mining_safety"],
            "branding_requirements": "co_branded"
        }
        
        # Customize
        customized_template = manager.customize_template_for_customer(base_template, customer_preferences)
        
        assert "customized_sections" in customized_template
        assert "language_settings" in customized_template  
        assert "certification_inclusions" in customized_template
        assert "branding_configuration" in customized_template
        assert customized_template["language_settings"] == "spanish"
    
    def test_generate_quote_document(self):
        """🔴 RED: Test quote document generation"""
        manager = QuoteTemplateManager()
        
        # Quote data
        quote_data = {
            "quote_id": "Q_MINING_001",
            "customer_name": "Codelco Norte",
            "products": [
                {"name": "Mining Cable 5kV", "quantity": 2500, "unit_price": 45.83, "total": 114575}
            ],
            "subtotal": 114575,
            "taxes": 20623.50,
            "total": 135198.50,
            "delivery_date": "2024-12-15",
            "validity_days": 30
        }
        
        # Template configuration
        template_config = {
            "template_id": "MINING_STANDARD",
            "format": "PDF",
            "include_technical_specs": True,
            "include_certifications": True,
            "language": "spanish"
        }
        
        # Generate document
        document = manager.generate_quote_document(quote_data, template_config)
        
        assert "document_id" in document
        assert "document_format" in document
        assert "document_size_kb" in document
        assert "generation_timestamp" in document
        assert "download_url" in document
        assert document["document_format"] == "PDF"


class TestDynamicPricingIntegrator:
    """🔴 RED: Test Dynamic Pricing Integration"""
    
    def test_pricing_integrator_initialization(self):
        """🔴 RED: Test DynamicPricingIntegrator initialization"""
        integrator = DynamicPricingIntegrator()
        assert integrator is not None
        assert hasattr(integrator, 'pricing_engines')
        assert hasattr(integrator, 'market_data_sources')
        assert hasattr(integrator, 'adjustment_algorithms')
    
    def test_integrate_real_time_pricing(self):
        """🔴 RED: Test real-time pricing integration"""
        integrator = DynamicPricingIntegrator()
        
        # Base quote
        base_quote = {
            "quote_id": "Q12345",
            "products": [
                {"product_id": "540317340", "quantity": 2000, "base_price": 45.83}
            ],
            "customer_segment": "mining",
            "quote_timestamp": datetime.now() - timedelta(hours=2)
        }
        
        # Current market conditions
        market_conditions = {
            "lme_copper_price": 9650.0,  # Current LME price
            "lme_price_change": 0.03,     # 3% increase since quote
            "competitor_activity": "aggressive_pricing",
            "demand_level": "high",
            "supply_status": "constrained"
        }
        
        # Integrate pricing
        updated_quote = integrator.integrate_real_time_pricing(base_quote, market_conditions)
        
        assert "updated_prices" in updated_quote
        assert "price_adjustments" in updated_quote
        assert "adjustment_reasons" in updated_quote
        assert "new_total" in updated_quote
        assert "price_validity" in updated_quote
    
    def test_calculate_competitive_adjustment(self):
        """🔴 RED: Test competitive pricing adjustment"""
        integrator = DynamicPricingIntegrator()
        
        # Current quote price
        current_price = 125000
        
        # Competitive intelligence
        competitive_data = {
            "competitor_prices": [118000, 132000, 128000, 135000],
            "market_position": "premium",
            "competitive_pressure": "medium",
            "differentiation_value": 8000,  # Value of Nexans differentiation
            "customer_loyalty": "high"
        }
        
        # Calculate adjustment
        adjustment = integrator.calculate_competitive_adjustment(current_price, competitive_data)
        
        assert "recommended_price" in adjustment
        assert "adjustment_amount" in adjustment
        assert "competitive_position" in adjustment
        assert "risk_assessment" in adjustment
        assert "confidence_level" in adjustment
    
    @pytest.mark.asyncio
    async def test_monitor_quote_performance(self):
        """🔴 RED: Test quote performance monitoring"""
        integrator = DynamicPricingIntegrator()
        
        # Active quotes to monitor
        active_quotes = [
            {"quote_id": "Q001", "customer_id": "CUSTOMER_A", "total_value": 125000, "days_outstanding": 5},
            {"quote_id": "Q002", "customer_id": "CUSTOMER_B", "total_value": 89000, "days_outstanding": 12},
            {"quote_id": "Q003", "customer_id": "CUSTOMER_C", "total_value": 156000, "days_outstanding": 3}
        ]
        
        # Performance monitoring
        performance_data = await integrator.monitor_quote_performance(active_quotes)
        
        assert "quotes_monitored" in performance_data
        assert "performance_metrics" in performance_data
        assert "alerts" in performance_data
        assert "recommendations" in performance_data
        assert len(performance_data["quotes_monitored"]) == 3


class TestQuoteGenerationIntegration:
    """🔴 RED: Test Quote Generation Agent integration"""
    
    @pytest.mark.asyncio
    async def test_full_quote_generation_workflow(self):
        """🔴 RED: Test complete quote generation workflow"""
        agent = QuoteGenerationAgent()
        
        # Complete customer request
        comprehensive_request = {
            "customer_id": "ENGIE_CHILE_001",
            "customer_segment": "utility", 
            "project_details": {
                "project_name": "Solar Farm Phase 2",
                "total_budget": 250000,
                "timeline": "6_months",
                "technical_requirements": ["UV_resistant", "underground_rated"]
            },
            "products_requested": [
                {"product_id": "540317340", "quantity": 3000, "application": "AC_collection"},
                {"product_id": "540317341", "quantity": 1500, "application": "DC_transmission"}
            ],
            "delivery_requirements": {
                "location": "atacama_desert",
                "max_delivery_time": 45,
                "installation_support": True
            },
            "commercial_preferences": {
                "payment_terms": "progressive",
                "warranty_extension": True,
                "maintenance_contract": "optional"
            }
        }
        
        # Generate comprehensive quote
        comprehensive_quote = await agent.generate_comprehensive_quote(comprehensive_request)
        
        # Validate all components
        assert "quote_package" in comprehensive_quote
        assert "technical_proposal" in comprehensive_quote
        assert "commercial_terms" in comprehensive_quote
        assert "risk_assessment" in comprehensive_quote
        assert "project_timeline" in comprehensive_quote
        assert "customer_value_proposition" in comprehensive_quote
    
    def test_quote_generation_error_handling(self):
        """🔴 RED: Test error handling in quote generation"""
        agent = QuoteGenerationAgent()
        
        # Test with invalid customer ID
        invalid_request = {
            "customer_id": "",  # Empty customer ID
            "products_requested": [{"product_id": "540317340", "quantity": 1000}]
        }
        
        with pytest.raises(QuoteGenerationError):
            agent.generate_automated_quote(invalid_request)
        
        # Test with invalid product data
        invalid_product_request = {
            "customer_id": "VALID_CUSTOMER",
            "products_requested": [{"product_id": "INVALID_PRODUCT", "quantity": -100}]  # Negative quantity
        }
        
        with pytest.raises(QuoteGenerationError):
            agent.generate_automated_quote(invalid_product_request)
    
    def test_quote_generation_performance(self):
        """🔴 RED: Test quote generation performance requirements"""
        agent = QuoteGenerationAgent()
        
        # Large quote request
        large_request = {
            "customer_id": "LARGE_CUSTOMER",
            "customer_segment": "mining",
            "products_requested": [
                {"product_id": f"54031734{i}", "quantity": 1000 + i*100}
                for i in range(10)  # 10 different products
            ]
        }
        
        # Measure generation time
        start_time = datetime.now()
        quote = agent.generate_automated_quote(large_request)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Should generate quote quickly
        assert generation_time < 5.0  # Should complete within 5 seconds
        assert quote is not None
        assert len(quote["line_items"]) == 10