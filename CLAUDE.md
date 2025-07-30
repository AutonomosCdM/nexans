# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Nexans Pricing AI - Enterprise Clean Architecture System

## 🎯 PROJECT STATUS
**PRODUCTION DEPLOYED**: Live on Streamlit Cloud ✅
**URL**: https://nexans-autonomos.streamlit.app/
**STATUS**: 68/71 tests passing (96% success rate) + 3,621 lines of agent code + 966 lines dashboard
**ACHIEVEMENT**: Complete system deployed and publicly accessible

## 🧪 TDD PROTOCOL (STRICTLY ENFORCED)
- **Red**: Write failing test first (🔴 markers)
- **Green**: Minimal code to pass  
- **Blue**: Refactor safely
- **NO code without tests first**

## ⚡ DEVELOPMENT COMMANDS

### Quick Start
```bash
# Setup environment
pip install -r requirements.txt

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests (fast)
pytest tests/integration/ -v             # Integration tests
pytest tests/e2e/ -v                     # End-to-end tests

# Run single test
pytest tests/unit/test_specific.py::test_function -v

# Start development server
uvicorn src.api.main:app --reload

# Start dashboard demo
streamlit run dashboard.py

# Check LME integration
python -c "from src.services.lme_api import get_lme_copper_price; print(f'Copper: ${get_lme_copper_price()}/ton')"
```

### Production Deployment
```bash
# Full stack deployment
docker-compose up -d

# API only
docker-compose up nexans-pricing-api redis -d

# View logs
docker-compose logs -f nexans-pricing-api
```

## 🏗️ CLEAN ARCHITECTURE STRUCTURE

The system follows **Clean Architecture** with strict layer separation:

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)         │ ← REST endpoints (/api/*)
├─────────────────────────────────────┤
│       Application Layer             │ ← Service orchestration
│   • QuoteApplicationService         │   Command/Query pattern
│   • PricingApplicationService       │   DTO contracts
├─────────────────────────────────────┤
│         Domain Layer                │ ← Business logic & entities
│   • Customer • Product • Quote      │   Rich domain models
│   • MaterialCost • VolumeDiscount   │   Domain services
├─────────────────────────────────────┤ 
│       Infrastructure Layer          │ ← External integrations
│   • Repositories • LME API • PDFs   │   Repository pattern
└─────────────────────────────────────┘
```

### Key Architectural Components

**Domain Models** (`src/domain/models/`):
- **Customer**: Segment-based pricing logic (mining 1.5x, industrial 1.3x, utility 1.2x, residential 1.0x)
- **Product**: Technical specifications with material calculations
- **Quote**: Business quote generation with validation rules
- **MaterialCost**: Real-time cost calculation with LME integration

**Application Services** (`src/application/services/`):
- **QuoteApplicationService**: Quote generation orchestration
- **PricingApplicationService**: Pricing calculation workflows  
- **CustomerApplicationService**: Customer management operations

**Infrastructure** (`src/infrastructure/`):
- **Repositories**: Data access abstraction layer
- **DIContainer**: Dependency injection with interface segregation

## 🧠 BUSINESS DOMAIN KNOWLEDGE

### Customer Segmentation & Multipliers
- **mining**: 1.5x (45% margin target) - harsh environments
- **industrial**: 1.3x (35% margin target) - standard industrial  
- **utility**: 1.2x (30% margin target) - utility grade
- **residential**: 1.0x (25% margin target) - residential grade

### Volume Discount Tiers
- 1-100m: 0% | 101-500m: 3% | 501-1000m: 5% | 1001-5000m: 8% | 5000m+: 12%

### Regional Factors
- chile_central: 1.0 (base) | chile_north: 1.15 (mining premium)
- chile_south: 1.08 (logistics) | international: 1.25 (export)

### Urgency Multipliers
- standard: 1.0 | urgent: 1.2 (+20%) | express: 1.35 (+35%)

## 🔗 EXTERNAL INTEGRATIONS

### LME Real-time Pricing
- **Primary**: Metals-API (https://metals-api.com/api/latest)
- **Backup**: TradingEconomics API
- **Update Frequency**: Every 5 minutes with intelligent caching
- **Current Prices**: Copper ~$9,500/ton, Aluminum ~$2,650/ton

### PDF Data Extraction
- **Source**: `/nexans_pdfs/datasheets/` (40+ Nexans products)
- **Parser**: PyMuPDF + PyPDF2 for technical specifications
- **Extracted Data**: Voltage, current, weight, copper/aluminum content

### ML Pricing Model
- **Algorithm**: XGBoost with 10+ engineered features
- **Features**: LME prices, material content, customer segment, volume, urgency
- **Training**: Synthetic + real data combination
- **Performance**: <200ms response time with caching

## 🧪 TEST STRUCTURE & COVERAGE

**Current Status**: 68/71 tests passing (96% success rate)

### Test Categories
- **Unit Tests** (`tests/unit/`): 40+ tests for domain logic
- **Integration Tests** (`tests/integration/`): Service interaction validation
- **Characterization Tests** (`tests/characterization/`): Behavior documentation  
- **E2E Tests** (`tests/e2e/`): Complete workflow validation

### Test Commands by Layer
```bash
# Domain layer tests
pytest tests/unit/domain/ -v

# Application layer tests  
pytest tests/unit/application/ -v

# Infrastructure tests
pytest tests/unit/infrastructure/ -v

# API integration tests
pytest tests/integration/api/ -v
```

## 🚀 API ENDPOINTS

**Core Endpoints**:
- `POST /api/quotes/generate` - Complete quote generation
- `POST /api/pricing/calculate` - Detailed pricing calculation  
- `GET /api/prices/current` - Real-time LME prices
- `GET /api/cables/search` - Advanced cable search
- `GET /api/cables/{reference}` - Specific cable details
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 📁 DATA SOURCES

- **Nexans PDFs**: `/nexans_pdfs/datasheets/` (40 technical datasheets)
- **Technical Specs**: `/nexans_pdfs/organized/technical_specs/` (33 Excel files)
- **LME Real-time**: Metals-API + TradingEconomics backup
- **Training Data**: Synthetic + real data combination for ML models

## 🔒 SECURITY & CONFIGURATION

### Environment Variables (.env)
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# External APIs
METALS_API_KEY=your_metals_api_key
TRADING_ECONOMICS_API_KEY=your_te_api_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/nexans_pricing
REDIS_URL=redis://localhost:6379
```

### Docker Services
- **nexans-pricing-api**: FastAPI application (port 8000)
- **nexans-dashboard**: Streamlit dashboard (port 8501)  
- **redis**: Caching layer (port 6379)
- **postgres**: Data persistence (port 5432)
- **nginx**: Reverse proxy (ports 80/443)

## ⚠️ CRITICAL DEVELOPMENT RULES

### TDD Compliance
- **NEVER** write implementation code before tests
- **ALWAYS** run tests before committing  
- **MAINTAIN** 96%+ test success rate
- **RED → GREEN → REFACTOR** cycle strictly enforced

### Clean Architecture Boundaries
- **Domain layer**: NO external dependencies
- **Application layer**: Only domain and infrastructure interfaces
- **Infrastructure layer**: Implements interfaces, handles externals
- **API layer**: Thin controllers, delegate to application services

### Code Quality Standards
```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking (if mypy is added)
mypy src/
```

## 🎯 CURRENT DEVELOPMENT FOCUS

**✅ COMPLETED PHASES**:
- **Phase 3**: Clean Architecture Transformation
- **Phase 4**: Intelligent Agents Implementation
  - ✅ MarketIntelligenceAgent (682 lines) - LME monitoring, price alerts, volatility detection
  - ✅ DemandForecastingAgent (1,179 lines) - ML predictions, seasonal analysis, ARIMA/Prophet/LSTM
  - ✅ QuoteGenerationAgent (1,683 lines) - Automated quotes, customer learning, dynamic pricing

- **Phase 5**: Dashboard Demo Implementation
  - ✅ Streamlit real-time interface (966 lines) - Executive dashboard with interactive components
  - ✅ LME price monitoring dashboard - Real-time metal prices with trend visualization
  - ✅ Automated quote generation UI - Complete quote workflow interface
  - ✅ Agent performance visualization - Market intelligence and forecasting displays

- **Phase 6**: Production Deployment
  - ✅ Streamlit Cloud deployment (https://nexans-autonomos.streamlit.app/)
  - ✅ Demo mode with fallback data for cloud compatibility
  - ✅ Python 3.13 compatibility fixes
  - ✅ Complete troubleshooting documentation
  - ✅ Public access with professional branding

## 🌐 **STREAMLIT CLOUD DEPLOYMENT**

### **Live Dashboard**: https://nexans-autonomos.streamlit.app/

### **Deployment Configuration**:
- **Platform**: Streamlit Community Cloud
- **Mode**: Demo mode with fallback data
- **Dependencies**: Python 3.13 compatible versions
- **Features**: Full dashboard functionality without backend dependency

### **Demo Data Configuration**:
```toml
[demo]
enabled = true
copper_price = 9598
aluminum_price = 2681
quotes_generated = 847
revenue_pipeline = 2400000
```

### **Key Implementation Details**:
- All API calls protected with DEMO_MODE checks
- Fallback demo data for cloud environment
- Zero localhost dependencies in production
- Professional corporate branding maintained

**Production Status**: ✅ LIVE and fully functional with public access 24/7.