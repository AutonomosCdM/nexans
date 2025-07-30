#!/bin/bash

# =============================================================================
# NEXANS PRICING INTELLIGENCE SYSTEM - DEPLOYMENT SCRIPT
# Script de despliegue automático para el sistema completo
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Main deployment function
main() {
    print_header "NEXANS PRICING INTELLIGENCE SYSTEM DEPLOYMENT"
    
    echo "🏭 Sistema completo de pricing inteligente con agentes IA"
    echo "📊 Desarrollado para: Gerardo Iniescar (CIO D&U AMEA)"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Deployment options
    echo "Seleccione método de despliegue:"
    echo "1) 🐳 Docker Compose (Recomendado)"
    echo "2) 🚀 Local Development"
    echo "3) ☁️  Production Deploy"
    echo "4) 🧪 Testing Only"
    
    read -p "Opción [1-4]: " deploy_option
    
    case $deploy_option in
        1) deploy_docker_compose ;;
        2) deploy_local ;;
        3) deploy_production ;;
        4) run_tests ;;
        *) print_error "Opción inválida" && exit 1 ;;
    esac
    
    print_success "Despliegue completado exitosamente!"
    show_access_info
}

# Check prerequisites
check_prerequisites() {
    print_info "Verificando prerequisitos..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 no está instalado"
        exit 1
    fi
    print_success "Python 3 encontrado: $(python3 --version)"
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        print_error "pip no está instalado"
        exit 1
    fi
    print_success "pip encontrado"
    
    # Check Docker (if needed)
    if command -v docker &> /dev/null; then
        print_success "Docker encontrado: $(docker --version)"
    else
        print_warning "Docker no encontrado - solo despliegue local disponible"
    fi
    
    # Check docker-compose
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose encontrado"
    else
        print_warning "Docker Compose no encontrado"
    fi
}

# Docker Compose deployment
deploy_docker_compose() {
    print_header "DOCKER COMPOSE DEPLOYMENT"
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose no está instalado"
        exit 1
    fi
    
    # Build and start services
    print_info "Construyendo imágenes Docker..."
    docker-compose build
    
    print_info "Iniciando servicios..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Esperando que los servicios estén listos..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API está funcionando"
    else
        print_warning "API no responde - puede necesitar más tiempo"
    fi
    
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        print_success "Dashboard está funcionando"
    else
        print_warning "Dashboard no responde - puede necesitar más tiempo"
    fi
    
    print_info "Para ver logs: docker-compose logs -f"
    print_info "Para detener: docker-compose down"
}

# Local development deployment
deploy_local() {
    print_header "LOCAL DEVELOPMENT DEPLOYMENT"
    
    # Create virtual environment
    print_info "Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    print_info "Instalando dependencias..."
    pip install -r requirements.txt
    
    # Install additional dashboard dependencies
    pip install streamlit plotly
    
    print_success "Dependencias instaladas"
    
    # Set environment variables
    export PYTHONPATH=$(pwd)
    
    # Start API in background
    print_info "Iniciando API FastAPI..."
    python app.py &
    API_PID=$!
    
    # Wait for API to start
    sleep 10
    
    # Start dashboard
    print_info "Iniciando Dashboard Streamlit..."
    streamlit run dashboard.py --server.port 8501 &
    DASHBOARD_PID=$!
    
    # Save PIDs for cleanup
    echo $API_PID > api.pid
    echo $DASHBOARD_PID > dashboard.pid
    
    print_success "Servicios iniciados localmente"
    print_info "Para detener: ./stop_local.sh"
}

# Production deployment
deploy_production() {
    print_header "PRODUCTION DEPLOYMENT"
    
    print_warning "Configuración de producción requiere:"
    echo "• Servidor con Docker instalado"
    echo "• Certificados SSL configurados"
    echo "• Variables de entorno de producción"
    echo "• Base de datos PostgreSQL"
    echo "• Redis para caching"
    echo ""
    
    read -p "¿Continuar con configuración de producción? [y/N] " confirm
    
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        print_info "Despliegue de producción cancelado"
        return
    fi
    
    # Production environment setup
    print_info "Configurando entorno de producción..."
    
    # Create production docker-compose
    cp docker-compose.yml docker-compose.prod.yml
    
    # Modify for production
    sed -i 's/restart: unless-stopped/restart: always/g' docker-compose.prod.yml
    
    print_info "Iniciando servicios de producción..."
    docker-compose -f docker-compose.prod.yml up -d
    
    print_success "Despliegue de producción iniciado"
    
    # Production health checks
    sleep 30
    check_production_health
}

# Run tests only
run_tests() {
    print_header "TESTING MODE"
    
    # Install test dependencies
    print_info "Instalando dependencias de testing..."
    pip install pytest pytest-cov pytest-asyncio
    
    # Run tests
    print_info "Ejecutando suite de tests..."
    
    echo "🧪 Ejecutando tests unitarios..."
    pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term
    
    echo "🔗 Ejecutando tests de integración..."
    pytest tests/integration/ -v
    
    echo "🎯 Ejecutando tests end-to-end..."
    pytest tests/e2e/ -v
    
    print_success "Tests completados"
    print_info "Reporte de cobertura disponible en: htmlcov/index.html"
}

# Check production health
check_production_health() {
    print_info "Verificando salud del sistema en producción..."
    
    # API health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API funcionando correctamente"
    else
        print_error "API no responde"
    fi
    
    # Dashboard health check
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        print_success "Dashboard funcionando correctamente"
    else
        print_error "Dashboard no responde"
    fi
    
    # Database health check
    if docker-compose exec postgres pg_isready > /dev/null 2>&1; then
        print_success "Base de datos funcionando"
    else
        print_warning "Base de datos no responde"
    fi
    
    # Redis health check
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis funcionando"
    else
        print_warning "Redis no responde"
    fi
}

# Show access information
show_access_info() {
    print_header "INFORMACIÓN DE ACCESO"
    
    echo "🌐 URLs de acceso:"
    echo ""
    echo "📊 API FastAPI:"
    echo "   • URL: http://localhost:8000"
    echo "   • Documentación: http://localhost:8000/docs"
    echo "   • Health Check: http://localhost:8000/health"
    echo ""
    echo "🎨 Dashboard Streamlit:"
    echo "   • URL: http://localhost:8501"
    echo ""
    echo "🔍 Endpoints principales:"
    echo "   • Pricing Calculator: POST /api/pricing/calculate"
    echo "   • Quote Generator: POST /api/quotes/generate"
    echo "   • Market Intelligence: GET /api/agents/market/status"
    echo "   • Demand Forecast: GET /api/agents/demand/forecast"
    echo ""
    echo "📈 Características disponibles:"
    echo "   ✅ Market Intelligence Agent (LME monitoring)"
    echo "   ✅ Demand Forecasting Agent (ML predictions)"
    echo "   ✅ Quote Generation Agent (automated quotes)"
    echo "   ✅ Real-time pricing engine"
    echo "   ✅ PDF data extraction"
    echo "   ✅ Interactive dashboard"
    echo ""
    echo "🤖 Sistema completo de pricing inteligente operacional"
    echo "🎯 Ready para demo con Gerardo Iniescar (CIO D&U AMEA)"
}

# Cleanup function
cleanup() {
    print_info "Limpiando recursos..."
    
    # Kill background processes if they exist
    if [ -f api.pid ]; then
        kill $(cat api.pid) 2>/dev/null || true
        rm api.pid
    fi
    
    if [ -f dashboard.pid ]; then
        kill $(cat dashboard.pid) 2>/dev/null || true
        rm dashboard.pid
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"