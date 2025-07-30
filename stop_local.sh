#!/bin/bash

# =============================================================================
# STOP LOCAL DEPLOYMENT - Nexans Pricing Intelligence System
# =============================================================================

echo "🛑 Deteniendo Nexans Pricing Intelligence System..."

# Kill API process
if [ -f api.pid ]; then
    echo "⏹️  Deteniendo API FastAPI..."
    kill $(cat api.pid) 2>/dev/null && echo "✅ API detenida" || echo "⚠️  API ya estaba detenida"
    rm api.pid
fi

# Kill Dashboard process
if [ -f dashboard.pid ]; then
    echo "⏹️  Deteniendo Dashboard Streamlit..."
    kill $(cat dashboard.pid) 2>/dev/null && echo "✅ Dashboard detenido" || echo "⚠️  Dashboard ya estaba detenido"
    rm dashboard.pid
fi

# Kill any remaining processes
pkill -f "python app.py" 2>/dev/null || true
pkill -f "streamlit run dashboard.py" 2>/dev/null || true

echo "✅ Todos los servicios han sido detenidos"
echo "💡 Para reiniciar, ejecute: ./deploy.sh"