"""
🟢 GREEN PHASE - FastAPI Main Application
Sprint 2.2.2: API endpoints para cotizaciones automáticas

IMPLEMENTATION TO MAKE TESTS PASS:
✅ FastAPI app with proper configuration
✅ Router inclusion for all endpoints
✅ CORS middleware
✅ Error handling middleware
✅ Health check endpoint
✅ API documentation
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import time
from typing import Dict, Any

# Import routers (will be created next)
from src.api.endpoints.quotes import router as quotes_router
from src.api.endpoints.pricing import router as pricing_router
from src.api.endpoints.cables import router as cables_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Nexans Pricing Intelligence API",
    description="API for intelligent cable pricing with ML and real-time market data",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware for timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Include routers
app.include_router(quotes_router, prefix="/api/quotes", tags=["Quotes"])
app.include_router(pricing_router, prefix="/api/pricing", tags=["Pricing"])
app.include_router(cables_router, prefix="/api/cables", tags=["Cables"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """🟢 GREEN: Health check endpoint"""
    try:
        # Check service dependencies
        services_status = {}
        
        # Check LME API service
        try:
            from src.services.lme_api import get_lme_copper_price
            copper_price = get_lme_copper_price(use_fallback=True)
            services_status["lme_api"] = "healthy" if copper_price > 0 else "degraded"
        except Exception:
            services_status["lme_api"] = "unhealthy"
        
        # Check ML Model service
        try:
            from src.pricing.ml_model import PricingModel
            model = PricingModel()
            services_status["ml_model"] = "healthy"
        except Exception:
            services_status["ml_model"] = "unhealthy"
        
        # Overall status
        overall_status = "healthy" if all(
            status in ["healthy", "degraded"] for status in services_status.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": app.version,
            "services": services_status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# API info endpoint
@app.get("/api/info")
async def api_info():
    """🟢 GREEN: API information endpoint"""
    return {
        "title": app.title,
        "version": app.version,
        "description": app.description,
        "endpoints": {
            "quotes": "/api/quotes",
            "pricing": "/api/pricing", 
            "cables": "/api/cables",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """🟢 GREEN: Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": exc.status_code,
            "message": "HTTP error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """🟢 GREEN: Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": 500,
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """🟢 GREEN: Root endpoint"""
    return {
        "message": "Nexans Pricing Intelligence API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)