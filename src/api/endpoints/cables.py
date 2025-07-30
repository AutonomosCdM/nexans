"""
🟢 GREEN PHASE - Cables API Endpoints
Sprint 2.2.2: Cable search and information endpoints

ENDPOINTS IMPLEMENTED:
✅ GET /api/cables/search - Search cables by specifications
✅ Cable filtering and pagination
✅ Response formatting
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
from typing import Optional, List
import math

from src.api.models.requests import CableSearchRequest
from src.api.models.responses import (
    CableSearchResponse,
    CableInfoResponse,
    PaginationResponse
)

# Import our existing components
from src.models.cable import CableProduct

router = APIRouter()


def get_mock_cables() -> List[CableProduct]:
    """🟢 GREEN: Get mock cable database (in production, this would be real DB)"""
    cables = [
        CableProduct(
            nexans_reference="540317340",
            product_name="Nexans SHD-GC-EU 3x4+2x8+1x6_5kV",
            voltage_rating=5000,
            current_rating=122,
            conductor_section_mm2=21.2,
            copper_content_kg=2.3,
            aluminum_content_kg=0.0,
            weight_kg_per_km=2300,
            applications=["mining"]
        ),
        CableProduct(
            nexans_reference="540317341",
            product_name="Nexans Industrial Cable 1kV",
            voltage_rating=1000,
            current_rating=50,
            conductor_section_mm2=10.0,
            copper_content_kg=1.2,
            aluminum_content_kg=0.0,
            weight_kg_per_km=800,
            applications=["industrial"]
        ),
        CableProduct(
            nexans_reference="540317342",
            product_name="Nexans Utility Cable 15kV",
            voltage_rating=15000,
            current_rating=200,
            conductor_section_mm2=35.0,
            copper_content_kg=3.5,
            aluminum_content_kg=1.2,
            weight_kg_per_km=3500,
            applications=["utility", "industrial"]
        ),
        CableProduct(
            nexans_reference="540317343",
            product_name="Nexans Residential Cable 230V",
            voltage_rating=230,
            current_rating=16,
            conductor_section_mm2=2.5,
            copper_content_kg=0.8,
            aluminum_content_kg=0.0,
            weight_kg_per_km=300,
            applications=["residential"]
        ),
        CableProduct(
            nexans_reference="540317344",
            product_name="Nexans Mining Heavy Duty 35kV",
            voltage_rating=35000,
            current_rating=500,
            conductor_section_mm2=120.0,
            copper_content_kg=8.5,
            aluminum_content_kg=2.8,
            weight_kg_per_km=7500,
            applications=["mining", "industrial"]
        )
    ]
    return cables


def filter_cables(cables: List[CableProduct], search_request: CableSearchRequest) -> List[CableProduct]:
    """🟢 GREEN: Filter cables based on search criteria"""
    filtered_cables = cables
    
    # Filter by reference
    if search_request.reference:
        filtered_cables = [c for c in filtered_cables 
                          if search_request.reference.lower() in c.nexans_reference.lower()]
    
    # Filter by voltage range
    if search_request.voltage_min is not None:
        filtered_cables = [c for c in filtered_cables 
                          if c.voltage_rating >= search_request.voltage_min]
    
    if search_request.voltage_max is not None:
        filtered_cables = [c for c in filtered_cables 
                          if c.voltage_rating <= search_request.voltage_max]
    
    # Filter by current range
    if search_request.current_min is not None:
        filtered_cables = [c for c in filtered_cables 
                          if c.current_rating and c.current_rating >= search_request.current_min]
    
    if search_request.current_max is not None:
        filtered_cables = [c for c in filtered_cables 
                          if c.current_rating and c.current_rating <= search_request.current_max]
    
    # Filter by application
    if search_request.application:
        filtered_cables = [c for c in filtered_cables 
                          if search_request.application in c.applications]
    
    # Filter by conductor material
    if search_request.conductor_material:
        if search_request.conductor_material == "copper":
            filtered_cables = [c for c in filtered_cables 
                             if c.copper_content_kg > 0 and c.aluminum_content_kg == 0]
        elif search_request.conductor_material == "aluminum":
            filtered_cables = [c for c in filtered_cables 
                             if c.aluminum_content_kg > 0 and c.copper_content_kg == 0]
        elif search_request.conductor_material == "copper_aluminum":
            filtered_cables = [c for c in filtered_cables 
                             if c.copper_content_kg > 0 and c.aluminum_content_kg > 0]
    
    return filtered_cables


def paginate_results(cables: List[CableProduct], page: int, limit: int) -> tuple:
    """🟢 GREEN: Paginate cable results"""
    total_results = len(cables)
    total_pages = math.ceil(total_results / limit) if total_results > 0 else 1
    
    start_index = (page - 1) * limit
    end_index = start_index + limit
    
    paginated_cables = cables[start_index:end_index]
    
    pagination = PaginationResponse(
        page=page,
        limit=limit,
        total_pages=total_pages,
        total_results=total_results
    )
    
    return paginated_cables, pagination


@router.get("/search", response_model=CableSearchResponse)
async def search_cables(
    reference: Optional[str] = Query(None, description="Search by Nexans reference"),
    voltage_min: Optional[int] = Query(None, ge=0, description="Minimum voltage rating"),
    voltage_max: Optional[int] = Query(None, ge=0, description="Maximum voltage rating"),
    current_min: Optional[int] = Query(None, ge=0, description="Minimum current rating"),
    current_max: Optional[int] = Query(None, ge=0, description="Maximum current rating"),
    application: Optional[str] = Query(None, description="Cable application"),
    conductor_material: Optional[str] = Query(None, description="Conductor material"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page")
):
    """🟢 GREEN: Search cables by specifications"""
    try:
        # Create search request object for validation
        search_request = CableSearchRequest(
            reference=reference,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            current_min=current_min,
            current_max=current_max,
            application=application,
            conductor_material=conductor_material,
            page=page,
            limit=limit
        )
        
        # Get all cables
        all_cables = get_mock_cables()
        
        # Filter cables based on search criteria
        filtered_cables = filter_cables(all_cables, search_request)
        
        # Paginate results
        paginated_cables, pagination = paginate_results(filtered_cables, page, limit)
        
        # Convert to response format
        cable_responses = []
        for cable in paginated_cables:
            cable_info = CableInfoResponse(
                nexans_reference=cable.nexans_reference,
                product_name=cable.product_name,
                voltage_rating=cable.voltage_rating,
                current_rating=cable.current_rating,
                conductor_section_mm2=cable.conductor_section_mm2,
                weight_kg_per_km=cable.weight_kg_per_km,
                applications=cable.applications,
                copper_content_kg=cable.copper_content_kg,
                aluminum_content_kg=cable.aluminum_content_kg,
                manufacturing_complexity=cable.get_complexity_level()
            )
            cable_responses.append(cable_info)
        
        # Build search criteria for response
        search_criteria = {}
        if reference: search_criteria["reference"] = reference
        if voltage_min: search_criteria["voltage_min"] = voltage_min
        if voltage_max: search_criteria["voltage_max"] = voltage_max
        if current_min: search_criteria["current_min"] = current_min  
        if current_max: search_criteria["current_max"] = current_max
        if application: search_criteria["application"] = application
        if conductor_material: search_criteria["conductor_material"] = conductor_material
        
        response = CableSearchResponse(
            cables=cable_responses,
            search_criteria=search_criteria,
            pagination=pagination,
            total_results=len(filtered_cables)
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cable search failed: {str(e)}"
        )


@router.get("/search/")  
async def search_cables_alias(
    reference: Optional[str] = Query(None),
    voltage_min: Optional[int] = Query(None, ge=0),
    voltage_max: Optional[int] = Query(None, ge=0),
    current_min: Optional[int] = Query(None, ge=0),
    current_max: Optional[int] = Query(None, ge=0),
    application: Optional[str] = Query(None),
    conductor_material: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """🟢 GREEN: Alias for search endpoint with trailing slash"""
    return await search_cables(
        reference, voltage_min, voltage_max, current_min, current_max,
        application, conductor_material, page, limit
    )


@router.get("/{cable_reference}", response_model=CableInfoResponse)
async def get_cable_by_reference(cable_reference: str):
    """🟢 GREEN: Get specific cable by reference"""
    try:
        # Get all cables
        all_cables = get_mock_cables()
        
        # Find cable by reference
        cable = None
        for c in all_cables:
            if c.nexans_reference == cable_reference:
                cable = c
                break
        
        if not cable:
            raise HTTPException(
                status_code=404, 
                detail=f"Cable not found: {cable_reference}"
            )
        
        # Convert to response format
        cable_info = CableInfoResponse(
            nexans_reference=cable.nexans_reference,
            product_name=cable.product_name,
            voltage_rating=cable.voltage_rating,
            current_rating=cable.current_rating,
            conductor_section_mm2=cable.conductor_section_mm2,
            weight_kg_per_km=cable.weight_kg_per_km,
            applications=cable.applications,
            copper_content_kg=cable.copper_content_kg,
            aluminum_content_kg=cable.aluminum_content_kg,
            manufacturing_complexity=cable.get_complexity_level()
        )
        
        return cable_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cable: {str(e)}"
        )


@router.get("/")
async def cables_info():
    """🟢 GREEN: Information about cables endpoints"""
    return {
        "endpoints": {
            "search": "GET /api/cables/search - Search cables by specifications",
            "get_by_reference": "GET /api/cables/{reference} - Get cable by reference"
        },
        "search_parameters": [
            "reference", "voltage_min", "voltage_max", 
            "current_min", "current_max", "application", 
            "conductor_material", "page", "limit"
        ],
        "available_applications": [
            "mining", "industrial", "utility", "residential", "marine", "construction"
        ],
        "conductor_materials": [
            "copper", "aluminum", "copper_aluminum"
        ]
    }