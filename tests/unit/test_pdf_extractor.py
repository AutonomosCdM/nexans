"""
🔴 RED PHASE - Tests para PDF Data Extractor - DEBEN FALLAR PRIMERO
Tarea 1.1.3: PDF Data Extractor con TDD - Usando data REAL de Nexans
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


def test_extract_cable_data_from_pdf():
    """🔴 RED: Test extraer data de PDF real Nexans - DEBE FALLAR PRIMERO"""
    from src.services.pdf_extractor import extract_cable_data
    
    # Usar PDF real del directorio nexans_pdfs
    pdf_path = "/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pdfs/datasheets/Nexans_540317340_4baccee92640.pdf"
    
    result = extract_cable_data(pdf_path)
    
    # Verificar que extrajo información básica
    assert result.nexans_ref == "540317340"
    assert result.copper_kg_per_km > 0
    assert result.aluminum_kg_per_km >= 0
    assert result.voltage_rating > 0
    assert result.product_name is not None


def test_extract_copper_content_from_pdf():
    """🔴 RED: Test extracción específica de contenido de cobre"""
    from src.services.pdf_extractor import extract_copper_content
    
    # Mock PDF text content (basado en PDFs reales)
    mock_text = """
    CABLE SPECIFICATIONS
    Conductor: Copper, stranded
    Copper content: 125.5 kg/km
    Cross-sectional area: 50 mm²
    """
    
    copper_kg = extract_copper_content(mock_text)
    assert copper_kg == 125.5


def test_extract_aluminum_content_from_pdf():  
    """🔴 RED: Test extracción de contenido de aluminio"""
    from src.services.pdf_extractor import extract_aluminum_content
    
    mock_text = """
    CONDUCTOR DETAILS
    Aluminum conductor AAC
    Al content: 89.2 kg/km
    Material: Aluminum alloy 1350
    """
    
    aluminum_kg = extract_aluminum_content(mock_text)
    assert aluminum_kg == 89.2


def test_extract_voltage_rating():
    """🔴 RED: Test extracción de tensión nominal"""
    from src.services.pdf_extractor import extract_voltage_rating
    
    mock_text = """
    ELECTRICAL CHARACTERISTICS
    Rated voltage: 15 kV
    Maximum operating voltage: 17.5 kV
    Insulation level: 15000V
    """
    
    voltage = extract_voltage_rating(mock_text)
    assert voltage == 15000  # En voltios


def test_extract_product_name():
    """🔴 RED: Test extracción de nombre de producto"""
    from src.services.pdf_extractor import extract_product_name
    
    mock_text = """
    NEXANS CABLE DATASHEET
    Product: NEXANS MINING CABLE 15kV EPR/CSP
    Model: SHD-GC-15kV-EPR
    Application: Mining underground
    """
    
    name = extract_product_name(mock_text)
    assert "MINING CABLE" in name
    assert "15kV" in name


def test_extract_multiple_pdfs():
    """🔴 RED: Test procesamiento de múltiples PDFs"""
    from src.services.pdf_extractor import extract_multiple_cables
    
    pdf_directory = "/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pdfs/datasheets/"
    
    results = extract_multiple_cables(pdf_directory, max_files=5)
    
    assert len(results) > 0
    assert len(results) <= 5
    assert all(result.nexans_ref for result in results)
    assert all(len(result.nexans_ref) == 9 for result in results)  # Format validation


def test_extract_with_error_handling():
    """🔴 RED: Test manejo de errores en extracción"""
    from src.services.pdf_extractor import extract_cable_data
    
    # Test con archivo que no existe
    with pytest.raises(FileNotFoundError):
        extract_cable_data("/path/to/nonexistent.pdf")


def test_extract_technical_specs():
    """🔴 RED: Test extracción de especificaciones técnicas"""
    from src.services.pdf_extractor import extract_technical_specs
    
    mock_text = """
    TECHNICAL SPECIFICATIONS
    Cross-sectional area: 95 mm²
    Current carrying capacity: 285 A
    Outer diameter: 28.4 mm
    Weight: 1.85 kg/m
    """
    
    specs = extract_technical_specs(mock_text)
    
    assert specs.conductor_section_mm2 == 95
    assert specs.current_rating == 285
    assert specs.outer_diameter_mm == 28.4
    assert specs.weight_kg_per_km == 1850  # Convertido a kg/km


def test_nexans_ref_extraction():
    """🔴 RED: Test validación formato referencia Nexans"""
    from src.services.pdf_extractor import extract_nexans_ref
    
    # Test con filename
    filename = "Nexans_540317340_4baccee92640.pdf"
    ref = extract_nexans_ref(filename=filename)
    assert ref == "540317340"
    
    # Test con texto PDF
    pdf_text = "Product Reference: 540317341"
    ref = extract_nexans_ref(text=pdf_text)
    assert ref == "540317341"


def test_application_areas_extraction():
    """🔴 RED: Test extracción de áreas de aplicación"""
    from src.services.pdf_extractor import extract_applications
    
    mock_text = """
    APPLICATIONS
    - Mining operations
    - Industrial installations  
    - Underground power distribution
    - Heavy duty applications
    """
    
    applications = extract_applications(mock_text)
    
    assert "mining" in applications
    assert "industrial" in applications
    assert len(applications) > 0


def test_manufacturing_complexity_assessment():
    """🔴 RED: Test assessment de complejidad de fabricación"""
    from src.services.pdf_extractor import assess_manufacturing_complexity
    
    # Cable complejo con múltiples capas
    complex_specs = {
        "layers": 5,
        "special_insulation": True,
        "armored": True,
        "fire_resistant": True
    }
    
    complexity = assess_manufacturing_complexity(complex_specs)
    assert complexity >= 4  # Alta complejidad
    
    # Cable básico
    simple_specs = {
        "layers": 2,
        "special_insulation": False,
        "armored": False
    }
    
    complexity = assess_manufacturing_complexity(simple_specs)
    assert complexity <= 2  # Baja complejidad


@pytest.fixture
def sample_cable_data():
    """🔴 RED: Fixture con data de cable sample"""
    return {
        "nexans_ref": "540317340",
        "product_name": "NEXANS Mining Cable 15kV",
        "copper_kg_per_km": 125.5,
        "aluminum_kg_per_km": 0.0,
        "voltage_rating": 15000,
        "current_rating": 285,
        "applications": ["mining", "industrial"]
    }


def test_cable_data_to_model_conversion(sample_cable_data):
    """🔴 RED: Test conversión de data extraída a modelo Pydantic"""
    from src.services.pdf_extractor import convert_to_cable_model
    from src.models.cable import CableProduct
    
    cable_model = convert_to_cable_model(sample_cable_data)
    
    assert isinstance(cable_model, CableProduct)
    assert cable_model.nexans_ref == "540317340" 
    assert cable_model.copper_content_kg == 125.5
    assert cable_model.is_mining_suitable() == True