"""
🟢 GREEN PHASE - PDF Data Extractor para Nexans
Basado en estructura real de PDFs Nexans observada
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CableExtractedData:
    """🟢 GREEN: Data structure para información extraída"""
    nexans_ref: str
    product_name: Optional[str] = None
    copper_kg_per_km: Optional[float] = None
    aluminum_kg_per_km: Optional[float] = None
    voltage_rating: Optional[int] = None
    current_rating: Optional[float] = None
    applications: Optional[List[str]] = None
    weight_kg_per_km: Optional[float] = None
    conductor_section_mm2: Optional[float] = None
    outer_diameter_mm: Optional[float] = None


def extract_cable_data(pdf_path: str) -> CableExtractedData:
    """🟢 GREEN: Implementación principal de extracción"""
    try:
        import fitz  # PyMuPDF
        
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text()
        
        # Extraer nexans_ref del filename como fallback
        filename = os.path.basename(pdf_path)
        nexans_ref = extract_nexans_ref(filename=filename, text=full_text)
        
        return CableExtractedData(
            nexans_ref=nexans_ref,
            product_name=extract_product_name(full_text),
            copper_kg_per_km=extract_copper_content(full_text),
            aluminum_kg_per_km=extract_aluminum_content(full_text),
            voltage_rating=extract_voltage_rating(full_text),
            current_rating=extract_current_rating(full_text),
            applications=extract_applications(full_text),
            weight_kg_per_km=extract_weight(full_text),
            conductor_section_mm2=extract_conductor_section(full_text),
            outer_diameter_mm=extract_outer_diameter(full_text)
        )
        
    except ImportError:
        # Fallback sin PyMuPDF - usar solo filename
        filename = os.path.basename(pdf_path)
        nexans_ref = extract_nexans_ref(filename=filename)
        
        return CableExtractedData(
            nexans_ref=nexans_ref,
            product_name="Extracted from filename only",
            copper_kg_per_km=0.0,
            aluminum_kg_per_km=0.0,
            voltage_rating=5000  # Default
        )


def extract_nexans_ref(filename: Optional[str] = None, text: Optional[str] = None) -> str:
    """🟢 GREEN: Extraer referencia Nexans"""
    if filename:
        # Pattern: Nexans_540317340_hash.pdf
        match = re.search(r'Nexans_(\d{9})_', filename)
        if match:
            return match.group(1)
    
    if text:
        # Buscar en texto del PDF
        patterns = [
            r'Product\s+Reference[:\s]+(\d{9})',
            r'Ref[:\s]+(\d{9})',
            r'\b(\d{9})\b'  # Any 9-digit number
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
    
    return "000000000"  # Default fallback


def extract_product_name(text: str) -> Optional[str]:
    """🟢 GREEN: Extraer nombre del producto basado en PDFs reales"""
    # Patrones observados en PDFs Nexans reales
    patterns = [
        r'Nexans\s+(SHD-GC-EU|SHD-GC-CP|POWERMINE[^\\n]*)',
        r'Product:\s*([^\\n]+)',
        r'NEXANS\s+([A-Z][^\\n]{10,50})',
        r'Portable cable[^\\n]+mining[^\\n]*'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1) if match.lastindex else match.group(0)
            return name.strip()[:100]  # Limit length
    
    return None


def extract_copper_content(text: str) -> Optional[float]:
    """♻️ REFACTOR: Extracción avanzada de contenido de cobre con cálculos precisos"""
    # Buscar menciones explícitas primero
    explicit_patterns = [
        r'Copper\s+content[:\s]*([0-9.]+)\s*kg/km',
        r'Cu\s+content[:\s]*([0-9.]+)\s*kg/km',
        r'Copper[:\s]*([0-9.]+)\s*kg/km'
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Si es conductor de cobre, calcular basado en especificaciones
    if re.search(r'Conductor\s+material\s+Copper', text, re.IGNORECASE):
        section = extract_conductor_section(text)
        num_conductors = extract_number_of_conductors(text)
        
        if section and num_conductors:
            # Cálculo más preciso: densidade cobre 8.96 g/cm³
            # Para conductores múltiples
            copper_density = 8.96  # kg/dm³
            total_section_dm2 = (section * num_conductors) / 100  # Convert mm² to dm²
            return total_section_dm2 * copper_density * 10  # *10 for km vs 100m
        elif section:
            # Single conductor fallback
            return section * 8.96
    
    return None


def extract_number_of_conductors(text: str) -> Optional[int]:
    """♻️ REFACTOR: Extraer número de conductores"""
    patterns = [
        r'(\d+)\s*x\s*\d+',  # Pattern like "3x4" 
        r'(\d+)\s*conductor',
        r'Total\s+number\s+of\s+wires\s+(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 1  # Default single conductor


def extract_aluminum_content(text: str) -> Optional[float]:
    """🟢 GREEN: Extraer contenido de aluminio"""
    patterns = [
        r'Aluminum\s+content[:\s]*([0-9.]+)\s*kg/km',
        r'Al\s+content[:\s]*([0-9.]+)\s*kg/km',
        r'AAC.*?([0-9.]+)\s*kg/km'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Si es conductor de cobre, aluminum es 0
    if re.search(r'Conductor\s+material\s+Copper', text, re.IGNORECASE):
        return 0.0
        
    return None


def extract_voltage_rating(text: str) -> Optional[int]:
    """🟢 GREEN: Extraer tensión nominal - basado en PDF real"""
    patterns = [
        r'Rated\s+voltage\s+Ur\s+([0-9.]+)\s*kV',
        r'([0-9.]+)\s*kV',  # Simple kV pattern
        r'voltage[:\s]*([0-9.]+)\s*kV',
        r'([0-9.]+)\s*V(?!\w)'  # Voltage in V
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                voltage = float(match)
                if voltage < 100:  # Assumed kV
                    return int(voltage * 1000)
                else:  # Assumed V
                    return int(voltage)
    
    return None


def extract_current_rating(text: str) -> Optional[float]:
    """🟢 GREEN: Extraer corriente nominal"""
    patterns = [
        r'Perm\s+current\s+rating[^0-9]*([0-9.]+)\s*A',
        r'Current\s+capacity[:\s]*([0-9.]+)\s*A',
        r'([0-9.]+)\s*A(?!\w)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return None


def extract_applications(text: str) -> List[str]:
    """🟢 GREEN: Extraer aplicaciones del cable"""
    applications = []
    
    # Keywords que indican aplicaciones específicas
    app_keywords = {
        'mining': ['mining', 'mine', 'underground', 'dredges', 'mobile equipment'],
        'industrial': ['industrial', 'stationary equipment'],
        'utility': ['distribution', 'transmission', 'power'],
        'marine': ['marine', 'offshore']
    }
    
    text_lower = text.lower()
    for app, keywords in app_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            applications.append(app)
    
    return applications if applications else ['general']


def extract_weight(text: str) -> Optional[float]:
    """🟢 GREEN: Extraer peso del cable"""
    patterns = [
        r'Approximate\s+weight\s+([0-9.]+)\s*kg/km',
        r'Weight[:\s]*([0-9.]+)\s*kg/km',
        r'([0-9.]+)\s*kg/m'  # Convert to kg/km
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            weight = float(match.group(1))
            if i == 2:  # kg/m pattern
                weight *= 1000  # Convert to kg/km
            return weight
    
    return None


def extract_conductor_section(text: str) -> Optional[float]:
    """🟢 GREEN: Extraer sección del conductor"""
    patterns = [
        r'Conductor\s+cross-section\s+([0-9.]+)\s*mm²',
        r'Cross-sectional\s+area[:\s]*([0-9.]+)\s*mm²',
        r'([0-9.]+)\s*mm²'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return None


def extract_outer_diameter(text: str) -> Optional[float]:
    """🟢 GREEN: Extraer diámetro exterior"""
    patterns = [
        r'Outer\s+Diameter\s+([0-9.]+)\s*mm',
        r'External\s+diameter[:\s]*([0-9.]+)\s*mm'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return None


def extract_multiple_cables(pdf_directory: str, max_files: int = 10) -> List[CableExtractedData]:
    """🟢 GREEN: Procesar múltiples PDFs"""
    results = []
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))[:max_files]
    
    for pdf_file in pdf_files:
        try:
            cable_data = extract_cable_data(str(pdf_file))
            results.append(cable_data)
        except Exception as e:
            # Log error but continue
            print(f"Error processing {pdf_file}: {e}")
            continue
    
    return results


def extract_technical_specs(text: str) -> CableExtractedData:
    """🟢 GREEN: Extraer especificaciones técnicas completas"""
    return CableExtractedData(
        nexans_ref="000000000",  # Will be overridden
        conductor_section_mm2=extract_conductor_section(text),
        current_rating=extract_current_rating(text),
        outer_diameter_mm=extract_outer_diameter(text),
        weight_kg_per_km=extract_weight(text)
    )


def assess_manufacturing_complexity(specs: Dict) -> int:
    """🟢 GREEN: Assessment básico de complejidad"""
    complexity = 1  # Base
    
    if specs.get('layers', 0) > 3:
        complexity += 1
    if specs.get('special_insulation', False):
        complexity += 1
    if specs.get('armored', False):
        complexity += 1
    if specs.get('fire_resistant', False):
        complexity += 1
    
    return min(complexity, 5)  # Max 5


def convert_to_cable_model(data: Dict) -> 'CableProduct':
    """🟢 GREEN: Convertir a modelo Pydantic"""
    from src.models.cable import CableProduct, ApplicationArea
    from decimal import Decimal
    
    # Map applications
    app_mapping = {
        'mining': ApplicationArea.MINING,
        'industrial': ApplicationArea.INDUSTRIAL,
        'utility': ApplicationArea.UTILITY,
        'residential': ApplicationArea.RESIDENTIAL
    }
    
    applications = []
    for app in data.get('applications', []):
        if app in app_mapping:
            applications.append(app_mapping[app])
    
    return CableProduct(
        nexans_ref=data['nexans_ref'],
        product_name=data.get('product_name'),
        copper_content_kg=Decimal(str(data.get('copper_kg_per_km', 0))) if data.get('copper_kg_per_km') else None,
        aluminum_content_kg=Decimal(str(data.get('aluminum_kg_per_km', 0))) if data.get('aluminum_kg_per_km') else None,
        voltage_rating=data.get('voltage_rating'),
        current_rating=Decimal(str(data.get('current_rating', 0))) if data.get('current_rating') else None,
        applications=applications
    )


# Add alias for the function name expected by app.py
def extract_cable_data_from_pdf(pdf_path: str) -> CableExtractedData:
    """Alias for extract_cable_data to match app.py imports"""
    return extract_cable_data(pdf_path)