"""
🔴 RED PHASE - Test que debe fallar primero
Tarea 1.1.1: Project Setup - Escribiendo tests PRIMERO según plan TDD
"""

import os
import pytest
from pathlib import Path


def test_project_structure_exists():
    """🔴 RED: Test estructura directorios existe - DEBE FALLAR PRIMERO"""
    base_path = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai")
    
    # Estructura requerida según plan
    required_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "src/models",
        "src/agents",
        "src/pricing",
        "src/api",
        "data/processed",
        "docs"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Directory {dir_path} should exist"
        assert full_path.is_dir(), f"{dir_path} should be a directory"


def test_requirements_file_exists():
    """🔴 RED: Test requirements.txt existe con dependencies correctas"""
    req_file = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai/requirements.txt")
    
    assert req_file.exists(), "requirements.txt should exist"
    
    # Leer contenido
    content = req_file.read_text()
    
    # Verificar dependencies críticas para pricing AI
    required_deps = [
        "pytest",
        "fastapi", 
        "pydantic",
        "pandas",
        "scikit-learn",
        "xgboost",
        "requests",
        "sqlalchemy",
        "streamlit",
        "PyMuPDF"
    ]
    
    for dep in required_deps:
        assert dep in content, f"Dependency {dep} should be in requirements.txt"


def test_pytest_config_exists():
    """🔴 RED: Test pytest.ini configurado correctamente"""
    pytest_file = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai/pytest.ini")
    
    assert pytest_file.exists(), "pytest.ini should exist"
    
    content = pytest_file.read_text()
    assert "testpaths = tests" in content
    assert "--cov" in content, "Coverage should be configured"


def test_claude_md_exists():
    """🔴 RED: Test CLAUDE.md específico del proyecto existe"""
    claude_file = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai/CLAUDE.md")
    
    assert claude_file.exists(), "CLAUDE.md should exist"
    
    content = claude_file.read_text()
    assert "Nexans Pricing Intelligence" in content
    assert "TDD MANDATORY" in content
    assert "RED-GREEN-REFACTOR" in content


def test_readme_exists():
    """🔴 RED: Test README.md del proyecto existe"""
    readme_file = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai/README.md")
    
    assert readme_file.exists(), "README.md should exist"
    
    content = readme_file.read_text()
    assert "Sistema de Pricing Inteligente" in content
    assert "TDD" in content


def test_init_files_exist():
    """🔴 RED: Test __init__.py files en directorios Python"""
    base_path = Path("/Users/autonomos_dev/Projects/autonomos_nexans/nexans_pricing_ai")
    
    python_dirs = [
        "src",
        "src/models", 
        "src/agents",
        "src/pricing",
        "src/api",
        "tests"
    ]
    
    for dir_path in python_dirs:
        init_file = base_path / dir_path / "__init__.py"
        assert init_file.exists(), f"__init__.py should exist in {dir_path}"