#!/usr/bin/env python3
"""
🚀 NEXANS PRICING INTELLIGENCE - QUICK START
Despliegue rápido local sin Docker para demo inmediato
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def print_header(title):
    print("\n" + "="*60)
    print(f"🏭 {title}")
    print("="*60)

def print_success(msg):
    print(f"✅ {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required")
        sys.exit(1)
    print_success(f"Python {sys.version.split()[0]} OK")

def install_dependencies():
    """Install minimal dependencies"""
    print_info("Installing core dependencies...")
    
    core_deps = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "pydantic==2.5.0",
        "requests==2.31.0",
        "streamlit==1.28.2",
        "plotly==5.17.0",
        "pandas==2.1.3",
        "python-dateutil==2.8.2",
        "python-multipart==0.0.6"
    ]
    
    for dep in core_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print_success(f"Installed {dep.split('==')[0]}")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {dep}: {e}")

def start_api():
    """Start FastAPI in background"""
    print_info("Starting FastAPI API...")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    api_process = subprocess.Popen(
        [sys.executable, "app.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to start
    time.sleep(5)
    
    # Check if API is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print_success("FastAPI started successfully")
            return api_process
        else:
            print_error("API health check failed")
            return None
    except Exception as e:
        print_error(f"API connection failed: {e}")
        return None

def start_dashboard():
    """Start Streamlit dashboard"""
    print_info("Starting Streamlit Dashboard...")
    
    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py", 
         "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3)
    print_success("Dashboard started successfully")
    return dashboard_process

def main():
    print_header("NEXANS PRICING INTELLIGENCE QUICK START")
    print("🎯 Demo rápido para Gerardo Iniescar (CIO D&U AMEA)")
    
    # Store process PIDs for cleanup
    processes = []
    
    def cleanup(signum=None, frame=None):
        print_info("Stopping services...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print_success("All services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Check prerequisites
        check_python()
        
        # Install dependencies
        install_dependencies()
        
        # Start API
        api_process = start_api()
        if api_process:
            processes.append(api_process)
        
        # Start Dashboard
        dashboard_process = start_dashboard()
        if dashboard_process:
            processes.append(dashboard_process)
        
        # Show access info
        print_header("SISTEMA LISTO - ACCESO")
        print("🌐 URLs de acceso:")
        print("   📊 FastAPI API: http://localhost:8000")
        print("   📖 API Docs: http://localhost:8000/docs")
        print("   🎨 Dashboard: http://localhost:8501")
        print()
        print("🤖 Características disponibles:")
        print("   ✅ Market Intelligence Agent")
        print("   ✅ Demand Forecasting Agent") 
        print("   ✅ Quote Generation Agent")
        print("   ✅ Real-time Pricing Calculator")
        print("   ✅ Interactive Dashboard")
        print()
        print("⏹️  Para detener: Ctrl+C")
        print("🎊 Sistema listo para demo!")
        
        # Keep running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process and api_process.poll() is not None:
                print_error("API process died")
                break
                
            if dashboard_process and dashboard_process.poll() is not None:
                print_error("Dashboard process died") 
                break
    
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print_error(f"Error: {e}")
        cleanup()

if __name__ == "__main__":
    main()