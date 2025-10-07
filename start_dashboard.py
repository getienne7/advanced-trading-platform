#!/usr/bin/env python3
"""
Startup script for the Advanced Trading Platform Dashboard
Starts both the AI/ML backend service and the React frontend
"""
import subprocess
import sys
import time
import os
from pathlib import Path
import threading
import signal

def start_backend():
    """Start the AI/ML backend service"""
    print("🚀 Starting AI/ML Backend Service...")
    
    # Change to the AI/ML service directory
    backend_dir = Path(__file__).parent / "services" / "ai-ml"
    
    try:
        # Start the FastAPI service
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print backend output
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                print(f"[Backend] {line.strip()}")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting React Frontend...")
    
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    
    try:
        # Check if node_modules exists, if not install dependencies
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            install_process = subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                capture_output=True,
                text=True
            )
            
            if install_process.returncode != 0:
                print(f"❌ Failed to install dependencies: {install_process.stderr}")
                return None
        
        # Start the React development server
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print frontend output
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                print(f"[Frontend] {line.strip()}")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import tensorflow
        import sklearn
        print("✅ Python dependencies OK")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js OK ({result.stdout.strip()})")
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        print("💡 Install Node.js from https://nodejs.org/")
        return False
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm OK ({result.stdout.strip()})")
        else:
            print("❌ npm not found")
            return False
    except FileNotFoundError:
        print("❌ npm not found")
        return False
    
    return True

def main():
    """Main function to start the dashboard"""
    print("🚀 Advanced Trading Platform Dashboard Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        return 1
    
    print("\n🎯 Starting services...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)
    
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    print("\n🎉 Dashboard is starting up!")
    print("📊 Backend API: http://localhost:8005")
    print("📊 API Docs: http://localhost:8005/docs")
    print("🎨 Frontend: http://localhost:3000")
    print("\n💡 Press Ctrl+C to stop all services")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)