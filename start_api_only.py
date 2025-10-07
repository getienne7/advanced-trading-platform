#!/usr/bin/env python3
"""
Start only the AI/ML API backend service
This allows you to test the API endpoints directly
"""
import subprocess
import sys
import os
from pathlib import Path

def start_api_service():
    """Start the AI/ML API service"""
    print("🚀 Starting Advanced Trading Platform AI/ML API Service")
    print("=" * 60)
    
    # Change to the AI/ML service directory
    service_dir = Path(__file__).parent / "services" / "ai-ml"
    
    if not service_dir.exists():
        print(f"❌ Service directory not found: {service_dir}")
        return 1
    
    if not (service_dir / "main.py").exists():
        print(f"❌ main.py not found in: {service_dir}")
        return 1
    
    print(f"📂 Service directory: {service_dir}")
    print("🔧 Starting FastAPI service...")
    print()
    
    try:
        # Start the FastAPI service
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=service_dir,
            check=False
        )
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return 1

def main():
    """Main function"""
    print("🎯 Advanced Trading Platform - API Service Starter")
    print()
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    expected_files = ["services", "requirements.txt", "README.md"]
    
    missing_files = [f for f in expected_files if not (current_dir / f).exists()]
    if missing_files:
        print("⚠️  Warning: Some expected files/directories are missing:")
        for f in missing_files:
            print(f"   - {f}")
        print()
    
    print("📋 Service Information:")
    print("   🌐 API Base URL: http://localhost:8005")
    print("   📚 API Documentation: http://localhost:8005/docs")
    print("   🔍 Health Check: http://localhost:8005/health")
    print("   📊 Metrics: http://localhost:8005/metrics")
    print()
    
    print("🔗 Available Endpoints:")
    print("   GET  /api/analysis/{symbol}     - Comprehensive AI analysis")
    print("   POST /api/sentiment/analyze     - Sentiment analysis")
    print("   POST /api/predictions/price     - Price predictions")
    print("   GET  /api/regime/{symbol}       - Market regime detection")
    print()
    
    print("💡 Example API Calls:")
    print("   curl http://localhost:8005/health")
    print("   curl http://localhost:8005/api/analysis/BTC/USDT")
    print("   curl http://localhost:8005/api/regime/ETH/USDT")
    print()
    
    print("🎨 Frontend Demo:")
    print("   Open: advanced_trading_platform/dashboard_demo.html")
    print("   This shows how the dashboard would look with live data")
    print()
    
    print("🚀 Starting API service...")
    print("   Press Ctrl+C to stop the service")
    print("=" * 60)
    
    return start_api_service()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)