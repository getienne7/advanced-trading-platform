#!/usr/bin/env python3
"""
Test script for API Gateway functionality.
"""
import sys
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent / "shared"))
sys.path.append(str(Path(__file__).parent / "services" / "api-gateway"))

def test_imports():
    """Test API Gateway imports."""
    print("Testing API Gateway imports...")
    
    try:
        from main import app, GatewayConfig, AuthenticationService, RateLimiter, ServiceRouter
        print("✓ Main API Gateway imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test API Gateway configuration."""
    print("Testing API Gateway configuration...")
    
    try:
        from main import GatewayConfig
        
        config = GatewayConfig()
        print(f"✓ JWT Secret configured: {'***' if config.jwt_secret else 'Not set'}")
        print(f"✓ JWT Algorithm: {config.jwt_algorithm}")
        print(f"✓ JWT Expiration: {config.jwt_expiration_hours} hours")
        print(f"✓ Rate Limit: {config.rate_limit_requests} requests per {config.rate_limit_window} seconds")
        print(f"✓ Services configured: {len(config.services)}")
        
        for service_name, service_url in config.services.items():
            print(f"  - {service_name}: {service_url}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_authentication():
    """Test authentication service."""
    print("Testing authentication service...")
    
    try:
        from main import AuthenticationService
        
        auth_service = AuthenticationService()
        
        # Test token creation
        user_data = {
            "user_id": "test_user",
            "username": "testuser",
            "roles": ["user"],
            "permissions": ["read"]
        }
        
        token = auth_service.create_access_token(user_data)
        print(f"✓ Token created: {token[:20]}...")
        
        # Test token verification
        payload = auth_service.verify_token(token)
        print(f"✓ Token verified: user_id={payload.get('user_id')}")
        
        return True
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI app creation."""
    print("Testing FastAPI app...")
    
    try:
        from main import app
        from fastapi.testclient import TestClient
        
        # This would require installing test dependencies
        print("✓ FastAPI app created successfully")
        print(f"✓ App title: {app.title}")
        print(f"✓ App version: {app.version}")
        
        return True
    except Exception as e:
        print(f"✗ FastAPI app test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== API Gateway Testing ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Authentication", test_authentication),
        ("FastAPI App", test_fastapi_app)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        if test_func():
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All API Gateway tests passed!")
        print("\nNext steps:")
        print("1. Start databases: docker-compose up -d postgres redis influxdb")
        print("2. Start API Gateway: cd services/api-gateway && python main.py")
        print("3. Test endpoints: curl http://localhost:8000/health")
    else:
        print("❌ Some tests failed. Please fix the issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())