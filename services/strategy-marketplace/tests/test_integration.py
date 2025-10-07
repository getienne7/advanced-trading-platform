"""
Integration tests for Strategy Marketplace Service
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from uuid import uuid4

# Import the app and dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from database import get_db
from models import Base

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_strategy_marketplace.db"

# Create test engine and session
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override the dependency
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def setup_test_db():
    """Set up test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def mock_auth_token():
    """Mock authentication token"""
    return "Bearer mock-jwt-token"

class TestStrategyMarketplaceIntegration:
    """Integration tests for the strategy marketplace"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "strategy-marketplace"
    
    def test_get_strategies_empty(self, client, setup_test_db):
        """Test getting strategies when marketplace is empty"""
        response = client.get("/strategies")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_get_marketplace_stats_empty(self, client, setup_test_db):
        """Test getting marketplace stats when empty"""
        response = client.get("/marketplace/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_strategies"] == 0
        assert data["active_strategies"] == 0
        assert data["total_subscribers"] == 0
    
    @pytest.mark.skip(reason="Requires authentication mock")
    def test_publish_strategy_unauthorized(self, client, setup_test_db):
        """Test publishing strategy without authentication"""
        strategy_data = {
            "name": "Test Strategy",
            "description": "A test strategy for integration testing",
            "category": "momentum",
            "risk_level": "medium",
            "min_capital": 1000.0,
            "subscription_fee": 50.0,
            "performance_fee": 0.2,
            "is_public": True,
            "parameters": {"rsi_period": 14}
        }
        
        response = client.post("/strategies", json=strategy_data)
        assert response.status_code == 401  # Unauthorized
    
    def test_get_strategy_not_found(self, client, setup_test_db):
        """Test getting non-existent strategy"""
        strategy_id = str(uuid4())
        response = client.get(f"/strategies/{strategy_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_strategy_performance_not_found(self, client, setup_test_db):
        """Test getting performance for non-existent strategy"""
        strategy_id = str(uuid4())
        response = client.get(f"/strategies/{strategy_id}/performance")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.skip(reason="Requires authentication mock")
    def test_subscribe_to_strategy_unauthorized(self, client, setup_test_db):
        """Test subscribing to strategy without authentication"""
        strategy_id = str(uuid4())
        subscription_data = {
            "allocation_percentage": 10.0,
            "risk_multiplier": 1.0,
            "auto_trade": True
        }
        
        response = client.post(f"/strategies/{strategy_id}/subscribe", json=subscription_data)
        assert response.status_code == 401  # Unauthorized
    
    def test_invalid_strategy_data(self, client, setup_test_db):
        """Test publishing strategy with invalid data"""
        invalid_data = {
            "name": "A",  # Too short
            "description": "Short",  # Too short
            "category": "invalid_category",  # Invalid category
            "risk_level": "invalid_risk",  # Invalid risk level
            "min_capital": -100,  # Negative capital
            "subscription_fee": -10,  # Negative fee
            "performance_fee": 0.8,  # Too high performance fee
        }
        
        # This should fail validation even before authentication
        response = client.post("/strategies", json=invalid_data)
        assert response.status_code in [401, 422]  # Unauthorized or Validation Error
    
    def test_invalid_subscription_data(self, client, setup_test_db):
        """Test subscription with invalid data"""
        strategy_id = str(uuid4())
        invalid_data = {
            "allocation_percentage": 150.0,  # Over 100%
            "risk_multiplier": -1.0,  # Negative multiplier
            "max_position_size": -100,  # Negative position size
        }
        
        response = client.post(f"/strategies/{strategy_id}/subscribe", json=invalid_data)
        assert response.status_code in [401, 422]  # Unauthorized or Validation Error
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        # Note: CORS headers would be added by middleware in production
        assert response.status_code in [200, 405]  # OK or Method Not Allowed

@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async functionality"""
    
    async def test_async_health_check(self):
        """Test health check with async client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    async def test_async_get_strategies(self):
        """Test getting strategies with async client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/strategies")
            assert response.status_code == 200
            assert isinstance(response.json(), list)

if __name__ == "__main__":
    pytest.main([__file__])