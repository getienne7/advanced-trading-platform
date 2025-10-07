"""
Tests for Strategy Service
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from ..services import StrategyService
from ..schemas import StrategyCreate
from ..models import Strategy, StrategyRating

@pytest.fixture
def mock_db():
    """Mock database session"""
    return Mock(spec=Session)

@pytest.fixture
def strategy_service():
    """Strategy service instance"""
    return StrategyService()

@pytest.fixture
def sample_strategy_data():
    """Sample strategy creation data"""
    return StrategyCreate(
        name="Test Momentum Strategy",
        description="A test momentum trading strategy for unit testing",
        category="momentum",
        risk_level="medium",
        min_capital=1000.0,
        max_drawdown=0.15,
        parameters={"rsi_period": 14, "ma_period": 20},
        subscription_fee=50.0,
        performance_fee=0.2,
        is_public=True
    )

class TestStrategyService:
    """Test cases for StrategyService"""
    
    @pytest.mark.asyncio
    async def test_publish_strategy_success(self, strategy_service, mock_db, sample_strategy_data):
        """Test successful strategy publication"""
        creator_id = str(uuid.uuid4())
        
        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock the strategy object that would be created
        mock_strategy = Mock()
        mock_strategy.id = uuid.uuid4()
        mock_strategy.name = sample_strategy_data.name
        mock_strategy.description = sample_strategy_data.description
        mock_strategy.creator_id = creator_id
        mock_strategy.category = sample_strategy_data.category
        
        # Mock external service calls
        with patch.object(strategy_service, '_get_user_name', return_value="TestUser"):
            with patch.object(strategy_service, '_register_strategy_with_engine', return_value=None):
                # Mock the Strategy constructor to return our mock
                with patch('advanced_trading_platform.services.strategy_marketplace.services.Strategy', return_value=mock_strategy):
                    result = await strategy_service.publish_strategy(sample_strategy_data, creator_id, mock_db)
                    
                    # Verify database operations were called
                    mock_db.add.assert_called_once()
                    mock_db.commit.assert_called_once()
                    mock_db.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_strategies_with_filters(self, strategy_service, mock_db):
        """Test getting strategies with filters"""
        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db.query.return_value = mock_query
        
        # Test with filters
        result = await strategy_service.get_strategies(
            mock_db, 
            skip=0, 
            limit=10, 
            category="momentum", 
            min_rating=4.0,
            sort_by="rating"
        )
        
        # Verify query was built correctly
        assert mock_db.query.called
        assert mock_query.filter.called
        assert mock_query.order_by.called
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_rate_strategy_new_rating(self, strategy_service, mock_db):
        """Test rating a strategy for the first time"""
        strategy_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        
        from ..schemas import RatingCreate
        rating_data = RatingCreate(
            rating=4.5,
            review="Great strategy!",
            performance_rating=4.0,
            risk_rating=5.0,
            communication_rating=4.5
        )
        
        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No existing rating
        mock_db.add = Mock()
        mock_db.commit = Mock()
        
        # Mock helper methods
        with patch.object(strategy_service, '_get_user_name', return_value="TestUser"):
            with patch.object(strategy_service, '_is_verified_subscriber', return_value=True):
                with patch.object(strategy_service, '_update_strategy_rating', return_value=None):
                    result = await strategy_service.rate_strategy(strategy_id, user_id, rating_data, mock_db)
                    
                    assert result["message"] == "Rating submitted successfully"
                    mock_db.add.assert_called_once()
                    mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_marketplace_stats(self, strategy_service, mock_db):
        """Test getting marketplace statistics"""
        # Mock database queries
        mock_db.query.return_value.filter.return_value.count.return_value = 100
        mock_db.query.return_value.scalar.return_value = 500
        mock_db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [
            ("momentum", 30), ("arbitrage", 25), ("mean_reversion", 20)
        ]
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        result = await strategy_service.get_marketplace_stats(mock_db)
        
        assert result.total_strategies == 100
        assert result.active_strategies == 100
        assert result.total_subscribers == 500
        assert "momentum" in result.categories
        assert result.categories["momentum"] == 30
    
    @pytest.mark.asyncio
    async def test_get_user_strategies(self, strategy_service, mock_db):
        """Test getting strategies by user"""
        creator_id = str(uuid.uuid4())
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        result = await strategy_service.get_user_strategies(creator_id, mock_db)
        
        assert isinstance(result, list)
        mock_db.query.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_strategy_by_id_found(self, strategy_service, mock_db):
        """Test getting strategy by ID when it exists"""
        strategy_id = str(uuid.uuid4())
        
        # Mock strategy object
        mock_strategy = Mock()
        mock_strategy.id = strategy_id
        mock_strategy.name = "Test Strategy"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_strategy
        
        with patch('advanced_trading_platform.services.strategy_marketplace.services.StrategyResponse') as mock_response:
            mock_response.from_orm.return_value = mock_strategy
            result = await strategy_service.get_strategy_by_id(strategy_id, mock_db)
            
            assert result is not None
            mock_response.from_orm.assert_called_once_with(mock_strategy)
    
    @pytest.mark.asyncio
    async def test_get_strategy_by_id_not_found(self, strategy_service, mock_db):
        """Test getting strategy by ID when it doesn't exist"""
        strategy_id = str(uuid.uuid4())
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await strategy_service.get_strategy_by_id(strategy_id, mock_db)
        
        assert result is None

if __name__ == "__main__":
    pytest.main([__file__])