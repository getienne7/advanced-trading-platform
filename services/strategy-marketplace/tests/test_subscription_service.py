"""
Tests for Subscription Service
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from ..services import SubscriptionService
from ..schemas import SubscriptionCreate, CopyTradeSignal
from ..models import StrategySubscription, Strategy

@pytest.fixture
def mock_db():
    """Mock database session"""
    return Mock(spec=Session)

@pytest.fixture
def subscription_service():
    """Subscription service instance"""
    return SubscriptionService()

@pytest.fixture
def sample_subscription_data():
    """Sample subscription creation data"""
    return SubscriptionCreate(
        allocation_percentage=15.0,
        max_position_size=1000.0,
        risk_multiplier=1.2,
        auto_trade=True
    )

@pytest.fixture
def sample_copy_trade_signal():
    """Sample copy trade signal"""
    return CopyTradeSignal(
        strategy_id=uuid.uuid4(),
        signal_type="open",
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        price=50000.0,
        stop_loss=48000.0,
        take_profit=52000.0,
        timestamp=datetime.utcnow(),
        metadata={}
    )

class TestSubscriptionService:
    """Test cases for SubscriptionService"""
    
    @pytest.mark.asyncio
    async def test_create_subscription_success(self, subscription_service, mock_db, sample_subscription_data):
        """Test successful subscription creation"""
        strategy_id = str(uuid.uuid4())
        subscriber_id = str(uuid.uuid4())
        
        # Mock database operations
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No existing subscription
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock strategy object
        mock_strategy = Mock()
        mock_strategy.total_subscribers = 5
        mock_db.query.return_value.filter.return_value.first.side_effect = [None, mock_strategy]
        
        # Mock external service calls
        with patch.object(subscription_service, '_get_user_name', return_value="TestUser"):
            with patch.object(subscription_service, '_register_copy_trading', return_value=None):
                with patch('advanced_trading_platform.services.strategy_marketplace.services.StrategySubscription') as mock_sub_class:
                    mock_subscription = Mock()
                    mock_subscription.id = uuid.uuid4()
                    mock_sub_class.return_value = mock_subscription
                    
                    result = await subscription_service.create_subscription(
                        strategy_id, subscriber_id, sample_subscription_data, mock_db
                    )
                    
                    assert "Successfully subscribed" in result["message"]
                    assert "subscription_id" in result
                    mock_db.add.assert_called()
                    mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_create_subscription_already_exists(self, subscription_service, mock_db, sample_subscription_data):
        """Test subscription creation when already subscribed"""
        strategy_id = str(uuid.uuid4())
        subscriber_id = str(uuid.uuid4())
        
        # Mock existing subscription
        existing_subscription = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = existing_subscription
        
        with pytest.raises(ValueError, match="Already subscribed"):
            await subscription_service.create_subscription(
                strategy_id, subscriber_id, sample_subscription_data, mock_db
            )
    
    @pytest.mark.asyncio
    async def test_cancel_subscription_success(self, subscription_service, mock_db):
        """Test successful subscription cancellation"""
        strategy_id = str(uuid.uuid4())
        subscriber_id = str(uuid.uuid4())
        
        # Mock existing subscription
        mock_subscription = Mock()
        mock_subscription.is_active = True
        
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.total_subscribers = 10
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_subscription, mock_strategy]
        mock_db.commit = Mock()
        
        with patch.object(subscription_service, '_unregister_copy_trading', return_value=None):
            result = await subscription_service.cancel_subscription(strategy_id, subscriber_id, mock_db)
            
            assert result is True
            assert mock_subscription.is_active is False
            mock_db.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_cancel_subscription_not_found(self, subscription_service, mock_db):
        """Test subscription cancellation when subscription doesn't exist"""
        strategy_id = str(uuid.uuid4())
        subscriber_id = str(uuid.uuid4())
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = await subscription_service.cancel_subscription(strategy_id, subscriber_id, mock_db)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_user_subscriptions(self, subscription_service, mock_db):
        """Test getting user subscriptions"""
        subscriber_id = str(uuid.uuid4())
        
        # Mock subscription
        mock_subscription = Mock()
        mock_subscription.strategy_id = uuid.uuid4()
        
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.name = "Test Strategy"
        
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_subscription]
        mock_db.query.return_value.filter.return_value.first.return_value = mock_strategy
        
        result = await subscription_service.get_user_subscriptions(subscriber_id, mock_db)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["subscription"] == mock_subscription
        assert result[0]["strategy"] == mock_strategy
    
    @pytest.mark.asyncio
    async def test_process_copy_trade_signal(self, subscription_service, mock_db, sample_copy_trade_signal):
        """Test processing copy trade signal"""
        # Mock active subscriptions
        mock_subscription1 = Mock()
        mock_subscription1.id = uuid.uuid4()
        mock_subscription1.allocation_percentage = 10.0
        mock_subscription1.risk_multiplier = 1.0
        mock_subscription1.max_position_size = None
        
        mock_subscription2 = Mock()
        mock_subscription2.id = uuid.uuid4()
        mock_subscription2.allocation_percentage = 20.0
        mock_subscription2.risk_multiplier = 1.5
        mock_subscription2.max_position_size = 500.0
        
        mock_db.query.return_value.filter.return_value.all.return_value = [
            mock_subscription1, mock_subscription2
        ]
        
        with patch.object(subscription_service, '_execute_copy_trade', return_value=None) as mock_execute:
            await subscription_service.process_copy_trade_signal(sample_copy_trade_signal, mock_db)
            
            # Should be called twice, once for each subscription
            assert mock_execute.call_count == 2
            
            # Check that quantities were adjusted correctly
            call_args_list = mock_execute.call_args_list
            
            # First subscription: 0.1 * 0.1 * 1.0 = 0.01
            assert call_args_list[0][0][2] == 0.01  # adjusted quantity
            
            # Second subscription: min(0.1 * 0.2 * 1.5, 500.0) = 0.03
            assert call_args_list[1][0][2] == 0.03  # adjusted quantity
    
    @pytest.mark.asyncio
    async def test_process_copy_trade_signal_with_max_position_limit(self, subscription_service, mock_db):
        """Test copy trade signal processing with position size limits"""
        signal = CopyTradeSignal(
            strategy_id=uuid.uuid4(),
            signal_type="open",
            symbol="BTCUSDT",
            side="buy",
            quantity=10.0,  # Large quantity
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        # Mock subscription with small max position size
        mock_subscription = Mock()
        mock_subscription.id = uuid.uuid4()
        mock_subscription.allocation_percentage = 100.0  # 100%
        mock_subscription.risk_multiplier = 2.0  # 2x
        mock_subscription.max_position_size = 5.0  # Max 5.0
        
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_subscription]
        
        with patch.object(subscription_service, '_execute_copy_trade', return_value=None) as mock_execute:
            await subscription_service.process_copy_trade_signal(signal, mock_db)
            
            # Should be limited to max_position_size
            # Calculated: 10.0 * 1.0 * 2.0 = 20.0, but limited to 5.0
            call_args = mock_execute.call_args[0]
            assert call_args[2] == 5.0  # Should be limited to max_position_size

if __name__ == "__main__":
    pytest.main([__file__])