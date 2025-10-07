"""
Test suite for advanced order types and execution.
"""
import asyncio
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from advanced_order_types import (
    AdvancedOrderExecutor, AdvancedOrderRequest, AdvancedOrderType,
    IcebergOrderConfig, TWAPConfig, SmartSplitConfig,
    ExecutionStatus, create_advanced_order_executor
)


class MockSmartRouter:
    """Mock smart router for testing."""
    
    async def route_order(self, request):
        return {
            'selected_exchange': 'binance',
            'reason': 'best_price',
            'expected_price': 50000.0
        }
    
    async def execute_with_failover(self, request, decision):
        return {
            'success': True,
            'exchange': 'binance',
            'result': {
                'filled_quantity': request['amount'],
                'average_price': 50000.0,
                'fees': 5.0,
                'order_id': 'test_order_123'
            }
        }


class MockMarketDataService:
    """Mock market data service for testing."""
    
    async def get_ticker(self, symbol):
        return {
            'bid': 49950.0,
            'ask': 50050.0,
            'last': 50000.0
        }
    
    async def get_order_book(self, symbol, limit=20):
        return {
            'bids': [[49950.0, 1.0], [49940.0, 2.0], [49930.0, 3.0]],
            'asks': [[50050.0, 1.0], [50060.0, 2.0], [50070.0, 3.0]]
        }
    
    async def get_recent_trades(self, symbol, limit=100):
        return [
            {'price': 50000.0, 'amount': 0.1, 'timestamp': datetime.utcnow()},
            {'price': 49995.0, 'amount': 0.2, 'timestamp': datetime.utcnow()},
            {'price': 50005.0, 'amount': 0.15, 'timestamp': datetime.utcnow()}
        ]


class MockExchangeClient:
    """Mock exchange client for testing."""
    
    def __init__(self, name):
        self.name = name
    
    async def place_order(self, symbol, side, type, amount, price=None):
        return {
            'order_id': f'{self.name}_order_123',
            'filled_quantity': amount,
            'average_price': 50000.0,
            'fees': amount * 0.001  # 0.1% fee
        }
    
    async def fetch_ticker(self, symbol):
        return {
            'bid': 49950.0,
            'ask': 50050.0,
            'last': 50000.0,
            'baseVolume': 1000.0,
            'quoteVolume': 50000000.0
        }
    
    async def fetch_order_book(self, symbol, limit=10):
        return {
            'bids': [[49950.0, 1.0], [49940.0, 2.0]],
            'asks': [[50050.0, 1.0], [50060.0, 2.0]]
        }


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    smart_router = MockSmartRouter()
    market_data_service = MockMarketDataService()
    exchange_clients = {
        'binance': MockExchangeClient('binance'),
        'coinbase': MockExchangeClient('coinbase'),
        'kraken': MockExchangeClient('kraken')
    }
    
    return smart_router, market_data_service, exchange_clients


@pytest.fixture
def advanced_executor(mock_services):
    """Create advanced order executor for testing."""
    smart_router, market_data_service, exchange_clients = mock_services
    return create_advanced_order_executor(smart_router, market_data_service, exchange_clients)


@pytest.mark.asyncio
async def test_iceberg_order_creation():
    """Test iceberg order creation and validation."""
    config = IcebergOrderConfig(
        total_quantity=Decimal('10.0'),
        visible_quantity=Decimal('1.0'),
        price_improvement_threshold_bps=5.0,
        max_slices=20,
        slice_interval_seconds=1.0,  # Short interval for testing
        randomize_timing=False,  # Disable for predictable testing
        randomize_quantity=False
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_iceberg_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=config,
        user_id="test_user",
        strategy_id="test_strategy"
    )
    
    assert order_request.order_id == "test_iceberg_001"
    assert order_request.order_type == AdvancedOrderType.ICEBERG
    assert config.total_quantity == Decimal('10.0')
    assert config.visible_quantity == Decimal('1.0')


@pytest.mark.asyncio
async def test_twap_order_creation():
    """Test TWAP order creation and validation."""
    config = TWAPConfig(
        total_quantity=Decimal('5.0'),
        duration_minutes=10,  # Short duration for testing
        slice_interval_minutes=1.0,
        participation_rate=0.15,
        adaptive_sizing=True,
        market_impact_threshold_bps=25.0
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_twap_001",
        symbol="ETH/USDT",
        side="sell",
        order_type=AdvancedOrderType.TWAP,
        config=config,
        user_id="test_user",
        strategy_id="test_strategy"
    )
    
    assert order_request.order_id == "test_twap_001"
    assert order_request.order_type == AdvancedOrderType.TWAP
    assert config.total_quantity == Decimal('5.0')
    assert config.duration_minutes == 10


@pytest.mark.asyncio
async def test_smart_split_order_creation():
    """Test smart split order creation and validation."""
    config = SmartSplitConfig(
        total_quantity=Decimal('20.0'),
        max_exchanges=3,
        min_slice_size=Decimal('0.1'),
        liquidity_threshold=0.05,
        rebalance_interval_seconds=10.0,  # Short interval for testing
        cost_optimization=True
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_split_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.SMART_SPLIT,
        config=config,
        user_id="test_user",
        strategy_id="test_strategy"
    )
    
    assert order_request.order_id == "test_split_001"
    assert order_request.order_type == AdvancedOrderType.SMART_SPLIT
    assert config.total_quantity == Decimal('20.0')
    assert config.max_exchanges == 3


@pytest.mark.asyncio
async def test_order_validation(advanced_executor):
    """Test order validation logic."""
    # Valid iceberg order
    valid_config = IcebergOrderConfig(
        total_quantity=Decimal('10.0'),
        visible_quantity=Decimal('1.0')
    )
    
    valid_order = AdvancedOrderRequest(
        order_id="valid_order",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=valid_config,
        user_id="test_user"
    )
    
    validation_result = await advanced_executor._validate_order(valid_order)
    assert validation_result['valid'] is True
    
    # Invalid order - visible quantity > total quantity
    invalid_config = IcebergOrderConfig(
        total_quantity=Decimal('1.0'),
        visible_quantity=Decimal('10.0')  # Invalid: larger than total
    )
    
    invalid_order = AdvancedOrderRequest(
        order_id="invalid_order",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=invalid_config,
        user_id="test_user"
    )
    
    validation_result = await advanced_executor._validate_order(invalid_order)
    assert validation_result['valid'] is False
    assert "cannot exceed total quantity" in validation_result['error']


@pytest.mark.asyncio
async def test_order_submission(advanced_executor):
    """Test order submission and status tracking."""
    config = IcebergOrderConfig(
        total_quantity=Decimal('2.0'),  # Small quantity for quick testing
        visible_quantity=Decimal('0.5'),
        slice_interval_seconds=0.1  # Very short interval for testing
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_submission_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=config,
        user_id="test_user"
    )
    
    # Submit order
    result = await advanced_executor.submit_advanced_order(order_request)
    assert result['success'] is True
    assert result['order_id'] == "test_submission_001"
    assert result['status'] == ExecutionStatus.ACTIVE
    
    # Check status
    status_result = await advanced_executor.get_order_status("test_submission_001")
    assert status_result['success'] is True
    assert status_result['order_id'] == "test_submission_001"
    
    # Wait a bit for some execution
    await asyncio.sleep(0.5)
    
    # Check status again
    status_result = await advanced_executor.get_order_status("test_submission_001")
    assert status_result['success'] is True


@pytest.mark.asyncio
async def test_order_cancellation(advanced_executor):
    """Test order cancellation."""
    config = IcebergOrderConfig(
        total_quantity=Decimal('10.0'),
        visible_quantity=Decimal('1.0'),
        slice_interval_seconds=1.0
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_cancel_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=config,
        user_id="test_user"
    )
    
    # Submit order
    submit_result = await advanced_executor.submit_advanced_order(order_request)
    assert submit_result['success'] is True
    
    # Cancel order
    cancel_result = await advanced_executor.cancel_order("test_cancel_001")
    assert cancel_result['success'] is True
    assert cancel_result['status'] == ExecutionStatus.CANCELLED.value
    
    # Check final status
    status_result = await advanced_executor.get_order_status("test_cancel_001")
    assert status_result['success'] is True
    assert status_result['status'] == ExecutionStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_active_orders_listing(advanced_executor):
    """Test listing active orders."""
    # Submit multiple orders
    for i in range(3):
        config = IcebergOrderConfig(
            total_quantity=Decimal('5.0'),
            visible_quantity=Decimal('1.0'),
            slice_interval_seconds=2.0
        )
        
        order_request = AdvancedOrderRequest(
            order_id=f"test_list_{i:03d}",
            symbol="BTC/USDT",
            side="buy",
            order_type=AdvancedOrderType.ICEBERG,
            config=config,
            user_id="test_user"
        )
        
        result = await advanced_executor.submit_advanced_order(order_request)
        assert result['success'] is True
    
    # List active orders
    active_orders = await advanced_executor.list_active_orders()
    assert active_orders['success'] is True
    assert active_orders['count'] >= 3  # At least the 3 we just submitted


@pytest.mark.asyncio
async def test_market_data_integration(advanced_executor):
    """Test market data integration."""
    # Test getting market data
    market_data = await advanced_executor._get_market_data("BTC/USDT")
    assert market_data is not None
    assert 'ticker' in market_data
    assert 'order_book' in market_data
    assert 'trades' in market_data
    
    # Test price extraction
    current_price = advanced_executor._get_current_price(market_data, "buy")
    assert current_price is not None
    assert current_price > 0


@pytest.mark.asyncio
async def test_execution_metrics_calculation(advanced_executor):
    """Test execution metrics calculation."""
    config = IcebergOrderConfig(
        total_quantity=Decimal('1.0'),  # Small for quick completion
        visible_quantity=Decimal('0.5'),
        slice_interval_seconds=0.1
    )
    
    order_request = AdvancedOrderRequest(
        order_id="test_metrics_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=config,
        user_id="test_user"
    )
    
    # Submit and wait for completion
    result = await advanced_executor.submit_advanced_order(order_request)
    assert result['success'] is True
    
    # Wait for execution to complete
    await asyncio.sleep(1.0)
    
    # Get metrics
    metrics_result = await advanced_executor.get_execution_metrics("test_metrics_001")
    if metrics_result['success']:
        metrics = metrics_result['metrics']
        assert 'total_quantity' in metrics
        assert 'filled_quantity' in metrics
        assert 'execution_time_seconds' in metrics
        assert 'slices_count' in metrics


if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        print("Running simple advanced order types test...")
        
        # Create mock services
        smart_router = MockSmartRouter()
        market_data_service = MockMarketDataService()
        exchange_clients = {
            'binance': MockExchangeClient('binance'),
            'coinbase': MockExchangeClient('coinbase')
        }
        
        # Create executor
        executor = create_advanced_order_executor(
            smart_router, market_data_service, exchange_clients
        )
        
        # Test iceberg order
        config = IcebergOrderConfig(
            total_quantity=Decimal('2.0'),
            visible_quantity=Decimal('0.5'),
            slice_interval_seconds=0.2
        )
        
        order_request = AdvancedOrderRequest(
            order_id="simple_test_001",
            symbol="BTC/USDT",
            side="buy",
            order_type=AdvancedOrderType.ICEBERG,
            config=config,
            user_id="test_user"
        )
        
        # Submit order
        result = await executor.submit_advanced_order(order_request)
        print(f"Order submission result: {result}")
        
        # Wait a bit
        await asyncio.sleep(1.0)
        
        # Check status
        status = await executor.get_order_status("simple_test_001")
        print(f"Order status: {status}")
        
        # List active orders
        active = await executor.list_active_orders()
        print(f"Active orders: {active}")
        
        print("Simple test completed successfully!")
    
    # Run the test
    asyncio.run(run_simple_test())