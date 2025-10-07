"""
Comprehensive tests for the backtesting engine.
Tests historical data replay, realistic execution simulation, and performance metrics calculation.
"""
import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from backtesting_engine import (
    BacktestingEngine, BacktestingConfig, WalkForwardAnalyzer,
    HistoricalDataReplay, SimpleMovingAverageStrategy,
    MarketData, Order, Trade, Position, PerformanceMetrics,
    OrderType, OrderSide, OrderStatus, ExecutionModel
)


class TestBacktestingEngine:
    """Test suite for BacktestingEngine."""
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data for testing."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # Simple price movement with some volatility
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            current_price = base_price * (1 + price_change * i * 0.1)
            
            volatility = abs(price_change) * current_price * 0.5
            high = current_price + volatility
            low = current_price - volatility
            open_price = current_price + np.random.uniform(-volatility/2, volatility/2)
            
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=open_price,
                high=high,
                low=low,
                close=current_price,
                volume=1000.0,
                bid=current_price * 0.999,
                ask=current_price * 1.001,
                spread=current_price * 0.002
            ))
        
        return data
    
    @pytest.fixture
    def backtesting_config(self) -> BacktestingConfig:
        """Create test backtesting configuration."""
        return BacktestingConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            execution_model=ExecutionModel.REALISTIC
        )
    
    @pytest.fixture
    def simple_strategy(self) -> SimpleMovingAverageStrategy:
        """Create simple moving average strategy for testing."""
        return SimpleMovingAverageStrategy(
            "test_ma_strategy",
            {'short_window': 5, 'long_window': 10, 'position_size': 1.0}
        )
    
    @pytest.mark.asyncio
    async def test_backtesting_engine_initialization(self, backtesting_config):
        """Test backtesting engine initialization."""
        engine = BacktestingEngine(backtesting_config)
        
        assert engine.config == backtesting_config
        assert engine.current_capital == backtesting_config.initial_capital
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0
    
    @pytest.mark.asyncio
    async def test_market_order_execution(self, backtesting_config, sample_market_data):
        """Test market order execution with realistic slippage and fees."""
        engine = BacktestingEngine(backtesting_config)
        
        # Create market buy order
        order = Order(
            id="test_order_1",
            timestamp=sample_market_data[0].timestamp,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # Test order fill
        fill_result = await engine._try_fill_order(order, sample_market_data[0])
        
        assert fill_result is True
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 1.0
        assert order.filled_price is not None
        assert order.fees > 0  # Should have fees
        assert order.slippage >= 0  # Should have some slippage
    
    @pytest.mark.asyncio
    async def test_limit_order_execution(self, backtesting_config, sample_market_data):
        """Test limit order execution logic."""
        engine = BacktestingEngine(backtesting_config)
        
        # Create limit buy order below market price
        market_price = sample_market_data[0].close
        limit_price = market_price * 0.99  # 1% below market
        
        order = Order(
            id="test_limit_order",
            timestamp=sample_market_data[0].timestamp,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=limit_price
        )
        
        # Should not fill at current market price
        fill_result = await engine._try_fill_order(order, sample_market_data[0])
        assert fill_result is False
        
        # Create market data where limit should fill
        low_price_data = MarketData(
            timestamp=sample_market_data[0].timestamp + timedelta(hours=1),
            symbol="BTCUSDT",
            open=market_price,
            high=market_price,
            low=limit_price * 0.99,  # Goes below limit price
            close=market_price * 0.995,
            volume=1000.0
        )
        
        fill_result = await engine._try_fill_order(order, low_price_data)
        assert fill_result is True
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_position_management(self, backtesting_config):
        """Test position tracking and P&L calculation."""
        engine = BacktestingEngine(backtesting_config)
        
        # Create buy trade
        buy_trade = Trade(
            id="trade_1",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            fees=50.0,
            slippage=0.0005,
            order_id="order_1"
        )
        
        await engine._execute_trade(buy_trade)
        
        # Check position
        assert "BTCUSDT" in engine.positions
        position = engine.positions["BTCUSDT"]
        assert position.quantity == 1.0
        assert position.average_price == 50000.0
        
        # Create sell trade
        sell_trade = Trade(
            id="trade_2",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.5,
            price=51000.0,
            fees=25.5,
            slippage=0.0005,
            order_id="order_2"
        )
        
        await engine._execute_trade(sell_trade)
        
        # Check updated position
        position = engine.positions["BTCUSDT"]
        assert position.quantity == 0.5
        assert position.realized_pnl == 500.0  # (51000 - 50000) * 0.5
        assert sell_trade.pnl == 500.0
    
    @pytest.mark.asyncio
    async def test_slippage_calculation(self, backtesting_config, sample_market_data):
        """Test slippage calculation based on order size and market conditions."""
        engine = BacktestingEngine(backtesting_config)
        
        # Small order
        small_order = Order(
            id="small_order",
            timestamp=sample_market_data[0].timestamp,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        small_slippage = await engine._calculate_slippage(
            small_order, sample_market_data[0].close, sample_market_data[0]
        )
        
        # Large order
        large_order = Order(
            id="large_order",
            timestamp=sample_market_data[0].timestamp,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0
        )
        
        large_slippage = await engine._calculate_slippage(
            large_order, sample_market_data[0].close, sample_market_data[0]
        )
        
        # Large orders should have more slippage
        assert large_slippage > small_slippage
        assert large_slippage <= 0.01  # Should be capped at 1%
    
    @pytest.mark.asyncio
    async def test_full_backtest_execution(self, backtesting_config, simple_strategy, sample_market_data):
        """Test complete backtest execution."""
        engine = BacktestingEngine(backtesting_config)
        
        result = await engine.run_backtest(
            simple_strategy,
            sample_market_data,
            sample_market_data[0].timestamp,
            sample_market_data[-1].timestamp
        )
        
        # Verify result structure
        assert 'strategy_name' in result
        assert 'performance_metrics' in result
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'final_capital' in result
        
        # Verify performance metrics
        metrics = result['performance_metrics']
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        
        # Verify equity curve
        assert len(result['equity_curve']) > 0
        assert all('timestamp' in point for point in result['equity_curve'])
        assert all('equity' in point for point in result['equity_curve'])
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, backtesting_config, simple_strategy, sample_market_data):
        """Test comprehensive performance metrics calculation."""
        engine = BacktestingEngine(backtesting_config)
        
        # Run backtest
        result = await engine.run_backtest(simple_strategy, sample_market_data)
        metrics = result['performance_metrics']
        
        # Test basic metrics
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.volatility, float)
        
        # Test risk metrics
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert 0 <= metrics.max_drawdown <= 1  # Should be between 0 and 1
        
        # Test trade metrics
        assert isinstance(metrics.win_rate, float)
        assert 0 <= metrics.win_rate <= 1  # Should be between 0 and 1
        assert isinstance(metrics.profit_factor, float)
        assert metrics.total_trades >= 0
        
        # Test advanced metrics
        assert isinstance(metrics.var_95, float)
        assert isinstance(metrics.expected_shortfall, float)
        assert isinstance(metrics.kelly_criterion, float)
    
    @pytest.mark.asyncio
    async def test_order_validation(self, backtesting_config):
        """Test order validation logic."""
        engine = BacktestingEngine(backtesting_config)
        
        # Valid buy order
        valid_order = Order(
            id="valid_order",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        is_valid = await engine._validate_order(valid_order)
        assert is_valid is True
        
        # Invalid order - insufficient capital
        expensive_order = Order(
            id="expensive_order",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=100000.0  # Very expensive
        )
        
        is_valid = await engine._validate_order(expensive_order)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_execution_models(self, sample_market_data):
        """Test different execution models."""
        # Perfect execution
        perfect_config = BacktestingConfig(execution_model=ExecutionModel.PERFECT)
        perfect_engine = BacktestingEngine(perfect_config)
        
        order = Order(
            id="perfect_order",
            timestamp=sample_market_data[0].timestamp,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        perfect_result = await perfect_engine._apply_execution_model(
            order, sample_market_data[0].close, sample_market_data[0]
        )
        
        assert perfect_result['slippage'] == 0.0
        assert perfect_result['fees'] == 0.0
        
        # Realistic execution
        realistic_config = BacktestingConfig(execution_model=ExecutionModel.REALISTIC)
        realistic_engine = BacktestingEngine(realistic_config)
        
        realistic_result = await realistic_engine._apply_execution_model(
            order, sample_market_data[0].close, sample_market_data[0]
        )
        
        assert realistic_result['slippage'] > 0.0
        assert realistic_result['fees'] > 0.0


class TestWalkForwardAnalyzer:
    """Test suite for WalkForwardAnalyzer."""
    
    @pytest.fixture
    def analyzer(self) -> WalkForwardAnalyzer:
        """Create walk-forward analyzer for testing."""
        return WalkForwardAnalyzer(
            training_period_days=30,
            testing_period_days=10,
            step_days=5
        )
    
    @pytest.fixture
    def extended_market_data(self) -> List[MarketData]:
        """Generate extended market data for walk-forward testing."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(200):  # 200 hours of data
            price_change = np.random.normal(0, 0.01)
            current_price = base_price * (1 + price_change * i * 0.05)
            
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=current_price,
                high=current_price * 1.01,
                low=current_price * 0.99,
                close=current_price,
                volume=1000.0
            ))
        
        return data
    
    @pytest.mark.asyncio
    async def test_period_generation(self, analyzer, extended_market_data):
        """Test walk-forward period generation."""
        periods = analyzer._generate_periods(extended_market_data)
        
        assert len(periods) > 0
        
        for period in periods:
            assert 'training_start' in period
            assert 'training_end' in period
            assert 'testing_start' in period
            assert 'testing_end' in period
            
            # Training period should come before testing period
            assert period['training_end'] == period['testing_start']
            
            # Periods should have correct duration
            training_duration = period['training_end'] - period['training_start']
            testing_duration = period['testing_end'] - period['testing_start']
            
            assert training_duration.days == analyzer.training_period_days
            assert testing_duration.days == analyzer.testing_period_days
    
    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, analyzer, extended_market_data):
        """Test complete walk-forward analysis."""
        strategy_parameters = {'short_window': 5, 'long_window': 10, 'position_size': 1.0}
        
        result = await analyzer.run_walk_forward_analysis(
            SimpleMovingAverageStrategy,
            strategy_parameters,
            extended_market_data
        )
        
        # Verify result structure
        assert 'strategy_name' in result
        assert 'parameters' in result
        assert 'periods' in result
        assert 'aggregated_metrics' in result
        assert 'total_periods' in result
        
        # Verify periods
        assert len(result['periods']) > 0
        
        for period_result in result['periods']:
            assert 'period' in period_result
            assert 'training_period' in period_result
            assert 'testing_period' in period_result
            assert 'performance' in period_result
            assert 'trades' in period_result
            assert 'final_capital' in period_result
        
        # Verify aggregated metrics
        aggregated = result['aggregated_metrics']
        assert 'mean_return' in aggregated
        assert 'std_return' in aggregated
        assert 'mean_sharpe' in aggregated
        assert 'consistency_score' in aggregated


class TestHistoricalDataReplay:
    """Test suite for HistoricalDataReplay."""
    
    @pytest.fixture
    def data_replay(self) -> HistoricalDataReplay:
        """Create historical data replay system."""
        return HistoricalDataReplay()
    
    @pytest.mark.asyncio
    async def test_synthetic_data_generation(self, data_replay):
        """Test synthetic data generation."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        data = await data_replay.load_historical_data(
            "BTCUSDT", start_date, end_date, "1h"
        )
        
        assert len(data) > 0
        
        # Verify data structure
        for point in data:
            assert isinstance(point, MarketData)
            assert point.symbol == "BTCUSDT"
            assert start_date <= point.timestamp <= end_date
            assert point.high >= point.low
            assert point.high >= point.open
            assert point.high >= point.close
            assert point.low <= point.open
            assert point.low <= point.close
            assert point.volume > 0
    
    @pytest.mark.asyncio
    async def test_data_caching(self, data_replay):
        """Test data caching functionality."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        # First load
        data1 = await data_replay.load_historical_data(
            "BTCUSDT", start_date, end_date, "1h"
        )
        
        # Second load (should use cache)
        data2 = await data_replay.load_historical_data(
            "BTCUSDT", start_date, end_date, "1h"
        )
        
        # Should be the same data
        assert len(data1) == len(data2)
        assert data1[0].timestamp == data2[0].timestamp
        assert data1[-1].timestamp == data2[-1].timestamp
    
    def test_timeframe_parsing(self, data_replay):
        """Test timeframe string parsing."""
        assert data_replay._parse_timeframe("1m") == 1
        assert data_replay._parse_timeframe("5m") == 5
        assert data_replay._parse_timeframe("1h") == 60
        assert data_replay._parse_timeframe("1d") == 1440
        assert data_replay._parse_timeframe("invalid") == 60  # Default


class TestSimpleMovingAverageStrategy:
    """Test suite for SimpleMovingAverageStrategy."""
    
    @pytest.fixture
    def strategy(self) -> SimpleMovingAverageStrategy:
        """Create simple moving average strategy."""
        return SimpleMovingAverageStrategy(
            "test_strategy",
            {'short_window': 3, 'long_window': 5, 'position_size': 1.0}
        )
    
    @pytest.fixture
    def trending_data(self) -> List[MarketData]:
        """Generate trending market data for strategy testing."""
        data = []
        base_time = datetime(2023, 1, 1)
        
        # Create uptrend
        for i in range(20):
            price = 50000 + i * 100  # Steady uptrend
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 50,
                low=price - 50,
                close=price,
                volume=1000.0
            ))
        
        return data
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, strategy, trending_data):
        """Test moving average signal generation."""
        signals = await strategy.generate_signals(trending_data)
        
        # Should generate some signals
        assert len(signals) > 0
        
        # Verify signal structure
        for signal in signals:
            assert 'timestamp' in signal
            assert 'symbol' in signal
            assert 'signal' in signal
            assert signal['signal'] in ['BUY', 'SELL']
            assert 'strength' in signal
            assert 'price' in signal
            assert 'metadata' in signal
    
    @pytest.mark.asyncio
    async def test_order_generation(self, strategy, trending_data):
        """Test order generation from market data."""
        orders = []
        
        for data_point in trending_data:
            order = await strategy.on_market_data(data_point)
            if order:
                orders.append(order)
        
        # Should generate some orders
        assert len(orders) > 0
        
        # Verify order structure
        for order in orders:
            assert isinstance(order, Order)
            assert order.symbol == "BTCUSDT"
            assert order.side in [OrderSide.BUY, OrderSide.SELL]
            assert order.order_type == OrderType.MARKET
            assert order.quantity > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])