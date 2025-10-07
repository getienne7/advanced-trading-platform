"""
Simple test to verify backtesting engine functionality.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from backtesting_engine import (
    BacktestingEngine, BacktestingConfig, SimpleMovingAverageStrategy,
    MarketData, ExecutionModel
)

async def test_simple_backtest():
    """Test a simple backtest execution."""
    print("Starting simple backtest test...")
    
    # Create sample market data
    market_data = []
    base_time = datetime(2023, 1, 1)
    base_price = 50000.0
    
    for i in range(100):
        # Create trending price data with some volatility
        trend = i * 50  # Uptrend
        volatility = 200 * (0.5 - abs((i % 20) - 10) / 10)  # Oscillating volatility
        price = base_price + trend + volatility
        market_data.append(MarketData(
            timestamp=base_time + timedelta(hours=i),
            symbol="BTCUSDT",
            open=price,
            high=price + 50,
            low=price - 50,
            close=price,
            volume=1000.0,
            bid=price * 0.999,
            ask=price * 1.001,
            spread=price * 0.002
        ))
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy(
        "test_ma_strategy",
        {'short_window': 5, 'long_window': 10, 'position_size': 1.0}
    )
    
    # Create backtesting config
    config = BacktestingConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        execution_model=ExecutionModel.REALISTIC
    )
    
    # Run backtest
    engine = BacktestingEngine(config)
    result = await engine.run_backtest(strategy, market_data)
    
    # Print results
    print(f"Strategy: {result['strategy_name']}")
    print(f"Total Return: {result['performance_metrics'].total_return:.4f}")
    print(f"Sharpe Ratio: {result['performance_metrics'].sharpe_ratio:.4f}")
    print(f"Max Drawdown: {result['performance_metrics'].max_drawdown:.4f}")
    print(f"Win Rate: {result['performance_metrics'].win_rate:.4f}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Final Capital: ${result['final_capital']:.2f}")
    
    print("Simple backtest test completed successfully!")
    return result

if __name__ == "__main__":
    asyncio.run(test_simple_backtest())