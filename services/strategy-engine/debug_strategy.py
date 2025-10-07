"""
Debug the strategy to see why it's not generating trades.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from backtesting_engine import SimpleMovingAverageStrategy, MarketData

async def debug_strategy():
    """Debug strategy signal generation."""
    print("Debugging strategy...")
    
    # Create simple trending data
    market_data = []
    base_time = datetime(2023, 1, 1)
    
    # Create clear uptrend
    prices = [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900,
              51000, 51100, 51200, 51300, 51400, 51500, 51600, 51700, 51800, 51900]
    
    for i, price in enumerate(prices):
        market_data.append(MarketData(
            timestamp=base_time + timedelta(hours=i),
            symbol="BTCUSDT",
            open=price,
            high=price + 50,
            low=price - 50,
            close=price,
            volume=1000.0
        ))
    
    # Create strategy with smaller windows
    strategy = SimpleMovingAverageStrategy(
        "debug_strategy",
        {'short_window': 3, 'long_window': 6, 'position_size': 1.0}
    )
    
    # Test signal generation
    signals = await strategy.generate_signals(market_data)
    print(f"Generated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal['timestamp']}: {signal['signal']} at {signal['price']}")
    
    # Test order generation
    print("\nTesting order generation:")
    for i, data_point in enumerate(market_data):
        order = await strategy.on_market_data(data_point)
        if order:
            print(f"  Hour {i}: {order.side.value} order for {order.quantity} at {data_point.close}")
        
        # Print moving averages for debugging
        if len(strategy.price_history) >= strategy.long_window:
            short_ma = sum(strategy.price_history[-strategy.short_window:]) / strategy.short_window
            long_ma = sum(strategy.price_history[-strategy.long_window:]) / strategy.long_window
            print(f"    Hour {i}: Price={data_point.close}, Short MA={short_ma:.2f}, Long MA={long_ma:.2f}")

if __name__ == "__main__":
    asyncio.run(debug_strategy())