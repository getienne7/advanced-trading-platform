"""Simple demo of arbitrage detection concepts."""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class MockTicker:
    """Mock ticker for demo."""
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    close_price: Decimal
    timestamp: datetime


@dataclass
class MockOrderBook:
    """Mock order book for demo."""
    symbol: str
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    timestamp: datetime


@dataclass
class ArbitrageOpportunity:
    """Simple arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    profit_pct: Decimal
    profit_absolute: Decimal
    timestamp: datetime


class SimpleArbitrageDetector:
    """Simple arbitrage detector for demo."""
    
    def __init__(self):
        self.opportunities: List[ArbitrageOpportunity] = []
        self.min_profit_pct = Decimal('0.5')
    
    def detect_simple_arbitrage(self, market_data: Dict[str, Dict[str, MockTicker]]) -> List[ArbitrageOpportunity]:
        """Detect simple arbitrage opportunities."""
        opportunities = []
        
        # Get all symbols
        all_symbols = set()
        for exchange_data in market_data.values():
            all_symbols.update(exchange_data.keys())
        
        for symbol in all_symbols:
            exchange_tickers = []
            
            # Collect tickers from all exchanges for this symbol
            for exchange, tickers in market_data.items():
                if symbol in tickers:
                    exchange_tickers.append((exchange, tickers[symbol]))
            
            if len(exchange_tickers) < 2:
                continue
            
            # Find best bid and ask
            best_bid_exchange, best_bid_ticker = max(exchange_tickers, key=lambda x: x[1].bid_price)
            best_ask_exchange, best_ask_ticker = min(exchange_tickers, key=lambda x: x[1].ask_price)
            
            if best_bid_exchange == best_ask_exchange:
                continue  # Same exchange
            
            # Calculate profit
            buy_price = best_ask_ticker.ask_price
            sell_price = best_bid_ticker.bid_price
            profit_absolute = sell_price - buy_price
            profit_pct = (profit_absolute / buy_price) * 100
            
            if profit_pct >= self.min_profit_pct:
                opportunity = ArbitrageOpportunity(
                    symbol=symbol,
                    buy_exchange=best_ask_exchange,
                    sell_exchange=best_bid_exchange,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    profit_pct=profit_pct,
                    profit_absolute=profit_absolute,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def detect_triangular_arbitrage(self, exchange: str, market_data: Dict[str, MockTicker]) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage opportunities."""
        opportunities = []
        
        # Check if we have the required pairs for BTC/ETH/USDT triangular arbitrage
        required_pairs = ['BTCUSDT', 'ETHUSDT', 'ETHBTC']
        if not all(pair in market_data for pair in required_pairs):
            return opportunities
        
        # Get tickers
        btc_usdt = market_data['BTCUSDT']
        eth_usdt = market_data['ETHUSDT']
        eth_btc = market_data['ETHBTC']
        
        # Calculate triangular arbitrage: USDT -> BTC -> ETH -> USDT
        try:
            # Start with 1000 USDT
            initial_amount = Decimal('1000')
            
            # USDT -> BTC (buy BTC with USDT)
            btc_amount = initial_amount / btc_usdt.ask_price
            
            # BTC -> ETH (buy ETH with BTC)
            eth_amount = btc_amount / eth_btc.ask_price
            
            # ETH -> USDT (sell ETH for USDT)
            final_usdt = eth_amount * eth_usdt.bid_price
            
            # Calculate profit
            profit_absolute = final_usdt - initial_amount
            profit_pct = (profit_absolute / initial_amount) * 100
            
            if profit_pct >= self.min_profit_pct:
                opportunity = ArbitrageOpportunity(
                    symbol='TRIANGULAR_BTC_ETH_USDT',
                    buy_exchange=exchange,
                    sell_exchange=exchange,
                    buy_price=initial_amount,
                    sell_price=final_usdt,
                    profit_pct=profit_pct,
                    profit_absolute=profit_absolute,
                    timestamp=datetime.now()
                )
                opportunities.append(opportunity)
        
        except Exception as e:
            print(f"Error in triangular arbitrage calculation: {e}")
        
        return opportunities


def create_mock_market_data() -> Dict[str, Dict[str, MockTicker]]:
    """Create mock market data for demo."""
    now = datetime.now()
    
    return {
        'binance': {
            'BTCUSDT': MockTicker('BTCUSDT', Decimal('49950'), Decimal('50050'), Decimal('50000'), now),
            'ETHUSDT': MockTicker('ETHUSDT', Decimal('2995'), Decimal('3005'), Decimal('3000'), now),
            'ETHBTC': MockTicker('ETHBTC', Decimal('0.05995'), Decimal('0.06005'), Decimal('0.06000'), now),
        },
        'coinbase': {
            'BTCUSDT': MockTicker('BTCUSDT', Decimal('50250'), Decimal('50350'), Decimal('50300'), now),
            'ETHUSDT': MockTicker('ETHUSDT', Decimal('3020'), Decimal('3030'), Decimal('3025'), now),
        },
        'kraken': {
            'BTCUSDT': MockTicker('BTCUSDT', Decimal('50100'), Decimal('50200'), Decimal('50150'), now),
            'ETHUSDT': MockTicker('ETHUSDT', Decimal('2980'), Decimal('2990'), Decimal('2985'), now),
        }
    }


async def main():
    """Run arbitrage detection demo."""
    print("=== Arbitrage Detection Engine Demo ===\\n")
    
    # Create detector
    detector = SimpleArbitrageDetector()
    
    # Create mock market data
    market_data = create_mock_market_data()
    
    print("Market Data:")
    for exchange, tickers in market_data.items():
        print(f"\\n{exchange.upper()}:")
        for symbol, ticker in tickers.items():
            print(f"  {symbol}: Bid={ticker.bid_price}, Ask={ticker.ask_price}")
    
    print("\\n" + "="*50)
    
    # Detect simple arbitrage
    simple_opportunities = detector.detect_simple_arbitrage(market_data)
    
    print(f"\\nSimple Arbitrage Opportunities Found: {len(simple_opportunities)}")
    for opp in simple_opportunities:
        print(f"\\n{opp.symbol}:")
        print(f"  Buy on {opp.buy_exchange} at ${opp.buy_price}")
        print(f"  Sell on {opp.sell_exchange} at ${opp.sell_price}")
        print(f"  Profit: ${opp.profit_absolute} ({opp.profit_pct:.2f}%)")
    
    # Detect triangular arbitrage on Binance
    triangular_opportunities = detector.detect_triangular_arbitrage('binance', market_data['binance'])
    
    print(f"\\nTriangular Arbitrage Opportunities Found: {len(triangular_opportunities)}")
    for opp in triangular_opportunities:
        print(f"\\n{opp.symbol}:")
        print(f"  Exchange: {opp.buy_exchange}")
        print(f"  Initial: ${opp.buy_price}")
        print(f"  Final: ${opp.sell_price}")
        print(f"  Profit: ${opp.profit_absolute} ({opp.profit_pct:.2f}%)")
    
    # Simulate price changes and detect new opportunities
    print("\\n" + "="*50)
    print("\\nSimulating price changes...")
    
    # Modify prices to create more arbitrage opportunities
    market_data['coinbase']['BTCUSDT'].bid_price = Decimal('50500')  # Higher bid on Coinbase
    market_data['kraken']['ETHUSDT'].ask_price = Decimal('2950')     # Lower ask on Kraken
    
    print("\\nUpdated Market Data:")
    print("COINBASE BTCUSDT: Bid=50500 (increased)")
    print("KRAKEN ETHUSDT: Ask=2950 (decreased)")
    
    # Detect opportunities again
    new_opportunities = detector.detect_simple_arbitrage(market_data)
    
    print(f"\\nNew Simple Arbitrage Opportunities: {len(new_opportunities)}")
    for opp in new_opportunities:
        print(f"\\n{opp.symbol}:")
        print(f"  Buy on {opp.buy_exchange} at ${opp.buy_price}")
        print(f"  Sell on {opp.sell_exchange} at ${opp.sell_price}")
        print(f"  Profit: ${opp.profit_absolute} ({opp.profit_pct:.2f}%)")
    
    print("\\n" + "="*50)
    print("\\nDemo completed! This shows how arbitrage opportunities can be detected")
    print("across multiple exchanges by comparing bid/ask prices.")


if __name__ == "__main__":
    asyncio.run(main())