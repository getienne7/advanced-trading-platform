"""
Demo exchange connector for testing and development.
Simulates exchange behavior without requiring real API keys.
"""
import asyncio
import random
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog

from .base_exchange import ExchangeInterface, ExchangeError

logger = structlog.get_logger(__name__)


class DemoExchange(ExchangeInterface):
    """Demo exchange connector for testing."""
    
    def __init__(self):
        super().__init__("demo")
        
        # Simulated account data
        self.balances = {
            'BTC': {'free': 1.5, 'locked': 0.1},
            'ETH': {'free': 10.0, 'locked': 0.5},
            'USDT': {'free': 50000.0, 'locked': 5000.0},
            'ADA': {'free': 1000.0, 'locked': 0.0},
            'DOT': {'free': 100.0, 'locked': 0.0}
        }
        
        # Simulated orders
        self.orders = {}
        self.order_history = []
        
        # Simulated market data
        self.market_data = {
            'BTC/USDT': {'last': 66500.0, 'bid': 66450.0, 'ask': 66550.0, 'volume': 1234.56},
            'ETH/USDT': {'last': 3990.0, 'bid': 3985.0, 'ask': 3995.0, 'volume': 5678.90},
            'ADA/USDT': {'last': 0.45, 'bid': 0.449, 'ask': 0.451, 'volume': 12345.67},
            'DOT/USDT': {'last': 6.75, 'bid': 6.74, 'ask': 6.76, 'volume': 2345.67},
            'LINK/USDT': {'last': 12.50, 'bid': 12.48, 'ask': 12.52, 'volume': 3456.78}
        }
    
    async def initialize(self) -> None:
        """Initialize the demo exchange."""
        self.is_initialized = True
        logger.info("Demo exchange initialized")
    
    async def close(self) -> None:
        """Close the demo exchange."""
        self.is_initialized = False
        logger.info("Demo exchange closed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get demo exchange status."""
        return {
            'status': 'online',
            'mode': 'demo',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get demo exchange information."""
        return {
            'name': 'Demo Exchange',
            'status': 'online',
            'mode': 'simulation',
            'symbols': list(self.market_data.keys()),
            'supported_order_types': ['market', 'limit'],
            'maker_fee': 0.001,  # 0.1%
            'taker_fee': 0.001   # 0.1%
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol."""
        if symbol not in self.market_data:
            raise ExchangeError(f"Symbol {symbol} not supported", "demo")
        
        # Add some random variation to simulate real market movement
        base_data = self.market_data[symbol]
        variation = random.uniform(-0.002, 0.002)  # ±0.2% variation
        
        last = base_data['last'] * (1 + variation)
        spread = base_data['ask'] - base_data['bid']
        
        return {
            'bid': last - spread/2,
            'ask': last + spread/2,
            'last': last,
            'volume': base_data['volume'] * random.uniform(0.8, 1.2),
            'timestamp': datetime.utcnow()
        }
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book for a symbol."""
        if symbol not in self.market_data:
            raise ExchangeError(f"Symbol {symbol} not supported", "demo")
        
        ticker = await self.get_ticker(symbol)
        
        # Generate realistic order book
        bids = []
        asks = []
        
        bid_price = ticker['bid']
        ask_price = ticker['ask']
        
        # Generate bids (decreasing prices)
        for i in range(min(limit, 50)):
            price = bid_price * (1 - i * 0.0001)  # 0.01% steps
            amount = random.uniform(0.1, 10.0)
            bids.append([price, amount])
        
        # Generate asks (increasing prices)
        for i in range(min(limit, 50)):
            price = ask_price * (1 + i * 0.0001)  # 0.01% steps
            amount = random.uniform(0.1, 10.0)
            asks.append([price, amount])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.utcnow()
        }
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        if symbol not in self.market_data:
            raise ExchangeError(f"Symbol {symbol} not supported", "demo")
        
        ticker = await self.get_ticker(symbol)
        
        trades = []
        base_time = datetime.utcnow()
        
        for i in range(min(limit, 50)):
            # Generate random trades around current price
            price_variation = random.uniform(-0.001, 0.001)  # ±0.1%
            price = ticker['last'] * (1 + price_variation)
            amount = random.uniform(0.01, 5.0)
            side = random.choice(['buy', 'sell'])
            
            trades.append({
                'id': str(uuid.uuid4()),
                'price': price,
                'amount': amount,
                'side': side,
                'timestamp': base_time - timedelta(seconds=i * 10)
            })
        
        return trades
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get candlestick/kline data for a symbol."""
        if symbol not in self.market_data:
            raise ExchangeError(f"Symbol {symbol} not supported", "demo")
        
        # Generate realistic OHLCV data
        base_price = self.market_data[symbol]['last']
        
        # Convert interval to minutes
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(interval, 60)
        
        klines = []
        current_time = end_time or datetime.utcnow()
        current_price = base_price
        
        for i in range(min(limit, 100)):
            # Generate OHLC with realistic movement
            open_price = current_price
            
            # Random price movement
            change = random.uniform(-0.02, 0.02)  # ±2% per candle
            close_price = open_price * (1 + change)
            
            # Generate high and low
            high_change = random.uniform(0, 0.01)  # Up to 1% higher
            low_change = random.uniform(0, 0.01)   # Up to 1% lower
            
            high = max(open_price, close_price) * (1 + high_change)
            low = min(open_price, close_price) * (1 - low_change)
            
            volume = random.uniform(10, 1000)
            
            klines.append({
                'timestamp': current_time - timedelta(minutes=i * interval_minutes),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            
            current_price = close_price
        
        # Sort by timestamp (oldest first)
        klines.sort(key=lambda x: x['timestamp'])
        
        return klines
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """Get account balances."""
        return self.balances.copy()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return {
            'maker_commission': 0.001,
            'taker_commission': 0.001,
            'can_trade': True,
            'can_withdraw': True,
            'can_deposit': True,
            'account_type': 'DEMO',
            'total_balance_usd': 100000.0
        }
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        **kwargs
    ) -> Dict[str, Any]:
        """Place a simulated order."""
        self.validate_order_params(symbol, side, type, amount, price)
        
        if symbol not in self.market_data:
            raise ExchangeError(f"Symbol {symbol} not supported", "demo")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Get current market price for market orders
        if type.lower() == 'market':
            ticker = await self.get_ticker(symbol)
            price = ticker['ask'] if side.lower() == 'buy' else ticker['bid']
        
        # Simulate order execution (90% fill rate)
        status = 'filled' if random.random() < 0.9 else 'new'
        filled_amount = amount if status == 'filled' else 0
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side.lower(),
            'type': type.lower(),
            'amount': amount,
            'filled': filled_amount,
            'price': price or 0,
            'average_price': price or 0,
            'status': status,
            'timestamp': datetime.utcnow(),
            'fees': {
                'currency': 'USDT',
                'amount': amount * (price or 0) * 0.001  # 0.1% fee
            }
        }
        
        # Store order
        self.orders[order_id] = order
        self.order_history.append(order.copy())
        
        # Update balances if filled
        if status == 'filled':
            await self._update_balances_after_trade(symbol, side, amount, price)
        
        logger.info("Demo order placed", order_id=order_id, symbol=symbol, side=side, status=status)
        
        return order
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a simulated order."""
        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found", "demo")
        
        order = self.orders[order_id]
        order['status'] = 'canceled'
        
        return {
            'order_id': order_id,
            'symbol': order['symbol'],
            'status': 'canceled',
            'timestamp': datetime.utcnow()
        }
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get order details."""
        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found", "demo")
        
        return self.orders[order_id].copy()
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        open_orders = []
        
        for order in self.orders.values():
            if order['status'] in ['new', 'partially_filled']:
                if symbol is None or order['symbol'] == symbol:
                    open_orders.append(order.copy())
        
        return open_orders
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        filtered_orders = []
        
        for order in self.order_history:
            # Filter by symbol
            if symbol and order['symbol'] != symbol:
                continue
            
            # Filter by time
            if start_time and order['timestamp'] < start_time:
                continue
            if end_time and order['timestamp'] > end_time:
                continue
            
            filtered_orders.append(order.copy())
        
        # Sort by timestamp (newest first) and limit
        filtered_orders.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered_orders[:limit]
    
    async def _update_balances_after_trade(self, symbol: str, side: str, amount: float, price: float):
        """Update balances after a simulated trade."""
        base_currency = symbol.split('/')[0]
        quote_currency = symbol.split('/')[1]
        
        if side.lower() == 'buy':
            # Buying: decrease quote currency, increase base currency
            if quote_currency in self.balances:
                cost = amount * price * 1.001  # Include 0.1% fee
                self.balances[quote_currency]['free'] -= cost
                self.balances[quote_currency]['free'] = max(0, self.balances[quote_currency]['free'])
            
            if base_currency not in self.balances:
                self.balances[base_currency] = {'free': 0, 'locked': 0}
            self.balances[base_currency]['free'] += amount
            
        else:  # sell
            # Selling: decrease base currency, increase quote currency
            if base_currency in self.balances:
                self.balances[base_currency]['free'] -= amount
                self.balances[base_currency]['free'] = max(0, self.balances[base_currency]['free'])
            
            if quote_currency not in self.balances:
                self.balances[quote_currency] = {'free': 0, 'locked': 0}
            
            proceeds = amount * price * 0.999  # Subtract 0.1% fee
            self.balances[quote_currency]['free'] += proceeds