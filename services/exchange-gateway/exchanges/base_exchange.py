"""
Base exchange interface for the Advanced Trading Platform.
Defines the common interface that all exchange connectors must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class ExchangeInterface(ABC):
    """Abstract base class for exchange connectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.rate_limits = {}
        self.last_request_time = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the exchange connector."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the exchange connector and cleanup resources."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get exchange status and connectivity."""
        pass
    
    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including trading pairs, limits, etc."""
        pass
    
    # Market Data Methods
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.
        
        Returns:
            {
                'bid': float,
                'ask': float,
                'last': float,
                'volume': float,
                'timestamp': datetime
            }
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Returns:
            {
                'bids': [[price, amount], ...],
                'asks': [[price, amount], ...],
                'timestamp': datetime
            }
        """
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Returns:
            [
                {
                    'id': str,
                    'price': float,
                    'amount': float,
                    'side': str,
                    'timestamp': datetime
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Get candlestick/kline data for a symbol.
        
        Returns:
            [
                {
                    'timestamp': datetime,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': float
                },
                ...
            ]
        """
        pass
    
    # Account Methods
    @abstractmethod
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances.
        
        Returns:
            {
                'BTC': {'free': 1.0, 'locked': 0.1},
                'USDT': {'free': 1000.0, 'locked': 100.0},
                ...
            }
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information."""
        pass
    
    # Trading Methods
    @abstractmethod
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
        """
        Place an order.
        
        Returns:
            {
                'order_id': str,
                'symbol': str,
                'side': str,
                'type': str,
                'amount': float,
                'price': float,
                'status': str,
                'timestamp': datetime,
                'fees': {'currency': str, 'amount': float}
            }
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get order details."""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        pass
    
    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        pass
    
    # Utility Methods
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the exchange."""
        # Default implementation - override in specific connectors
        return symbol.upper().replace('/', '')
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert exchange symbol back to standard format."""
        # Default implementation - override in specific connectors
        return symbol
    
    def validate_order_params(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None
    ) -> None:
        """Validate order parameters."""
        if side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            raise ValueError(f"Invalid order side: {side}")
        
        if type not in [ot.value for ot in OrderType]:
            raise ValueError(f"Invalid order type: {type}")
        
        if amount <= 0:
            raise ValueError(f"Invalid amount: {amount}")
        
        if type in [OrderType.LIMIT.value, OrderType.STOP_LOSS_LIMIT.value, OrderType.TAKE_PROFIT_LIMIT.value]:
            if price is None or price <= 0:
                raise ValueError(f"Price required for {type} orders")
    
    async def check_rate_limit(self, endpoint: str) -> None:
        """Check and enforce rate limits."""
        # Basic rate limiting implementation
        # Override in specific connectors for exchange-specific limits
        current_time = datetime.utcnow()
        
        if endpoint in self.last_request_time:
            time_diff = (current_time - self.last_request_time[endpoint]).total_seconds()
            min_interval = self.rate_limits.get(endpoint, 0.1)  # Default 100ms
            
            if time_diff < min_interval:
                import asyncio
                await asyncio.sleep(min_interval - time_diff)
        
        self.last_request_time[endpoint] = current_time
    
    def format_price(self, price: float, symbol: str) -> float:
        """Format price according to exchange requirements."""
        # Default implementation - override in specific connectors
        return round(price, 8)
    
    def format_amount(self, amount: float, symbol: str) -> float:
        """Format amount according to exchange requirements."""
        # Default implementation - override in specific connectors
        return round(amount, 8)
    
    def calculate_fees(self, amount: float, price: float, side: str) -> Dict[str, float]:
        """Calculate trading fees."""
        # Default implementation - override in specific connectors
        fee_rate = 0.001  # 0.1% default fee
        fee_amount = amount * price * fee_rate
        
        return {
            'currency': 'USDT',  # Assume USDT fees
            'amount': fee_amount,
            'rate': fee_rate
        }
    
    def get_trading_rules(self, symbol: str) -> Dict[str, Any]:
        """Get trading rules for a symbol."""
        # Default implementation - override in specific connectors
        return {
            'min_order_size': 0.001,
            'max_order_size': 1000000,
            'min_price': 0.00000001,
            'max_price': 1000000,
            'price_precision': 8,
            'amount_precision': 8,
            'tick_size': 0.00000001,
            'step_size': 0.00000001
        }
    
    # Advanced Trading Methods (Optional - implement in specific connectors)
    async def place_futures_order(self, *args, **kwargs) -> Dict[str, Any]:
        """Place a futures order (if supported)."""
        raise NotImplementedError("Futures trading not supported on this exchange")
    
    async def get_futures_positions(self) -> List[Dict[str, Any]]:
        """Get futures positions (if supported)."""
        raise NotImplementedError("Futures trading not supported on this exchange")
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for futures trading (if supported)."""
        raise NotImplementedError("Leverage setting not supported on this exchange")
    
    async def place_margin_order(self, *args, **kwargs) -> Dict[str, Any]:
        """Place a margin order (if supported)."""
        raise NotImplementedError("Margin trading not supported on this exchange")
    
    async def get_margin_positions(self) -> List[Dict[str, Any]]:
        """Get margin positions (if supported)."""
        raise NotImplementedError("Margin trading not supported on this exchange")
    
    async def place_conditional_order(self, *args, **kwargs) -> Dict[str, Any]:
        """Place a conditional order (if supported)."""
        raise NotImplementedError("Conditional orders not supported on this exchange")
    
    async def place_advanced_order(self, *args, **kwargs) -> Dict[str, Any]:
        """Place an advanced order with institutional features (if supported)."""
        raise NotImplementedError("Advanced orders not supported on this exchange")
    
    async def get_fills(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get trade fills (if supported)."""
        raise NotImplementedError("Fill data not available on this exchange")
    
    async def get_funding_records(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get funding records (if supported)."""
        raise NotImplementedError("Funding records not available on this exchange")
    
    async def get_trade_history(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Get detailed trade history (if supported)."""
        raise NotImplementedError("Detailed trade history not available on this exchange")
    
    # Exchange Health and Monitoring
    async def get_exchange_health(self) -> Dict[str, Any]:
        """Get detailed exchange health information."""
        try:
            status = await self.get_status()
            
            # Test market data latency
            start_time = datetime.utcnow()
            try:
                await self.get_ticker("BTC/USDT")
                market_data_latency = (datetime.utcnow() - start_time).total_seconds()
                market_data_ok = True
            except:
                market_data_latency = None
                market_data_ok = False
            
            # Test account access
            try:
                await self.get_balances()
                account_access_ok = True
            except:
                account_access_ok = False
            
            return {
                "exchange": self.name,
                "status": status.get('status', 'unknown'),
                "market_data_ok": market_data_ok,
                "market_data_latency_ms": market_data_latency * 1000 if market_data_latency else None,
                "account_access_ok": account_access_ok,
                "initialized": self.is_initialized,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "exchange": self.name,
                "status": "error",
                "error": str(e),
                "initialized": self.is_initialized,
                "last_check": datetime.utcnow().isoformat()
            }
    
    # Utility Methods for Smart Routing
    def calculate_liquidity_score(self, order_book: Dict[str, Any], side: str, amount: float) -> float:
        """Calculate liquidity score for smart routing."""
        try:
            if side.lower() == 'buy':
                levels = order_book.get('asks', [])
            else:
                levels = order_book.get('bids', [])
            
            total_liquidity = sum(level[1] for level in levels[:10])  # Top 10 levels
            liquidity_ratio = total_liquidity / amount if amount > 0 else 0
            
            return min(liquidity_ratio, 10.0)  # Cap at 10x
        except:
            return 0.0
    
    def calculate_execution_cost(self, order_book: Dict[str, Any], side: str, amount: float) -> Dict[str, float]:
        """Calculate estimated execution cost and slippage."""
        try:
            if side.lower() == 'buy':
                levels = order_book.get('asks', [])
            else:
                levels = order_book.get('bids', [])
            
            remaining_amount = amount
            total_cost = 0.0
            weighted_price = 0.0
            
            for price, size in levels:
                if remaining_amount <= 0:
                    break
                
                fill_amount = min(remaining_amount, size)
                total_cost += fill_amount * price
                weighted_price += fill_amount * price
                remaining_amount -= fill_amount
            
            if amount > 0 and total_cost > 0:
                avg_price = total_cost / (amount - remaining_amount)
                best_price = levels[0][0] if levels else 0
                slippage = abs(avg_price - best_price) / best_price if best_price > 0 else 0
                
                return {
                    'average_price': avg_price,
                    'best_price': best_price,
                    'slippage_pct': slippage * 100,
                    'fillable_amount': amount - remaining_amount,
                    'total_cost': total_cost
                }
            
            return {
                'average_price': 0,
                'best_price': 0,
                'slippage_pct': 100,  # Cannot fill
                'fillable_amount': 0,
                'total_cost': 0
            }
        except:
            return {
                'average_price': 0,
                'best_price': 0,
                'slippage_pct': 100,
                'fillable_amount': 0,
                'total_cost': 0
            }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ExchangeError(Exception):
    """Base exception for exchange-related errors."""
    
    def __init__(self, message: str, exchange: str, error_code: Optional[str] = None):
        self.message = message
        self.exchange = exchange
        self.error_code = error_code
        super().__init__(f"{exchange}: {message}")


class RateLimitError(ExchangeError):
    """Exception raised when rate limit is exceeded."""
    pass


class InsufficientBalanceError(ExchangeError):
    """Exception raised when account has insufficient balance."""
    pass


class InvalidOrderError(ExchangeError):
    """Exception raised when order parameters are invalid."""
    pass


class OrderNotFoundError(ExchangeError):
    """Exception raised when order is not found."""
    pass


class ExchangeConnectionError(ExchangeError):
    """Exception raised when connection to exchange fails."""
    pass