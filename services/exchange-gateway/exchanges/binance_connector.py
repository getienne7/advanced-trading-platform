"""
Binance exchange connector for the Advanced Trading Platform.
Implements the ExchangeInterface for Binance API integration.
"""
import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import structlog

from .base_exchange import (
    ExchangeInterface, ExchangeError, RateLimitError, 
    InsufficientBalanceError, InvalidOrderError, OrderNotFoundError
)

logger = structlog.get_logger(__name__)


class BinanceConnector(ExchangeInterface):
    """Binance exchange connector."""
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        super().__init__("binance")
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
        
        self.session = None
        
        # Rate limits (requests per minute)
        self.rate_limits = {
            'order': 10,      # 10 orders per second
            'general': 1200,  # 1200 requests per minute
            'market_data': 6000  # 6000 requests per minute
        }
        
        # Symbol mapping
        self.symbol_map = {}
        self.reverse_symbol_map = {}
    
    async def initialize(self) -> None:
        """Initialize the Binance connector."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'X-MBX-APIKEY': self.api_key,
                    'Content-Type': 'application/json'
                }
            )
            
            # Test connectivity
            await self._test_connectivity()
            
            # Load exchange info
            exchange_info = await self.get_exchange_info()
            self._build_symbol_maps(exchange_info)
            
            self.is_initialized = True
            logger.info("Binance connector initialized", testnet=self.testnet)
            
        except Exception as e:
            logger.error("Failed to initialize Binance connector", error=str(e))
            raise ExchangeError(f"Initialization failed: {e}", "binance")
    
    async def close(self) -> None:
        """Close the connector and cleanup resources."""
        if self.session:
            await self.session.close()
        self.is_initialized = False
        logger.info("Binance connector closed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Binance status and connectivity."""
        try:
            response = await self._make_request('GET', '/api/v3/ping')
            
            return {
                'status': 'online',
                'latency_ms': response.get('latency_ms', 0),
                'testnet': self.testnet,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'offline',
                'error': str(e),
                'testnet': self.testnet,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get Binance exchange information."""
        try:
            response = await self._make_request('GET', '/api/v3/exchangeInfo')
            
            return {
                'name': 'Binance',
                'status': response.get('status', 'UNKNOWN'),
                'timezone': response.get('timezone', 'UTC'),
                'server_time': response.get('serverTime', 0),
                'symbols': len(response.get('symbols', [])),
                'rate_limits': response.get('rateLimits', []),
                'exchange_filters': response.get('exchangeFilters', []),
                'testnet': self.testnet
            }
        except Exception as e:
            logger.error("Failed to get exchange info", error=str(e))
            raise ExchangeError(f"Failed to get exchange info: {e}", "binance")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol."""
        try:
            binance_symbol = self.normalize_symbol(symbol)
            
            response = await self._make_request('GET', '/api/v3/ticker/bookTicker', {
                'symbol': binance_symbol
            })
            
            return {
                'bid': float(response['bidPrice']),
                'ask': float(response['askPrice']),
                'last': float(response.get('price', response['askPrice'])),
                'volume': 0.0,  # Not available in bookTicker
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to get ticker", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get ticker: {e}", "binance")
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book for a symbol."""
        try:
            binance_symbol = self.normalize_symbol(symbol)
            
            # Binance supports limits: 5, 10, 20, 50, 100, 500, 1000, 5000
            valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
            limit = min(valid_limits, key=lambda x: abs(x - limit))
            
            response = await self._make_request('GET', '/api/v3/depth', {
                'symbol': binance_symbol,
                'limit': limit
            })
            
            return {
                'bids': [[float(price), float(amount)] for price, amount in response['bids']],
                'asks': [[float(price), float(amount)] for price, amount in response['asks']],
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to get order book", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get order book: {e}", "binance")
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        try:
            binance_symbol = self.normalize_symbol(symbol)
            
            response = await self._make_request('GET', '/api/v3/trades', {
                'symbol': binance_symbol,
                'limit': min(limit, 1000)
            })
            
            trades = []
            for trade in response:
                trades.append({
                    'id': str(trade['id']),
                    'price': float(trade['price']),
                    'amount': float(trade['qty']),
                    'side': 'buy' if trade['isBuyerMaker'] else 'sell',
                    'timestamp': datetime.fromtimestamp(trade['time'] / 1000)
                })
            
            return trades
        except Exception as e:
            logger.error("Failed to get trades", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get trades: {e}", "binance")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get candlestick/kline data for a symbol."""
        try:
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }
            
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = await self._make_request('GET', '/api/v3/klines', params)
            
            klines = []
            for kline in response:
                klines.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            return klines
        except Exception as e:
            logger.error("Failed to get klines", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get klines: {e}", "binance")
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """Get account balances."""
        try:
            response = await self._make_signed_request('GET', '/api/v3/account')
            
            balances = {}
            for balance in response['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked
                    }
            
            return balances
        except Exception as e:
            logger.error("Failed to get balances", error=str(e))
            raise ExchangeError(f"Failed to get balances: {e}", "binance")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information."""
        try:
            response = await self._make_signed_request('GET', '/api/v3/account')
            
            return {
                'maker_commission': response.get('makerCommission', 0) / 10000,  # Convert to percentage
                'taker_commission': response.get('takerCommission', 0) / 10000,
                'buyer_commission': response.get('buyerCommission', 0) / 10000,
                'seller_commission': response.get('sellerCommission', 0) / 10000,
                'can_trade': response.get('canTrade', False),
                'can_withdraw': response.get('canWithdraw', False),
                'can_deposit': response.get('canDeposit', False),
                'account_type': response.get('accountType', 'UNKNOWN'),
                'update_time': datetime.fromtimestamp(response.get('updateTime', 0) / 1000)
            }
        except Exception as e:
            logger.error("Failed to get account info", error=str(e))
            raise ExchangeError(f"Failed to get account info: {e}", "binance")
    
    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Place a futures order on Binance."""
        try:
            self.validate_order_params(symbol, side, type, amount, price)
            
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'type': type.upper(),
                'quantity': self.format_amount(amount, symbol),
                'timeInForce': time_in_force
            }
            
            if type.upper() in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                params['price'] = self.format_price(price, symbol)
            
            if type.upper() in ['STOP_LOSS', 'STOP_LOSS_LIMIT']:
                params['stopPrice'] = kwargs.get('stop_price', price)
            
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            # Use futures endpoint
            response = await self._make_signed_request('POST', '/fapi/v1/order', params)
            
            return {
                'order_id': str(response['orderId']),
                'symbol': symbol,
                'side': side.lower(),
                'type': type.lower(),
                'amount': float(response['origQty']),
                'price': float(response.get('price', 0)),
                'status': response['status'].lower(),
                'timestamp': datetime.fromtimestamp(response['updateTime'] / 1000),
                'fees': self._extract_fees(response)
            }
        except Exception as e:
            logger.error("Failed to place futures order", symbol=symbol, side=side, error=str(e))
            if "insufficient balance" in str(e).lower():
                raise InsufficientBalanceError(str(e), "binance")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(str(e), "binance")
            else:
                raise ExchangeError(f"Failed to place futures order: {e}", "binance")

    async def get_futures_positions(self) -> List[Dict[str, Any]]:
        """Get futures positions."""
        try:
            response = await self._make_signed_request('GET', '/fapi/v2/positionRisk')
            
            positions = []
            for position in response:
                if float(position['positionAmt']) != 0:
                    positions.append({
                        'symbol': self.denormalize_symbol(position['symbol']),
                        'size': float(position['positionAmt']),
                        'side': 'long' if float(position['positionAmt']) > 0 else 'short',
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'percentage': float(position['percentage']),
                        'margin_type': position['marginType'],
                        'isolated_margin': float(position['isolatedMargin']),
                        'leverage': int(position['leverage'])
                    })
            
            return positions
        except Exception as e:
            logger.error("Failed to get futures positions", error=str(e))
            raise ExchangeError(f"Failed to get futures positions: {e}", "binance")

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a futures symbol."""
        try:
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'leverage': leverage
            }
            
            response = await self._make_signed_request('POST', '/fapi/v1/leverage', params)
            
            return {
                'symbol': symbol,
                'leverage': response['leverage'],
                'max_notional_value': response['maxNotionalValue']
            }
        except Exception as e:
            logger.error("Failed to set leverage", symbol=symbol, leverage=leverage, error=str(e))
            raise ExchangeError(f"Failed to set leverage: {e}", "binance")

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
        """Place an order."""
        try:
            self.validate_order_params(symbol, side, type, amount, price)
            
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'type': type.upper(),
                'quantity': self.format_amount(amount, symbol),
                'timeInForce': time_in_force
            }
            
            if type.upper() in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                params['price'] = self.format_price(price, symbol)
            
            if type.upper() in ['STOP_LOSS', 'STOP_LOSS_LIMIT']:
                params['stopPrice'] = kwargs.get('stop_price', price)
            
            response = await self._make_signed_request('POST', '/api/v3/order', params)
            
            return {
                'order_id': str(response['orderId']),
                'symbol': symbol,
                'side': side.lower(),
                'type': type.lower(),
                'amount': float(response['origQty']),
                'price': float(response.get('price', 0)),
                'status': response['status'].lower(),
                'timestamp': datetime.fromtimestamp(response['transactTime'] / 1000),
                'fees': self._extract_fees(response)
            }
        except Exception as e:
            logger.error("Failed to place order", symbol=symbol, side=side, error=str(e))
            if "insufficient balance" in str(e).lower():
                raise InsufficientBalanceError(str(e), "binance")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(str(e), "binance")
            else:
                raise ExchangeError(f"Failed to place order: {e}", "binance")
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            if not symbol:
                raise ValueError("Symbol is required for Binance order cancellation")
            
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'orderId': order_id
            }
            
            response = await self._make_signed_request('DELETE', '/api/v3/order', params)
            
            return {
                'order_id': str(response['orderId']),
                'symbol': symbol,
                'status': 'canceled',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            if "not found" in str(e).lower():
                raise OrderNotFoundError(str(e), "binance")
            else:
                raise ExchangeError(f"Failed to cancel order: {e}", "binance")
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get order details."""
        try:
            if not symbol:
                raise ValueError("Symbol is required for Binance order lookup")
            
            binance_symbol = self.normalize_symbol(symbol)
            
            params = {
                'symbol': binance_symbol,
                'orderId': order_id
            }
            
            response = await self._make_signed_request('GET', '/api/v3/order', params)
            
            return {
                'order_id': str(response['orderId']),
                'symbol': symbol,
                'side': response['side'].lower(),
                'type': response['type'].lower(),
                'amount': float(response['origQty']),
                'filled': float(response['executedQty']),
                'price': float(response.get('price', 0)),
                'average_price': float(response.get('avgPrice', 0)),
                'status': response['status'].lower(),
                'timestamp': datetime.fromtimestamp(response['time'] / 1000),
                'update_time': datetime.fromtimestamp(response['updateTime'] / 1000)
            }
        except Exception as e:
            logger.error("Failed to get order", order_id=order_id, error=str(e))
            if "not found" in str(e).lower():
                raise OrderNotFoundError(str(e), "binance")
            else:
                raise ExchangeError(f"Failed to get order: {e}", "binance")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        try:
            params = {}
            if symbol:
                params['symbol'] = self.normalize_symbol(symbol)
            
            response = await self._make_signed_request('GET', '/api/v3/openOrders', params)
            
            orders = []
            for order in response:
                orders.append({
                    'order_id': str(order['orderId']),
                    'symbol': self.denormalize_symbol(order['symbol']),
                    'side': order['side'].lower(),
                    'type': order['type'].lower(),
                    'amount': float(order['origQty']),
                    'filled': float(order['executedQty']),
                    'price': float(order.get('price', 0)),
                    'status': order['status'].lower(),
                    'timestamp': datetime.fromtimestamp(order['time'] / 1000)
                })
            
            return orders
        except Exception as e:
            logger.error("Failed to get open orders", error=str(e))
            raise ExchangeError(f"Failed to get open orders: {e}", "binance")
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        try:
            params = {
                'limit': min(limit, 1000)
            }
            
            if symbol:
                params['symbol'] = self.normalize_symbol(symbol)
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = await self._make_signed_request('GET', '/api/v3/allOrders', params)
            
            orders = []
            for order in response:
                orders.append({
                    'order_id': str(order['orderId']),
                    'symbol': self.denormalize_symbol(order['symbol']),
                    'side': order['side'].lower(),
                    'type': order['type'].lower(),
                    'amount': float(order['origQty']),
                    'filled': float(order['executedQty']),
                    'price': float(order.get('price', 0)),
                    'average_price': float(order.get('avgPrice', 0)),
                    'status': order['status'].lower(),
                    'timestamp': datetime.fromtimestamp(order['time'] / 1000),
                    'update_time': datetime.fromtimestamp(order['updateTime'] / 1000)
                })
            
            return orders
        except Exception as e:
            logger.error("Failed to get order history", error=str(e))
            raise ExchangeError(f"Failed to get order history: {e}", "binance")
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Binance."""
        # Convert BTC/USDT to BTCUSDT
        return symbol.upper().replace('/', '')
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert Binance symbol back to standard format."""
        # This is a simplified implementation
        # In production, you'd use the symbol mapping from exchange info
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        elif symbol.endswith('BTC'):
            base = symbol[:-3]
            return f"{base}/BTC"
        elif symbol.endswith('ETH'):
            base = symbol[:-3]
            return f"{base}/ETH"
        else:
            return symbol
    
    async def _test_connectivity(self) -> None:
        """Test connectivity to Binance."""
        try:
            await self._make_request('GET', '/api/v3/ping')
            logger.info("Binance connectivity test successful")
        except Exception as e:
            logger.error("Binance connectivity test failed", error=str(e))
            raise
    
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to Binance API."""
        if not self.session:
            raise ExchangeError("Session not initialized", "binance")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method == 'POST':
                async with self.session.post(url, data=params) as response:
                    return await self._handle_response(response)
            elif method == 'DELETE':
                async with self.session.delete(url, data=params) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            raise ExchangeError(f"Request failed: {e}", "binance")
    
    async def _make_signed_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a signed request to Binance API."""
        if params is None:
            params = {}
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Create signature
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        return await self._make_request(method, endpoint, params)
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle API response."""
        text = await response.text()
        
        if response.status == 200:
            return json.loads(text)
        elif response.status == 429:
            raise RateLimitError("Rate limit exceeded", "binance")
        else:
            try:
                error_data = json.loads(text)
                error_msg = error_data.get('msg', text)
                error_code = error_data.get('code', response.status)
            except:
                error_msg = text
                error_code = response.status
            
            raise ExchangeError(f"API error: {error_msg}", "binance", str(error_code))
    
    def _build_symbol_maps(self, exchange_info: Dict[str, Any]) -> None:
        """Build symbol mapping from exchange info."""
        # This would be implemented to map standard symbols to Binance symbols
        # For now, using simple normalization
        pass
    
    def _extract_fees(self, order_response: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract fees from order response."""
        # Binance doesn't return fees in order response by default
        # You'd need to call the trades endpoint to get actual fees
        return None