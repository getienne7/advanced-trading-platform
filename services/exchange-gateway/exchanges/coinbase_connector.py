"""
Coinbase Pro exchange connector for the Advanced Trading Platform.
Implements the ExchangeInterface for Coinbase Pro API integration.
"""
import asyncio
import aiohttp
import hmac
import hashlib
import base64
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


class CoinbaseConnector(ExchangeInterface):
    """Coinbase Pro exchange connector."""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str, sandbox: bool = True):
        super().__init__("coinbase")
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # API endpoints
        if sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.pro.coinbase.com"
        
        self.session = None
        
        # Rate limits (requests per second)
        self.rate_limits = {
            'private': 5,    # 5 requests per second for private endpoints
            'public': 10     # 10 requests per second for public endpoints
        }
        
        # Product mapping
        self.products = {}
    
    async def initialize(self) -> None:
        """Initialize the Coinbase connector."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Advanced-Trading-Platform/1.0'
                }
            )
            
            # Test connectivity
            await self._test_connectivity()
            
            # Load products
            await self._load_products()
            
            self.is_initialized = True
            logger.info("Coinbase connector initialized", sandbox=self.sandbox)
            
        except Exception as e:
            logger.error("Failed to initialize Coinbase connector", error=str(e))
            raise ExchangeError(f"Initialization failed: {e}", "coinbase")
    
    async def close(self) -> None:
        """Close the connector and cleanup resources."""
        if self.session:
            await self.session.close()
        self.is_initialized = False
        logger.info("Coinbase connector closed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Coinbase status and connectivity."""
        try:
            response = await self._make_request('GET', '/time')
            
            return {
                'status': 'online',
                'server_time': response.get('iso'),
                'sandbox': self.sandbox,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'offline',
                'error': str(e),
                'sandbox': self.sandbox,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get Coinbase exchange information."""
        try:
            products = await self._make_request('GET', '/products')
            
            return {
                'name': 'Coinbase Pro',
                'status': 'online',
                'products': len(products),
                'sandbox': self.sandbox,
                'supported_order_types': ['market', 'limit', 'stop'],
                'maker_fee': 0.005,  # 0.5%
                'taker_fee': 0.005   # 0.5%
            }
        except Exception as e:
            logger.error("Failed to get exchange info", error=str(e))
            raise ExchangeError(f"Failed to get exchange info: {e}", "coinbase")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for a symbol."""
        try:
            coinbase_symbol = self.normalize_symbol(symbol)
            
            # Get ticker
            ticker_response = await self._make_request('GET', f'/products/{coinbase_symbol}/ticker')
            
            # Get 24hr stats for volume
            stats_response = await self._make_request('GET', f'/products/{coinbase_symbol}/stats')
            
            return {
                'bid': float(ticker_response.get('bid', 0)),
                'ask': float(ticker_response.get('ask', 0)),
                'last': float(ticker_response.get('price', 0)),
                'volume': float(stats_response.get('volume', 0)),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to get ticker", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get ticker: {e}", "coinbase")
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book for a symbol."""
        try:
            coinbase_symbol = self.normalize_symbol(symbol)
            
            # Coinbase supports levels 1, 2, 3
            level = 2 if limit <= 50 else 3
            
            response = await self._make_request('GET', f'/products/{coinbase_symbol}/book', {
                'level': level
            })
            
            # Limit the results
            bids = response.get('bids', [])[:limit]
            asks = response.get('asks', [])[:limit]
            
            return {
                'bids': [[float(price), float(size)] for price, size, _ in bids],
                'asks': [[float(price), float(size)] for price, size, _ in asks],
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to get order book", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get order book: {e}", "coinbase")
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol."""
        try:
            coinbase_symbol = self.normalize_symbol(symbol)
            
            response = await self._make_request('GET', f'/products/{coinbase_symbol}/trades')
            
            trades = []
            for trade in response[:limit]:
                trades.append({
                    'id': str(trade['trade_id']),
                    'price': float(trade['price']),
                    'amount': float(trade['size']),
                    'side': trade['side'],
                    'timestamp': datetime.fromisoformat(trade['time'].replace('Z', '+00:00'))
                })
            
            return trades
        except Exception as e:
            logger.error("Failed to get trades", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get trades: {e}", "coinbase")
    
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
            coinbase_symbol = self.normalize_symbol(symbol)
            
            # Convert interval to Coinbase granularity (seconds)
            granularity_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '1h': 3600,
                '6h': 21600,
                '1d': 86400
            }
            granularity = granularity_map.get(interval, 3600)
            
            params = {
                'granularity': granularity
            }
            
            if start_time:
                params['start'] = start_time.isoformat()
            if end_time:
                params['end'] = end_time.isoformat()
            
            response = await self._make_request('GET', f'/products/{coinbase_symbol}/candles', params)
            
            klines = []
            for candle in response[:limit]:
                # Coinbase returns: [timestamp, low, high, open, close, volume]
                klines.append({
                    'timestamp': datetime.fromtimestamp(candle[0]),
                    'open': float(candle[3]),
                    'high': float(candle[2]),
                    'low': float(candle[1]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            # Sort by timestamp (Coinbase returns newest first)
            klines.sort(key=lambda x: x['timestamp'])
            
            return klines
        except Exception as e:
            logger.error("Failed to get klines", symbol=symbol, error=str(e))
            raise ExchangeError(f"Failed to get klines: {e}", "coinbase")
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """Get account balances."""
        try:
            response = await self._make_signed_request('GET', '/accounts')
            
            balances = {}
            for account in response:
                currency = account['currency']
                balance = float(account['balance'])
                available = float(account['available'])
                hold = float(account['hold'])
                
                if balance > 0:
                    balances[currency] = {
                        'free': available,
                        'locked': hold
                    }
            
            return balances
        except Exception as e:
            logger.error("Failed to get balances", error=str(e))
            raise ExchangeError(f"Failed to get balances: {e}", "coinbase")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information."""
        try:
            # Get profile info
            profiles = await self._make_signed_request('GET', '/profiles')
            
            # Get fees
            fees = await self._make_signed_request('GET', '/fees')
            
            return {
                'maker_commission': float(fees.get('maker_fee_rate', 0.005)),
                'taker_commission': float(fees.get('taker_fee_rate', 0.005)),
                'can_trade': True,
                'can_withdraw': True,
                'can_deposit': True,
                'account_type': 'PRO',
                'profiles': len(profiles),
                'usd_volume': float(fees.get('usd_volume', 0))
            }
        except Exception as e:
            logger.error("Failed to get account info", error=str(e))
            raise ExchangeError(f"Failed to get account info: {e}", "coinbase")
    
    async def place_advanced_order(
        self,
        symbol: str,
        side: str,
        type: str,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        post_only: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Place an advanced order with institutional features."""
        try:
            self.validate_order_params(symbol, side, type, amount, price)
            
            coinbase_symbol = self.normalize_symbol(symbol)
            
            params = {
                'product_id': coinbase_symbol,
                'side': side.lower(),
                'type': type.lower()
            }
            
            if type.lower() == 'market':
                if side.lower() == 'buy':
                    if 'funds' in kwargs:
                        params['funds'] = str(kwargs['funds'])
                    else:
                        ticker = await self.get_ticker(symbol)
                        params['funds'] = str(amount * ticker['ask'] * 1.01)
                else:
                    params['size'] = str(self.format_amount(amount, symbol))
            elif type.lower() == 'limit':
                params['size'] = str(self.format_amount(amount, symbol))
                params['price'] = str(self.format_price(price, symbol))
                
                if post_only:
                    params['post_only'] = True
                    
                if time_in_force != "GTC":
                    params['time_in_force'] = time_in_force
            elif type.lower() == 'stop':
                params['size'] = str(self.format_amount(amount, symbol))
                params['stop'] = 'loss' if side.lower() == 'sell' else 'entry'
                params['stop_price'] = str(stop_price or price)
                
                if price:  # Stop limit order
                    params['price'] = str(self.format_price(price, symbol))
            
            # Add client order ID for tracking
            if 'client_oid' in kwargs:
                params['client_oid'] = kwargs['client_oid']
            
            response = await self._make_signed_request('POST', '/orders', params)
            
            return {
                'order_id': response['id'],
                'symbol': symbol,
                'side': side.lower(),
                'type': type.lower(),
                'amount': float(response.get('size', amount)),
                'price': float(response.get('price', price or 0)),
                'status': response['status'],
                'timestamp': datetime.fromisoformat(response['created_at'].replace('Z', '+00:00')),
                'fees': None,
                'post_only': post_only,
                'client_oid': response.get('client_oid')
            }
        except Exception as e:
            logger.error("Failed to place advanced order", symbol=symbol, side=side, error=str(e))
            if "insufficient" in str(e).lower():
                raise InsufficientBalanceError(str(e), "coinbase")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(str(e), "coinbase")
            else:
                raise ExchangeError(f"Failed to place advanced order: {e}", "coinbase")

    async def get_fills(self, order_id: Optional[str] = None, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trade fills for institutional reporting."""
        try:
            params = {}
            if order_id:
                params['order_id'] = order_id
            if product_id:
                params['product_id'] = product_id
            
            response = await self._make_signed_request('GET', '/fills', params)
            
            fills = []
            for fill in response:
                fills.append({
                    'trade_id': fill['trade_id'],
                    'order_id': fill['order_id'],
                    'product_id': fill['product_id'],
                    'side': fill['side'],
                    'size': float(fill['size']),
                    'price': float(fill['price']),
                    'fee': float(fill['fee']),
                    'created_at': datetime.fromisoformat(fill['created_at'].replace('Z', '+00:00')),
                    'liquidity': fill['liquidity'],  # 'M' for maker, 'T' for taker
                    'settled': fill['settled']
                })
            
            return fills
        except Exception as e:
            logger.error("Failed to get fills", error=str(e))
            raise ExchangeError(f"Failed to get fills: {e}", "coinbase")

    async def get_funding_records(self, currency: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get funding records for institutional accounting."""
        try:
            params = {}
            if currency:
                params['currency'] = currency
            
            response = await self._make_signed_request('GET', '/transfers', params)
            
            records = []
            for record in response:
                records.append({
                    'id': record['id'],
                    'type': record['type'],
                    'amount': float(record['amount']),
                    'currency': record['details'].get('currency', ''),
                    'created_at': datetime.fromisoformat(record['created_at'].replace('Z', '+00:00')),
                    'completed_at': datetime.fromisoformat(record['completed_at'].replace('Z', '+00:00')) if record.get('completed_at') else None,
                    'details': record['details']
                })
            
            return records
        except Exception as e:
            logger.error("Failed to get funding records", error=str(e))
            raise ExchangeError(f"Failed to get funding records: {e}", "coinbase")

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
            
            coinbase_symbol = self.normalize_symbol(symbol)
            
            params = {
                'product_id': coinbase_symbol,
                'side': side.lower(),
                'type': type.lower()
            }
            
            if type.lower() == 'market':
                if side.lower() == 'buy':
                    # For market buy orders, specify funds (quote currency amount)
                    if 'funds' in kwargs:
                        params['funds'] = str(kwargs['funds'])
                    else:
                        # Estimate funds needed
                        ticker = await self.get_ticker(symbol)
                        params['funds'] = str(amount * ticker['ask'] * 1.01)  # Add 1% buffer
                else:
                    # For market sell orders, specify size (base currency amount)
                    params['size'] = str(self.format_amount(amount, symbol))
            else:
                # Limit order
                params['size'] = str(self.format_amount(amount, symbol))
                params['price'] = str(self.format_price(price, symbol))
                
                if time_in_force != "GTC":
                    params['time_in_force'] = time_in_force
            
            response = await self._make_signed_request('POST', '/orders', params)
            
            return {
                'order_id': response['id'],
                'symbol': symbol,
                'side': side.lower(),
                'type': type.lower(),
                'amount': float(response.get('size', amount)),
                'price': float(response.get('price', price or 0)),
                'status': response['status'],
                'timestamp': datetime.fromisoformat(response['created_at'].replace('Z', '+00:00')),
                'fees': None  # Fees are calculated after execution
            }
        except Exception as e:
            logger.error("Failed to place order", symbol=symbol, side=side, error=str(e))
            if "insufficient" in str(e).lower():
                raise InsufficientBalanceError(str(e), "coinbase")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(str(e), "coinbase")
            else:
                raise ExchangeError(f"Failed to place order: {e}", "coinbase")
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            response = await self._make_signed_request('DELETE', f'/orders/{order_id}')
            
            return {
                'order_id': order_id,
                'symbol': symbol,
                'status': 'canceled',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            if "not found" in str(e).lower():
                raise OrderNotFoundError(str(e), "coinbase")
            else:
                raise ExchangeError(f"Failed to cancel order: {e}", "coinbase")
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get order details."""
        try:
            response = await self._make_signed_request('GET', f'/orders/{order_id}')
            
            return {
                'order_id': response['id'],
                'symbol': self.denormalize_symbol(response['product_id']),
                'side': response['side'],
                'type': response['type'],
                'amount': float(response.get('size', 0)),
                'filled': float(response.get('filled_size', 0)),
                'price': float(response.get('price', 0)),
                'average_price': float(response.get('executed_value', 0)) / max(float(response.get('filled_size', 1)), 1),
                'status': response['status'],
                'timestamp': datetime.fromisoformat(response['created_at'].replace('Z', '+00:00')),
                'update_time': datetime.fromisoformat(response.get('done_at', response['created_at']).replace('Z', '+00:00'))
            }
        except Exception as e:
            logger.error("Failed to get order", order_id=order_id, error=str(e))
            if "not found" in str(e).lower():
                raise OrderNotFoundError(str(e), "coinbase")
            else:
                raise ExchangeError(f"Failed to get order: {e}", "coinbase")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        try:
            params = {}
            if symbol:
                params['product_id'] = self.normalize_symbol(symbol)
            
            response = await self._make_signed_request('GET', '/orders', params)
            
            orders = []
            for order in response:
                if order['status'] in ['open', 'pending']:
                    orders.append({
                        'order_id': order['id'],
                        'symbol': self.denormalize_symbol(order['product_id']),
                        'side': order['side'],
                        'type': order['type'],
                        'amount': float(order.get('size', 0)),
                        'filled': float(order.get('filled_size', 0)),
                        'price': float(order.get('price', 0)),
                        'status': order['status'],
                        'timestamp': datetime.fromisoformat(order['created_at'].replace('Z', '+00:00'))
                    })
            
            return orders
        except Exception as e:
            logger.error("Failed to get open orders", error=str(e))
            raise ExchangeError(f"Failed to get open orders: {e}", "coinbase")
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history."""
        try:
            params = {}
            if symbol:
                params['product_id'] = self.normalize_symbol(symbol)
            if start_time:
                params['start_date'] = start_time.isoformat()
            if end_time:
                params['end_date'] = end_time.isoformat()
            
            response = await self._make_signed_request('GET', '/orders', params)
            
            orders = []
            for order in response[:limit]:
                orders.append({
                    'order_id': order['id'],
                    'symbol': self.denormalize_symbol(order['product_id']),
                    'side': order['side'],
                    'type': order['type'],
                    'amount': float(order.get('size', 0)),
                    'filled': float(order.get('filled_size', 0)),
                    'price': float(order.get('price', 0)),
                    'average_price': float(order.get('executed_value', 0)) / max(float(order.get('filled_size', 1)), 1),
                    'status': order['status'],
                    'timestamp': datetime.fromisoformat(order['created_at'].replace('Z', '+00:00')),
                    'update_time': datetime.fromisoformat(order.get('done_at', order['created_at']).replace('Z', '+00:00'))
                })
            
            return orders
        except Exception as e:
            logger.error("Failed to get order history", error=str(e))
            raise ExchangeError(f"Failed to get order history: {e}", "coinbase")
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Coinbase."""
        # Convert BTC/USDT to BTC-USDT
        return symbol.upper().replace('/', '-')
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert Coinbase symbol back to standard format."""
        # Convert BTC-USDT to BTC/USDT
        return symbol.upper().replace('-', '/')
    
    async def _test_connectivity(self) -> None:
        """Test connectivity to Coinbase."""
        try:
            await self._make_request('GET', '/time')
            logger.info("Coinbase connectivity test successful")
        except Exception as e:
            logger.error("Coinbase connectivity test failed", error=str(e))
            raise
    
    async def _load_products(self) -> None:
        """Load available products from Coinbase."""
        try:
            products = await self._make_request('GET', '/products')
            self.products = {p['id']: p for p in products}
            logger.info("Loaded Coinbase products", count=len(self.products))
        except Exception as e:
            logger.warning("Failed to load products", error=str(e))
    
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a request to Coinbase API."""
        if not self.session:
            raise ExchangeError("Session not initialized", "coinbase")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method == 'POST':
                async with self.session.post(url, json=params) as response:
                    return await self._handle_response(response)
            elif method == 'DELETE':
                async with self.session.delete(url) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            raise ExchangeError(f"Request failed: {e}", "coinbase")
    
    async def _make_signed_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a signed request to Coinbase API."""
        timestamp = str(time.time())
        
        # Create message to sign
        if params and method in ['POST', 'PUT']:
            body = json.dumps(params)
        else:
            body = ''
        
        message = timestamp + method + endpoint + body
        
        # Create signature
        signature = base64.b64encode(
            hmac.new(
                base64.b64decode(self.secret_key),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        # Add headers
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase
        }
        
        if not self.session:
            raise ExchangeError("Session not initialized", "coinbase")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                async with self.session.get(url, headers=headers) as response:
                    return await self._handle_response(response)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                async with self.session.post(url, headers=headers, data=body) as response:
                    return await self._handle_response(response)
            elif method == 'DELETE':
                async with self.session.delete(url, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            raise ExchangeError(f"Request failed: {e}", "coinbase")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Any:
        """Handle API response."""
        text = await response.text()
        
        if response.status == 200:
            return json.loads(text) if text else {}
        elif response.status == 429:
            raise RateLimitError("Rate limit exceeded", "coinbase")
        else:
            try:
                error_data = json.loads(text)
                error_msg = error_data.get('message', text)
            except:
                error_msg = text
            
            raise ExchangeError(f"API error: {error_msg}", "coinbase", str(response.status)) + body

        
        # Decode the secret
        secret = base64.b64decode(self.api_secret)
        
        # Create signature
        signature = hmac.new(secret, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
        
        return signature_b64
    
    async def _make_request(self, 
                           method: str,
                           endpoint: str,
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None,
                           signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Coinbase Pro API."""
        if not self.session:
            raise RuntimeError("Not connected to exchange")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.get_base_url()}{endpoint}"
        headers = self._get_headers(signed)
        
        # Prepare request body
        body = ''
        if data:
            body = json.dumps(data)
        
        if signed:
            timestamp = str(time.time())
            path = endpoint
            if params:
                path += '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
            
            signature = self._generate_signature(timestamp, method, path, body)
            
            headers.update({
                'CB-ACCESS-KEY': self.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.passphrase
            })
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                data=body if body else None,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            raise