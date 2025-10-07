"""Kraken Exchange Connector
Implements Kraken-specific API integration.
"""
import asyncio
import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import structlog

from ..exchange_abstraction import (
    ExchangeConnector, OrderSide, OrderType, OrderStatus, ExchangeStatus,
    TradingPair, OrderBook, Ticker, Trade, Order, Balance, ExchangeInfo
)

logger = structlog.get_logger("kraken-connector")


class KrakenConnector(ExchangeConnector):
    """Kraken exchange connector."""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox, rate_limit_requests=20, rate_limit_window=1)
        # Kraken doesn't have a sandbox, but we keep the parameter for consistency
        self.base_url = "https://api.kraken.com"
    
    def get_name(self) -> str:
        """Get exchange name."""
        return "kraken"
    
    def get_base_url(self) -> str:
        """Get base API URL."""
        return self.base_url
    
    async def get_exchange_info(self) -> ExchangeInfo:
        """Get exchange information and trading pairs."""
        try:
            response = await self._make_request("GET", "/0/public/AssetPairs")
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            trading_pairs = {}
            for pair_name, pair_info in response.get('result', {}).items():
                if pair_info.get('status') == 'online':
                    # Parse Kraken's complex pair naming
                    base = pair_info['base']
                    quote = pair_info['quote']
                    
                    trading_pair = TradingPair(
                        symbol=pair_name,
                        base_asset=base,
                        quote_asset=quote,
                        min_quantity=Decimal(pair_info.get('ordermin', '0')),
                        max_quantity=Decimal('999999999'),  # No explicit max
                        quantity_precision=pair_info.get('lot_decimals', 8),
                        min_price=Decimal('0.00000001'),  # Kraken doesn't specify min price
                        max_price=Decimal('999999999'),
                        price_precision=pair_info.get('pair_decimals', 8),
                        min_notional=Decimal(pair_info.get('ordermin', '0')),
                        is_active=pair_info.get('status') == 'online',
                        fees={
                            'maker': Decimal(str(pair_info.get('fees_maker', [[0, 0.26]])[0][1])),
                            'taker': Decimal(str(pair_info.get('fees', [[0, 0.26]])[0][1]))
                        }
                    )
                    trading_pairs[pair_name] = trading_pair
            
            return ExchangeInfo(
                name=self.get_name(),
                status=self.status,
                trading_pairs=trading_pairs,
                rate_limits={
                    'api_calls_per_minute': {'limit': 60, 'window': 60},
                    'order_calls_per_minute': {'limit': 60, 'window': 60}
                },
                server_time=datetime.now(),  # Kraken doesn't provide server time in asset pairs
                timezone="UTC"
            )
        
        except Exception as e:
            logger.error(f"Failed to get exchange info: {str(e)}")
            raise
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Get order book for a symbol."""
        try:
            params = {'pair': symbol, 'count': min(limit, 500)}
            response = await self._make_request("GET", "/0/public/Depth", params=params)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            # Kraken returns data with the pair name as key
            pair_data = list(response['result'].values())[0]
            
            bids = [(Decimal(price), Decimal(volume)) for price, volume, _ in pair_data.get('bids', [])]
            asks = [(Decimal(price), Decimal(volume)) for price, volume, _ in pair_data.get('asks', [])]
            
            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks
            )
        
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {str(e)}")
            raise
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data for a symbol."""
        try:
            params = {'pair': symbol}
            response = await self._make_request("GET", "/0/public/Ticker", params=params)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            # Kraken returns data with the pair name as key
            ticker_data = list(response['result'].values())[0]
            
            # Parse Kraken's ticker format
            last_price = Decimal(ticker_data['c'][0])  # Last trade price
            open_price = Decimal(ticker_data['o'])     # Open price
            high_price = Decimal(ticker_data['h'][1])  # High price (24h)
            low_price = Decimal(ticker_data['l'][1])   # Low price (24h)
            volume = Decimal(ticker_data['v'][1])      # Volume (24h)
            
            return Ticker(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=last_price,
                volume=volume,
                quote_volume=volume * last_price,  # Approximate quote volume
                price_change=last_price - open_price,
                price_change_percent=((last_price - open_price) / open_price) * 100 if open_price > 0 else Decimal('0'),
                bid_price=Decimal(ticker_data['b'][0]),  # Best bid price
                ask_price=Decimal(ticker_data['a'][0]),  # Best ask price
                bid_quantity=Decimal(ticker_data['b'][2]),  # Bid volume
                ask_quantity=Decimal(ticker_data['a'][2])   # Ask volume
            )
        
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {str(e)}")
            raise
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for a symbol."""
        try:
            params = {'pair': symbol, 'count': min(limit, 1000)}
            response = await self._make_request("GET", "/0/public/Trades", params=params)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            # Kraken returns data with the pair name as key
            trades_data = list(response['result'].values())[0]
            
            trades = []
            for i, trade_data in enumerate(trades_data):
                price, volume, timestamp, side, order_type, misc = trade_data
                
                trades.append(Trade(
                    id=f"{symbol}_{int(float(timestamp))}_{i}",  # Generate ID from timestamp and index
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(float(timestamp)),
                    price=Decimal(price),
                    quantity=Decimal(volume),
                    side=OrderSide.BUY if side == 'b' else OrderSide.SELL,
                    is_buyer_maker=side == 's'  # Sell side is typically maker
                ))
            
            return trades
        
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {str(e)}")
            raise
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        try:
            response = await self._make_request("POST", "/0/private/Balance", signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            balances = {}
            for asset, balance_str in response.get('result', {}).items():
                balance_value = Decimal(balance_str)
                if balance_value > 0:  # Only include non-zero balances
                    balances[asset] = Balance(
                        asset=asset,
                        free=balance_value,  # Kraken doesn't separate free/locked in balance endpoint
                        locked=Decimal('0')
                    )
            
            return balances
        
        except Exception as e:
            logger.error(f"Failed to get balances: {str(e)}")
            raise
    
    async def place_order(self, 
                         symbol: str,
                         side: OrderSide,
                         order_type: OrderType,
                         quantity: Decimal,
                         price: Optional[Decimal] = None,
                         stop_price: Optional[Decimal] = None,
                         client_order_id: Optional[str] = None) -> Order:
        """Place an order."""
        try:
            data = {
                'pair': symbol,
                'type': side.value,
                'ordertype': self._convert_order_type(order_type),
                'volume': str(quantity)
            }
            
            if client_order_id:
                data['userref'] = client_order_id
            
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if price is None:
                    raise ValueError(f"Price required for {order_type} orders")
                data['price'] = str(price)
            
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if stop_price is None:
                    raise ValueError(f"Stop price required for {order_type} orders")
                data['price2'] = str(stop_price)  # Kraken uses price2 for stop price
            
            response = await self._make_request("POST", "/0/private/AddOrder", data=data, signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            # Kraken returns transaction IDs
            txids = response['result']['txid']
            order_id = txids[0] if txids else None
            
            if not order_id:
                raise Exception("No order ID returned from Kraken")
            
            # Get the order details
            return await self.get_order(symbol, order_id)
        
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel an order."""
        try:
            data = {'txid': order_id}
            response = await self._make_request("POST", "/0/private/CancelOrder", data=data, signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            # Get the order details after cancellation
            return await self.get_order(symbol, order_id)
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise
    
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        try:
            data = {'txid': order_id}
            response = await self._make_request("POST", "/0/private/QueryOrders", data=data, signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            order_data = response['result'].get(order_id)
            if not order_data:
                raise Exception(f"Order {order_id} not found")
            
            return self._parse_order_response(order_id, order_data)
        
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {str(e)}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        try:
            response = await self._make_request("POST", "/0/private/OpenOrders", signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            orders = []
            for order_id, order_data in response['result']['open'].items():
                # Filter by symbol if specified
                if symbol and order_data['descr']['pair'] != symbol:
                    continue
                
                orders.append(self._parse_order_response(order_id, order_data))
            
            return orders
        
        except Exception as e:
            logger.error(f"Failed to get open orders: {str(e)}")
            raise
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """Get order history."""
        try:
            data = {'ofs': 0}  # Offset
            response = await self._make_request("POST", "/0/private/ClosedOrders", data=data, signed=True)
            
            if response.get('error'):
                raise Exception(f"Kraken API error: {response['error']}")
            
            orders = []
            count = 0
            for order_id, order_data in response['result']['closed'].items():
                if count >= limit:
                    break
                
                # Filter by symbol if specified
                if symbol and order_data['descr']['pair'] != symbol:
                    continue
                
                orders.append(self._parse_order_response(order_id, order_data))
                count += 1
            
            return orders
        
        except Exception as e:
            logger.error(f"Failed to get order history: {str(e)}")
            raise
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Kraken format."""
        type_map = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop-loss',
            OrderType.STOP_LIMIT: 'stop-loss-limit'
        }
        return type_map.get(order_type, 'market')
    
    def _convert_order_status(self, status: str) -> OrderStatus:
        """Convert Kraken order status to internal format."""
        status_map = {
            'pending': OrderStatus.PENDING,
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED
        }
        return status_map.get(status, OrderStatus.PENDING)
    
    def _parse_order_response(self, order_id: str, order_data: Dict[str, Any]) -> Order:
        """Parse Kraken order response to internal Order format."""
        descr = order_data.get('descr', {})
        
        return Order(
            id=order_id,
            client_order_id=order_data.get('userref'),
            symbol=descr.get('pair', ''),
            side=OrderSide.BUY if descr.get('type') == 'buy' else OrderSide.SELL,
            type=self._parse_order_type(descr.get('ordertype', 'market')),
            status=self._convert_order_status(order_data.get('status', 'pending')),
            quantity=Decimal(order_data.get('vol', '0')),
            filled_quantity=Decimal(order_data.get('vol_exec', '0')),
            price=Decimal(descr.get('price', '0')) if descr.get('price') else None,
            stop_price=Decimal(descr.get('price2', '0')) if descr.get('price2') else None,
            timestamp=datetime.fromtimestamp(float(order_data.get('opentm', 0))),
            update_time=datetime.fromtimestamp(float(order_data.get('closetm', order_data.get('opentm', 0)))),
            fees=[]
        )
    
    def _parse_order_type(self, kraken_type: str) -> OrderType:
        """Parse Kraken order type to internal format."""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop-loss': OrderType.STOP,
            'stop-loss-limit': OrderType.STOP_LIMIT
        }
        return type_map.get(kraken_type, OrderType.MARKET)
    
    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """Get Kraken-specific request headers."""
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'AdvancedTradingPlatform/1.0'
        }
        
        return headers
    
    def _generate_signature(self, endpoint: str, data: Dict[str, Any], nonce: str) -> str:
        """Generate Kraken API signature."""
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode('utf-8')
        message = endpoint.encode('utf-8') + hashlib.sha256(encoded).digest()
        
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    async def _make_request(self, 
                           method: str,
                           endpoint: str,
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None,
                           signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Kraken API."""
        if not self.session:
            raise RuntimeError("Not connected to exchange")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.get_base_url()}{endpoint}"
        headers = self._get_headers(signed)
        
        if signed:
            if not data:
                data = {}
            
            # Add nonce for private endpoints
            nonce = str(int(time.time() * 1000000))
            data['nonce'] = nonce
            
            # Generate signature
            signature = self._generate_signature(endpoint, data, nonce)
            headers['API-Key'] = self.api_key
            headers['API-Sign'] = signature
            
            # Convert data to form-encoded string
            request_data = urllib.parse.urlencode(data)
        else:
            request_data = None
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                data=request_data,
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