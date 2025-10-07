"""
Arbitrage Detection Engine for Advanced Trading Platform.
Implements real-time price comparison, triangular arbitrage detection, and funding rate arbitrage.
"""
import asyncio
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal
import itertools
import math

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from database import get_postgres_session, get_redis
from utils import setup_logging
from models import ArbitrageOpportunity, ExchangeName

# Configure logging
logger = setup_logging("arbitrage-engine")

# Metrics
ARBITRAGE_OPPORTUNITIES_FOUND = Counter('arbitrage_opportunities_found_total', 'Total arbitrage opportunities found', ['type', 'symbol'])
ARBITRAGE_OPPORTUNITIES_EXECUTED = Counter('arbitrage_opportunities_executed_total', 'Total arbitrage opportunities executed', ['type', 'symbol'])
ARBITRAGE_PROFIT_TOTAL = Counter('arbitrage_profit_total', 'Total arbitrage profit', ['type', 'symbol'])
ARBITRAGE_SCAN_DURATION = Histogram('arbitrage_scan_duration_seconds', 'Time taken to scan for arbitrage opportunities', ['type'])
ACTIVE_OPPORTUNITIES = Gauge('active_arbitrage_opportunities', 'Number of active arbitrage opportunities', ['type'])

# Configuration
class ArbitrageConfig:
    """Arbitrage engine configuration."""
    
    def __init__(self):
        # Minimum profit thresholds
        self.min_simple_arbitrage_profit_pct = float(os.getenv("MIN_SIMPLE_ARBITRAGE_PROFIT_PCT", "0.5"))
        self.min_triangular_arbitrage_profit_pct = float(os.getenv("MIN_TRIANGULAR_ARBITRAGE_PROFIT_PCT", "0.3"))
        self.min_funding_arbitrage_profit_pct = float(os.getenv("MIN_FUNDING_ARBITRAGE_PROFIT_PCT", "1.0"))
        
        # Risk parameters
        self.max_position_size_usd = float(os.getenv("MAX_ARBITRAGE_POSITION_SIZE_USD", "10000"))
        self.max_execution_time_ms = int(os.getenv("MAX_ARBITRAGE_EXECUTION_TIME_MS", "5000"))
        self.min_liquidity_ratio = float(os.getenv("MIN_LIQUIDITY_RATIO", "2.0"))
        
        # Scanning intervals
        self.simple_arbitrage_scan_interval = int(os.getenv("SIMPLE_ARBITRAGE_SCAN_INTERVAL", "5"))  # seconds
        self.triangular_arbitrage_scan_interval = int(os.getenv("TRIANGULAR_ARBITRAGE_SCAN_INTERVAL", "10"))  # seconds
        self.funding_arbitrage_scan_interval = int(os.getenv("FUNDING_ARBITRAGE_SCAN_INTERVAL", "300"))  # seconds
        
        # Exchange gateway configuration
        self.exchange_gateway_url = os.getenv("EXCHANGE_GATEWAY_URL", "http://localhost:8006")
        
        # Supported symbols for arbitrage
        self.supported_symbols = os.getenv("ARBITRAGE_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT,ADA/USDT,SOL/USDT").split(",")
        
        # Triangular arbitrage base currencies
        self.triangular_base_currencies = os.getenv("TRIANGULAR_BASE_CURRENCIES", "USDT,BTC,ETH").split(",")

config = ArbitrageConfig()

# Pydantic models
class SimpleArbitrageOpportunity(BaseModel):
    """Simple arbitrage opportunity between two exchanges."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    profit_amount_per_unit: float
    max_position_size: float
    liquidity_score: float
    execution_time_estimate_ms: int
    risk_score: float
    detected_at: datetime
    expires_at: datetime

class TriangularArbitrageOpportunity(BaseModel):
    """Triangular arbitrage opportunity within a single exchange."""
    exchange: str
    base_currency: str
    path: List[str]  # e.g., ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
    trade_sequence: List[Dict[str, Any]]  # Detailed trade steps
    profit_pct: float
    required_capital: float
    execution_time_estimate_ms: int
    risk_score: float
    detected_at: datetime
    expires_at: datetime

class FundingArbitrageOpportunity(BaseModel):
    """Funding rate arbitrage opportunity."""
    symbol: str
    spot_exchange: str
    futures_exchange: str
    spot_price: float
    futures_price: float
    funding_rate: float
    funding_interval_hours: int
    annualized_profit_pct: float
    required_capital: float
    risk_score: float
    detected_at: datetime
    expires_at: datetime

class ArbitrageScanResult(BaseModel):
    """Result of arbitrage scanning."""
    scan_type: str
    opportunities_found: int
    scan_duration_ms: float
    timestamp: datetime
    opportunities: List[Any] = []

# FastAPI app
app = FastAPI(
    title="Arbitrage Detection Engine",
    description="Real-time arbitrage opportunity detection and analysis",
    version="1.0.0"
)

# Global state
arbitrage_opportunities: Dict[str, List[Any]] = {
    "simple": [],
    "triangular": [],
    "funding": []
}

exchange_data_cache: Dict[str, Dict[str, Any]] = {}
scanning_tasks: Dict[str, asyncio.Task] = {}

class ArbitrageDetector:
    """Main arbitrage detection engine."""
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.exchange_client = None  # Will be initialized with HTTP client
        
    async def initialize(self):
        """Initialize the arbitrage detector."""
        import aiohttp
        self.exchange_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.logger.info("Arbitrage detector initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.exchange_client:
            await self.exchange_client.close()
        self.logger.info("Arbitrage detector cleaned up")
    
    async def get_exchange_data(self, exchange: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data from exchange gateway."""
        try:
            # Get ticker data
            ticker_url = f"{self.config.exchange_gateway_url}/api/ticker/{exchange}/{symbol}"
            async with self.exchange_client.get(ticker_url) as response:
                if response.status != 200:
                    return None
                ticker_data = await response.json()
            
            # Get order book data
            orderbook_url = f"{self.config.exchange_gateway_url}/api/orderbook/{exchange}/{symbol}"
            async with self.exchange_client.get(orderbook_url, params={"limit": 20}) as response:
                if response.status != 200:
                    return None
                orderbook_data = await response.json()
            
            return {
                "ticker": ticker_data,
                "orderbook": orderbook_data,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.warning("Failed to get exchange data", 
                              exchange=exchange, symbol=symbol, error=str(e))
            return None
    
    async def scan_simple_arbitrage(self) -> ArbitrageScanResult:
        """Scan for simple arbitrage opportunities across exchanges."""
        start_time = datetime.utcnow()
        opportunities = []
        
        try:
            with ARBITRAGE_SCAN_DURATION.labels(type="simple").time():
                # Get available exchanges
                exchanges_url = f"{self.config.exchange_gateway_url}/api/exchanges"
                async with self.exchange_client.get(exchanges_url) as response:
                    if response.status != 200:
                        raise Exception("Failed to get exchange list")
                    exchanges_data = await response.json()
                
                available_exchanges = list(exchanges_data.get("exchanges", {}).keys())
                
                if len(available_exchanges) < 2:
                    self.logger.warning("Need at least 2 exchanges for simple arbitrage")
                    return ArbitrageScanResult(
                        scan_type="simple",
                        opportunities_found=0,
                        scan_duration_ms=0,
                        timestamp=datetime.utcnow(),
                        opportunities=[]
                    )
                
                # Scan each symbol across all exchange pairs
                for symbol in self.config.supported_symbols:
                    symbol_opportunities = await self._scan_symbol_simple_arbitrage(
                        symbol, available_exchanges
                    )
                    opportunities.extend(symbol_opportunities)
                
                # Update metrics
                ARBITRAGE_OPPORTUNITIES_FOUND.labels(type="simple", symbol="all").inc(len(opportunities))
                ACTIVE_OPPORTUNITIES.labels(type="simple").set(len(opportunities))
                
                # Store opportunities
                arbitrage_opportunities["simple"] = opportunities
                opportunity_store.add_opportunities("simple", opportunities)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self.logger.info("Simple arbitrage scan completed", 
                               opportunities_found=len(opportunities),
                               duration_ms=duration_ms)
                
                return ArbitrageScanResult(
                    scan_type="simple",
                    opportunities_found=len(opportunities),
                    scan_duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    opportunities=opportunities
                )
                
        except Exception as e:
            self.logger.error("Simple arbitrage scan failed", error=str(e))
            return ArbitrageScanResult(
                scan_type="simple",
                opportunities_found=0,
                scan_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                timestamp=datetime.utcnow(),
                opportunities=[]
            )
    
    async def _scan_symbol_simple_arbitrage(self, symbol: str, exchanges: List[str]) -> List[SimpleArbitrageOpportunity]:
        """Scan for simple arbitrage opportunities for a specific symbol."""
        opportunities = []
        exchange_data = {}
        
        # Collect data from all exchanges concurrently for better performance
        tasks = []
        for exchange in exchanges:
            task = self.get_exchange_data(exchange, symbol)
            tasks.append((exchange, task))
        
        # Wait for all data collection tasks to complete
        for exchange, task in tasks:
            try:
                data = await task
                if data:
                    # Add timestamp validation to ensure data freshness
                    data_age = (datetime.utcnow() - data["timestamp"]).total_seconds()
                    if data_age < 30:  # Only use data that's less than 30 seconds old
                        exchange_data[exchange] = data
                    else:
                        self.logger.warning("Stale data detected", exchange=exchange, symbol=symbol, age_seconds=data_age)
            except Exception as e:
                self.logger.warning("Failed to get data for exchange", exchange=exchange, symbol=symbol, error=str(e))
        
        if len(exchange_data) < 2:
            return opportunities
        
        # Compare all exchange pairs with enhanced validation
        for exchange_a, exchange_b in itertools.combinations(exchange_data.keys(), 2):
            data_a = exchange_data[exchange_a]
            data_b = exchange_data[exchange_b]
            
            # Validate data quality before comparison
            if not self._validate_market_data(data_a) or not self._validate_market_data(data_b):
                continue
            
            # Check A -> B arbitrage (buy on A, sell on B)
            opportunity_ab = self._calculate_simple_arbitrage(
                symbol, exchange_a, exchange_b, data_a, data_b
            )
            if opportunity_ab:
                opportunities.append(opportunity_ab)
            
            # Check B -> A arbitrage (buy on B, sell on A)
            opportunity_ba = self._calculate_simple_arbitrage(
                symbol, exchange_b, exchange_a, data_b, data_a
            )
            if opportunity_ba:
                opportunities.append(opportunity_ba)
        
        return opportunities
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data quality and completeness."""
        try:
            ticker = data.get("ticker", {})
            orderbook = data.get("orderbook", {})
            
            # Check ticker data
            required_ticker_fields = ["bid", "ask", "last"]
            for field in required_ticker_fields:
                if field not in ticker or ticker[field] is None or ticker[field] <= 0:
                    return False
            
            # Check spread sanity (bid should be less than ask)
            if ticker["bid"] >= ticker["ask"]:
                return False
            
            # Check for reasonable spread (not more than 5%)
            spread_pct = ((ticker["ask"] - ticker["bid"]) / ticker["bid"]) * 100
            if spread_pct > 5.0:
                return False
            
            # Check order book data
            if not orderbook.get("bids") or not orderbook.get("asks"):
                return False
            
            # Ensure order book has reasonable depth
            if len(orderbook["bids"]) < 3 or len(orderbook["asks"]) < 3:
                return False
            
            # Validate order book price levels
            for bid in orderbook["bids"][:5]:  # Check first 5 levels
                if len(bid) != 2 or bid[0] <= 0 or bid[1] <= 0:
                    return False
            
            for ask in orderbook["asks"][:5]:  # Check first 5 levels
                if len(ask) != 2 or ask[0] <= 0 or ask[1] <= 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_simple_arbitrage(
        self, 
        symbol: str, 
        buy_exchange: str, 
        sell_exchange: str,
        buy_data: Dict[str, Any], 
        sell_data: Dict[str, Any]
    ) -> Optional[SimpleArbitrageOpportunity]:
        """Calculate simple arbitrage opportunity between two exchanges."""
        try:
            buy_ticker = buy_data["ticker"]
            sell_ticker = sell_data["ticker"]
            buy_orderbook = buy_data["orderbook"]
            sell_orderbook = sell_data["orderbook"]
            
            # Get best prices
            buy_price = buy_ticker["ask"]  # We buy at ask price
            sell_price = sell_ticker["bid"]  # We sell at bid price
            
            # Calculate profit
            profit_per_unit = sell_price - buy_price
            profit_pct = (profit_per_unit / buy_price) * 100
            
            # Check if profitable
            if profit_pct < self.config.min_simple_arbitrage_profit_pct:
                return None
            
            # Calculate liquidity and position sizing
            buy_liquidity = self._calculate_liquidity(buy_orderbook["asks"], "buy")
            sell_liquidity = self._calculate_liquidity(sell_orderbook["bids"], "sell")
            
            max_position_size = min(
                buy_liquidity * self.config.min_liquidity_ratio,
                sell_liquidity * self.config.min_liquidity_ratio,
                self.config.max_position_size_usd / buy_price
            )
            
            if max_position_size < 0.001:  # Minimum viable position
                return None
            
            # Calculate liquidity score
            liquidity_score = min(buy_liquidity, sell_liquidity) / max_position_size
            
            # Estimate execution time (based on exchange latency and order book depth)
            execution_time_ms = self._estimate_execution_time(buy_orderbook, sell_orderbook)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                profit_pct, liquidity_score, execution_time_ms, max_position_size
            )
            
            # Check if execution time is acceptable
            if execution_time_ms > self.config.max_execution_time_ms:
                return None
            
            return SimpleArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                profit_pct=profit_pct,
                profit_amount_per_unit=profit_per_unit,
                max_position_size=max_position_size,
                liquidity_score=liquidity_score,
                execution_time_estimate_ms=execution_time_ms,
                risk_score=risk_score,
                detected_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=30)  # 30 second expiry
            )
            
        except Exception as e:
            self.logger.warning("Failed to calculate simple arbitrage", 
                              symbol=symbol, buy_exchange=buy_exchange, 
                              sell_exchange=sell_exchange, error=str(e))
            return None
    
    async def scan_triangular_arbitrage(self) -> ArbitrageScanResult:
        """Scan for triangular arbitrage opportunities within exchanges."""
        start_time = datetime.utcnow()
        opportunities = []
        
        try:
            with ARBITRAGE_SCAN_DURATION.labels(type="triangular").time():
                # Get available exchanges
                exchanges_url = f"{self.config.exchange_gateway_url}/api/exchanges"
                async with self.exchange_client.get(exchanges_url) as response:
                    if response.status != 200:
                        raise Exception("Failed to get exchange list")
                    exchanges_data = await response.json()
                
                available_exchanges = list(exchanges_data.get("exchanges", {}).keys())
                
                # Scan each exchange for triangular arbitrage
                for exchange in available_exchanges:
                    exchange_opportunities = await self._scan_exchange_triangular_arbitrage(exchange)
                    opportunities.extend(exchange_opportunities)
                
                # Update metrics
                ARBITRAGE_OPPORTUNITIES_FOUND.labels(type="triangular", symbol="all").inc(len(opportunities))
                ACTIVE_OPPORTUNITIES.labels(type="triangular").set(len(opportunities))
                
                # Store opportunities
                arbitrage_opportunities["triangular"] = opportunities
                opportunity_store.add_opportunities("triangular", opportunities)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self.logger.info("Triangular arbitrage scan completed", 
                               opportunities_found=len(opportunities),
                               duration_ms=duration_ms)
                
                return ArbitrageScanResult(
                    scan_type="triangular",
                    opportunities_found=len(opportunities),
                    scan_duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    opportunities=opportunities
                )
                
        except Exception as e:
            self.logger.error("Triangular arbitrage scan failed", error=str(e))
            return ArbitrageScanResult(
                scan_type="triangular",
                opportunities_found=0,
                scan_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                timestamp=datetime.utcnow(),
                opportunities=[]
            )
    
    async def _scan_exchange_triangular_arbitrage(self, exchange: str) -> List[TriangularArbitrageOpportunity]:
        """Scan for triangular arbitrage opportunities within a single exchange."""
        opportunities = []
        
        try:
            # Get exchange info to find available trading pairs
            exchange_info_url = f"{self.config.exchange_gateway_url}/api/exchanges/{exchange}/status"
            async with self.exchange_client.get(exchange_info_url) as response:
                if response.status != 200:
                    return opportunities
                exchange_info = await response.json()
            
            # For each base currency, find triangular paths
            for base_currency in self.config.triangular_base_currencies:
                triangular_opportunities = await self._find_triangular_paths(exchange, base_currency)
                opportunities.extend(triangular_opportunities)
            
        except Exception as e:
            self.logger.warning("Failed to scan triangular arbitrage for exchange", 
                              exchange=exchange, error=str(e))
        
        return opportunities
    
    async def _find_triangular_paths(self, exchange: str, base_currency: str) -> List[TriangularArbitrageOpportunity]:
        """Find triangular arbitrage paths for a base currency."""
        opportunities = []
        
        # Enhanced triangular paths with more comprehensive coverage
        if base_currency == "USDT":
            paths = [
                # Major crypto triangular paths
                ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
                ["BTC/USDT", "BNB/BTC", "BNB/USDT"],
                ["ETH/USDT", "BNB/ETH", "BNB/USDT"],
                ["BTC/USDT", "ADA/BTC", "ADA/USDT"],
                ["ETH/USDT", "SOL/ETH", "SOL/USDT"],
                ["BTC/USDT", "DOT/BTC", "DOT/USDT"],
                ["ETH/USDT", "LINK/ETH", "LINK/USDT"],
                ["BNB/USDT", "ADA/BNB", "ADA/USDT"],
                # Cross-chain opportunities
                ["BTC/USDT", "AVAX/BTC", "AVAX/USDT"],
                ["ETH/USDT", "MATIC/ETH", "MATIC/USDT"]
            ]
        elif base_currency == "BTC":
            paths = [
                ["ETH/BTC", "BNB/ETH", "BNB/BTC"],
                ["ETH/BTC", "ADA/ETH", "ADA/BTC"],
                ["BNB/BTC", "ADA/BNB", "ADA/BTC"],
                ["ETH/BTC", "SOL/ETH", "SOL/BTC"],
                ["BNB/BTC", "DOT/BNB", "DOT/BTC"]
            ]
        elif base_currency == "ETH":
            paths = [
                ["BNB/ETH", "ADA/BNB", "ADA/ETH"],
                ["SOL/ETH", "AVAX/SOL", "AVAX/ETH"],
                ["LINK/ETH", "DOT/LINK", "DOT/ETH"]
            ]
        else:
            paths = []
        
        # Check each path for arbitrage opportunities with concurrent processing
        path_tasks = []
        for path in paths:
            # Validate path exists on exchange before calculating
            if await self._validate_triangular_path(exchange, path):
                task = self._calculate_triangular_arbitrage(exchange, base_currency, path)
                path_tasks.append(task)
        
        # Process all paths concurrently
        if path_tasks:
            results = await asyncio.gather(*path_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, TriangularArbitrageOpportunity):
                    opportunities.append(result)
                elif isinstance(result, Exception):
                    self.logger.warning("Triangular arbitrage calculation failed", error=str(result))
        
        return opportunities
    
    async def _validate_triangular_path(self, exchange: str, path: List[str]) -> bool:
        """Validate that all symbols in a triangular path exist on the exchange."""
        try:
            # Check if exchange supports all symbols in the path
            exchange_info_url = f"{self.config.exchange_gateway_url}/api/exchanges/{exchange}/symbols"
            async with self.exchange_client.get(exchange_info_url) as response:
                if response.status != 200:
                    return True  # Assume valid if we can't check
                
                symbols_data = await response.json()
                available_symbols = symbols_data.get("symbols", [])
                
                # Check if all path symbols are available
                for symbol in path:
                    if symbol not in available_symbols:
                        return False
                
                return True
                
        except Exception:
            return True  # Assume valid if validation fails
    
    async def _calculate_triangular_arbitrage(
        self, 
        exchange: str, 
        base_currency: str, 
        path: List[str]
    ) -> Optional[TriangularArbitrageOpportunity]:
        """Calculate triangular arbitrage opportunity for a specific path."""
        try:
            # Get market data for all symbols in the path
            market_data = {}
            for symbol in path:
                data = await self.get_exchange_data(exchange, symbol)
                if not data:
                    return None
                market_data[symbol] = data
            
            # Calculate the triangular arbitrage
            # Start with 1 unit of base currency
            starting_amount = 1.0
            current_amount = starting_amount
            trade_sequence = []
            
            # Execute the triangular path
            for i, symbol in enumerate(path):
                ticker = market_data[symbol]["ticker"]
                orderbook = market_data[symbol]["orderbook"]
                
                # Determine trade direction based on path position
                if i == 0:
                    # First trade: base currency -> intermediate currency
                    # Buy the symbol (spend base currency)
                    price = ticker["ask"]
                    new_amount = current_amount / price
                    side = "buy"
                elif i == len(path) - 1:
                    # Last trade: intermediate currency -> base currency
                    # Sell the symbol (get base currency back)
                    price = ticker["bid"]
                    new_amount = current_amount * price
                    side = "sell"
                else:
                    # Middle trade: intermediate currency -> another intermediate currency
                    # Determine direction based on symbol format
                    base_asset, quote_asset = symbol.split("/")
                    if i == 1:
                        # Usually sell the intermediate for the next intermediate
                        price = ticker["bid"]
                        new_amount = current_amount * price
                        side = "sell"
                    else:
                        price = ticker["ask"]
                        new_amount = current_amount / price
                        side = "buy"
                
                trade_sequence.append({
                    "step": i + 1,
                    "symbol": symbol,
                    "side": side,
                    "amount_in": current_amount,
                    "price": price,
                    "amount_out": new_amount,
                    "liquidity": self._calculate_liquidity(
                        orderbook["asks"] if side == "buy" else orderbook["bids"], 
                        side
                    )
                })
                
                current_amount = new_amount
            
            # Calculate profit
            final_amount = current_amount
            profit_amount = final_amount - starting_amount
            profit_pct = (profit_amount / starting_amount) * 100
            
            # Check if profitable
            if profit_pct < self.config.min_triangular_arbitrage_profit_pct:
                return None
            
            # Calculate required capital (in USD equivalent)
            first_trade = trade_sequence[0]
            if base_currency == "USDT":
                required_capital = starting_amount
            else:
                # Convert to USD equivalent (simplified)
                btc_usdt_data = await self.get_exchange_data(exchange, "BTC/USDT")
                if btc_usdt_data and base_currency == "BTC":
                    required_capital = starting_amount * btc_usdt_data["ticker"]["last"]
                else:
                    required_capital = starting_amount * 1000  # Rough estimate
            
            # Check position size limits
            if required_capital > self.config.max_position_size_usd:
                return None
            
            # Calculate execution time estimate
            execution_time_ms = len(path) * 1000  # Rough estimate: 1 second per trade
            
            # Calculate risk score
            min_liquidity = min(trade["liquidity"] for trade in trade_sequence)
            risk_score = self._calculate_triangular_risk_score(
                profit_pct, min_liquidity, execution_time_ms, len(path)
            )
            
            return TriangularArbitrageOpportunity(
                exchange=exchange,
                base_currency=base_currency,
                path=path,
                trade_sequence=trade_sequence,
                profit_pct=profit_pct,
                required_capital=required_capital,
                execution_time_estimate_ms=execution_time_ms,
                risk_score=risk_score,
                detected_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=60)  # 60 second expiry
            )
            
        except Exception as e:
            self.logger.warning("Failed to calculate triangular arbitrage", 
                              exchange=exchange, base_currency=base_currency, 
                              path=path, error=str(e))
            return None
    
    async def scan_funding_arbitrage(self) -> ArbitrageScanResult:
        """Scan for funding rate arbitrage opportunities."""
        start_time = datetime.utcnow()
        opportunities = []
        
        try:
            with ARBITRAGE_SCAN_DURATION.labels(type="funding").time():
                # Get available exchanges that support futures trading
                exchanges_url = f"{self.config.exchange_gateway_url}/api/exchanges"
                async with self.exchange_client.get(exchanges_url) as response:
                    if response.status != 200:
                        raise Exception("Failed to get exchange list")
                    exchanges_data = await response.json()
                
                available_exchanges = list(exchanges_data.get("exchanges", {}).keys())
                
                # Filter exchanges that support futures (for funding rates)
                futures_exchanges = []
                spot_exchanges = []
                
                for exchange in available_exchanges:
                    # Check if exchange supports futures
                    exchange_info_url = f"{self.config.exchange_gateway_url}/api/exchanges/{exchange}/status"
                    try:
                        async with self.exchange_client.get(exchange_info_url) as response:
                            if response.status == 200:
                                exchange_info = await response.json()
                                if exchange_info.get("supports_futures", False):
                                    futures_exchanges.append(exchange)
                                if exchange_info.get("supports_spot", True):
                                    spot_exchanges.append(exchange)
                    except Exception as e:
                        self.logger.warning("Failed to get exchange info", exchange=exchange, error=str(e))
                        # Assume it supports spot trading by default
                        spot_exchanges.append(exchange)
                
                # Scan each symbol for funding arbitrage opportunities
                for symbol in self.config.supported_symbols:
                    symbol_opportunities = await self._scan_symbol_funding_arbitrage(
                        symbol, spot_exchanges, futures_exchanges
                    )
                    opportunities.extend(symbol_opportunities)
                
                # Update metrics
                ARBITRAGE_OPPORTUNITIES_FOUND.labels(type="funding", symbol="all").inc(len(opportunities))
                ACTIVE_OPPORTUNITIES.labels(type="funding").set(len(opportunities))
                
                # Store opportunities
                arbitrage_opportunities["funding"] = opportunities
                opportunity_store.add_opportunities("funding", opportunities)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                self.logger.info("Funding arbitrage scan completed", 
                               opportunities_found=len(opportunities),
                               duration_ms=duration_ms)
                
                return ArbitrageScanResult(
                    scan_type="funding",
                    opportunities_found=len(opportunities),
                    scan_duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    opportunities=opportunities
                )
                
        except Exception as e:
            self.logger.error("Funding arbitrage scan failed", error=str(e))
            return ArbitrageScanResult(
                scan_type="funding",
                opportunities_found=0,
                scan_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                timestamp=datetime.utcnow(),
                opportunities=[]
            )
    
    async def _scan_symbol_funding_arbitrage(
        self, 
        symbol: str, 
        spot_exchanges: List[str], 
        futures_exchanges: List[str]
    ) -> List[FundingArbitrageOpportunity]:
        """Scan for funding arbitrage opportunities for a specific symbol."""
        opportunities = []
        
        if not spot_exchanges or not futures_exchanges:
            return opportunities
        
        try:
            # Get spot prices from all spot exchanges
            spot_data = {}
            for exchange in spot_exchanges:
                data = await self.get_exchange_data(exchange, symbol)
                if data:
                    spot_data[exchange] = data
            
            # Get futures data and funding rates
            futures_data = {}
            for exchange in futures_exchanges:
                # Get futures price data
                futures_symbol = symbol.replace("/", "")  # Convert BTC/USDT to BTCUSDT for futures
                futures_data_result = await self._get_futures_data(exchange, futures_symbol)
                if futures_data_result:
                    futures_data[exchange] = futures_data_result
            
            # Compare each spot exchange with each futures exchange
            for spot_exchange, spot_info in spot_data.items():
                for futures_exchange, futures_info in futures_data.items():
                    if spot_exchange == futures_exchange:
                        continue  # Skip same exchange comparisons
                    
                    opportunity = self._calculate_funding_arbitrage(
                        symbol, spot_exchange, futures_exchange, spot_info, futures_info
                    )
                    if opportunity:
                        opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.warning("Failed to scan funding arbitrage for symbol", 
                              symbol=symbol, error=str(e))
        
        return opportunities
    
    async def _get_futures_data(self, exchange: str, futures_symbol: str) -> Optional[Dict[str, Any]]:
        """Get futures market data including funding rate."""
        try:
            # Get futures ticker data
            futures_ticker_url = f"{self.config.exchange_gateway_url}/api/futures/ticker/{exchange}/{futures_symbol}"
            async with self.exchange_client.get(futures_ticker_url) as response:
                if response.status != 200:
                    return None
                futures_ticker = await response.json()
            
            # Get funding rate data
            funding_rate_url = f"{self.config.exchange_gateway_url}/api/futures/funding_rate/{exchange}/{futures_symbol}"
            async with self.exchange_client.get(funding_rate_url) as response:
                if response.status != 200:
                    # If funding rate endpoint doesn't exist, estimate it
                    funding_rate_data = {
                        "funding_rate": 0.0001,  # Default 0.01% funding rate
                        "funding_interval_hours": 8,  # Default 8-hour funding interval
                        "next_funding_time": datetime.utcnow() + timedelta(hours=8)
                    }
                else:
                    funding_rate_data = await response.json()
            
            return {
                "ticker": futures_ticker,
                "funding_rate": funding_rate_data["funding_rate"],
                "funding_interval_hours": funding_rate_data.get("funding_interval_hours", 8),
                "next_funding_time": funding_rate_data.get("next_funding_time"),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.warning("Failed to get futures data", 
                              exchange=exchange, symbol=futures_symbol, error=str(e))
            return None
    
    def _calculate_funding_arbitrage(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
        spot_data: Dict[str, Any],
        futures_data: Dict[str, Any]
    ) -> Optional[FundingArbitrageOpportunity]:
        """Calculate funding rate arbitrage opportunity."""
        try:
            spot_ticker = spot_data["ticker"]
            futures_ticker = futures_data["ticker"]
            
            spot_price = spot_ticker["last"]
            futures_price = futures_ticker["last"]
            funding_rate = futures_data["funding_rate"]
            funding_interval_hours = futures_data["funding_interval_hours"]
            
            # Calculate price difference (basis)
            price_diff = futures_price - spot_price
            price_diff_pct = (price_diff / spot_price) * 100
            
            # Calculate annualized funding rate
            funding_periods_per_year = (365 * 24) / funding_interval_hours
            annualized_funding_rate = funding_rate * funding_periods_per_year * 100
            
            # Determine arbitrage strategy based on funding rate and price difference
            if funding_rate > 0:
                # Positive funding rate: long pays short
                # Strategy: Short futures, long spot
                strategy_type = "short_futures_long_spot"
                expected_profit_pct = abs(annualized_funding_rate) - abs(price_diff_pct)
            else:
                # Negative funding rate: short pays long
                # Strategy: Long futures, short spot
                strategy_type = "long_futures_short_spot"
                expected_profit_pct = abs(annualized_funding_rate) - abs(price_diff_pct)
            
            # Check if profitable
            if expected_profit_pct < self.config.min_funding_arbitrage_profit_pct:
                return None
            
            # Calculate required capital (for both spot and futures positions)
            base_capital = self.config.max_position_size_usd / 2  # Split between spot and futures
            required_capital = base_capital * 2  # Total capital needed
            
            # Check capital limits
            if required_capital > self.config.max_position_size_usd:
                required_capital = self.config.max_position_size_usd
            
            # Calculate risk score based on various factors
            risk_score = self._calculate_funding_risk_score(
                abs(funding_rate), abs(price_diff_pct), expected_profit_pct, required_capital
            )
            
            # Estimate holding period (until next funding payment)
            next_funding_time = futures_data.get("next_funding_time")
            if next_funding_time:
                if isinstance(next_funding_time, str):
                    next_funding_time = datetime.fromisoformat(next_funding_time.replace('Z', '+00:00'))
                holding_hours = (next_funding_time - datetime.utcnow()).total_seconds() / 3600
            else:
                holding_hours = funding_interval_hours
            
            return FundingArbitrageOpportunity(
                symbol=symbol,
                spot_exchange=spot_exchange,
                futures_exchange=futures_exchange,
                spot_price=spot_price,
                futures_price=futures_price,
                funding_rate=funding_rate,
                funding_interval_hours=funding_interval_hours,
                annualized_profit_pct=expected_profit_pct,
                required_capital=required_capital,
                risk_score=risk_score,
                detected_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=min(holding_hours, 24))  # Max 24 hour expiry
            )
            
        except Exception as e:
            self.logger.warning("Failed to calculate funding arbitrage", 
                              symbol=symbol, spot_exchange=spot_exchange, 
                              futures_exchange=futures_exchange, error=str(e))
            return None
    
    def _calculate_funding_risk_score(
        self,
        funding_rate: float,
        price_diff_pct: float,
        expected_profit_pct: float,
        required_capital: float
    ) -> float:
        """Calculate risk score for funding arbitrage (0-100, lower is better)."""
        # Risk factors for funding arbitrage
        funding_volatility_risk = min(abs(funding_rate) * 10000, 30)  # Higher risk for volatile funding rates
        basis_risk = min(abs(price_diff_pct) * 2, 25)  # Risk from price difference between spot and futures
        profit_risk = max(0, (2.0 - expected_profit_pct) * 10)  # Higher risk for lower expected profit
        capital_risk = (required_capital / self.config.max_position_size_usd) * 15  # Higher risk for larger positions
        
        # Liquidity risk (simplified - in real implementation would check order book depth)
        liquidity_risk = 10  # Base liquidity risk
        
        total_risk = funding_volatility_risk + basis_risk + profit_risk + capital_risk + liquidity_risk
        return min(100, max(0, total_risk))
    
    def _calculate_liquidity(self, order_levels: List[List[float]], side: str) -> float:
        """Calculate available liquidity from order book levels."""
        try:
            # Sum up the first 10 levels
            total_liquidity = sum(level[1] for level in order_levels[:10])
            return total_liquidity
        except:
            return 0.0
    
    def _estimate_execution_time(self, buy_orderbook: Dict, sell_orderbook: Dict) -> int:
        """Estimate execution time based on order book depth and exchange latency."""
        # Simplified estimation based on order book depth
        buy_depth = len(buy_orderbook.get("asks", []))
        sell_depth = len(sell_orderbook.get("bids", []))
        
        # Base latency + depth factor
        base_latency_ms = 500  # 500ms base latency
        depth_factor = max(0, 10 - min(buy_depth, sell_depth)) * 100  # Penalty for low depth
        
        return base_latency_ms + depth_factor
    
    def _calculate_risk_score(
        self, 
        profit_pct: float, 
        liquidity_score: float, 
        execution_time_ms: int, 
        position_size: float
    ) -> float:
        """Calculate risk score for simple arbitrage (0-100, lower is better)."""
        # Risk factors
        profit_risk = max(0, (2.0 - profit_pct) * 10)  # Higher risk for lower profit
        liquidity_risk = max(0, (2.0 - liquidity_score) * 20)  # Higher risk for lower liquidity
        time_risk = (execution_time_ms / 1000) * 5  # Higher risk for longer execution
        size_risk = (position_size / self.config.max_position_size_usd) * 10  # Higher risk for larger positions
        
        total_risk = profit_risk + liquidity_risk + time_risk + size_risk
        return min(100, max(0, total_risk))
    
    def _calculate_triangular_risk_score(
        self, 
        profit_pct: float, 
        min_liquidity: float, 
        execution_time_ms: int, 
        num_trades: int
    ) -> float:
        """Calculate risk score for triangular arbitrage (0-100, lower is better)."""
        # Risk factors for triangular arbitrage
        profit_risk = max(0, (1.0 - profit_pct) * 15)  # Higher risk for lower profit
        liquidity_risk = max(0, (1.0 - min_liquidity) * 25)  # Higher risk for lower liquidity
        time_risk = (execution_time_ms / 1000) * 8  # Higher risk for longer execution
        complexity_risk = num_trades * 5  # Higher risk for more complex paths
        
        total_risk = profit_risk + liquidity_risk + time_risk + complexity_risk
        return min(100, max(0, total_risk))

# Global arbitrage detector instance
arbitrage_detector = ArbitrageDetector()

# Real-time price monitoring and alerting
class PriceMonitor:
    """Real-time price monitoring for arbitrage detection."""
    
    def __init__(self):
        self.price_cache = {}
        self.price_alerts = {}
        self.volatility_tracker = {}
    
    def update_price(self, exchange: str, symbol: str, price: float):
        """Update price cache and check for alerts."""
        key = f"{exchange}:{symbol}"
        current_time = datetime.utcnow()
        
        if key in self.price_cache:
            old_price = self.price_cache[key]["price"]
            price_change_pct = ((price - old_price) / old_price) * 100
            
            # Track volatility
            if key not in self.volatility_tracker:
                self.volatility_tracker[key] = []
            
            self.volatility_tracker[key].append({
                "timestamp": current_time,
                "price_change_pct": abs(price_change_pct)
            })
            
            # Keep only last 100 price changes
            if len(self.volatility_tracker[key]) > 100:
                self.volatility_tracker[key] = self.volatility_tracker[key][-100:]
            
            # Check for significant price movements
            if abs(price_change_pct) > 2.0:  # 2% price change
                logger.info("Significant price movement detected", 
                          exchange=exchange, symbol=symbol, 
                          price_change_pct=price_change_pct)
        
        self.price_cache[key] = {
            "price": price,
            "timestamp": current_time
        }
    
    def get_volatility_score(self, exchange: str, symbol: str) -> float:
        """Calculate volatility score for a symbol on an exchange."""
        key = f"{exchange}:{symbol}"
        if key not in self.volatility_tracker or not self.volatility_tracker[key]:
            return 0.0
        
        # Calculate average volatility over recent periods
        recent_changes = [item["price_change_pct"] for item in self.volatility_tracker[key][-20:]]
        if not recent_changes:
            return 0.0
        
        return sum(recent_changes) / len(recent_changes)
    
    def cleanup_old_data(self):
        """Clean up old price data."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=1)
        
        # Clean price cache
        keys_to_remove = []
        for key, data in self.price_cache.items():
            if data["timestamp"] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.price_cache[key]
        
        # Clean volatility tracker
        for key in self.volatility_tracker:
            self.volatility_tracker[key] = [
                item for item in self.volatility_tracker[key] 
                if item["timestamp"] > cutoff_time
            ]

# Global price monitor
price_monitor = PriceMonitor()

# Enhanced arbitrage opportunity storage with persistence
class ArbitrageOpportunityStore:
    """Enhanced storage for arbitrage opportunities with persistence and analytics."""
    
    def __init__(self):
        self.opportunities = {
            "simple": [],
            "triangular": [],
            "funding": []
        }
        self.opportunity_history = []
        self.performance_stats = {
            "total_opportunities_detected": 0,
            "profitable_opportunities": 0,
            "average_profit_pct": 0.0,
            "best_opportunity_profit_pct": 0.0
        }
    
    def add_opportunities(self, opportunity_type: str, opportunities: List[Any]):
        """Add new opportunities and update statistics."""
        if opportunity_type not in self.opportunities:
            return
        
        # Store current opportunities
        self.opportunities[opportunity_type] = opportunities
        
        # Update statistics
        for opp in opportunities:
            self.performance_stats["total_opportunities_detected"] += 1
            
            # Get profit percentage based on opportunity type
            if hasattr(opp, 'profit_pct'):
                profit_pct = opp.profit_pct
            elif hasattr(opp, 'annualized_profit_pct'):
                profit_pct = opp.annualized_profit_pct
            else:
                continue
            
            if profit_pct > 0:
                self.performance_stats["profitable_opportunities"] += 1
                
                # Update average profit
                current_avg = self.performance_stats["average_profit_pct"]
                total_profitable = self.performance_stats["profitable_opportunities"]
                new_avg = ((current_avg * (total_profitable - 1)) + profit_pct) / total_profitable
                self.performance_stats["average_profit_pct"] = new_avg
                
                # Update best opportunity
                if profit_pct > self.performance_stats["best_opportunity_profit_pct"]:
                    self.performance_stats["best_opportunity_profit_pct"] = profit_pct
            
            # Add to history
            self.opportunity_history.append({
                "type": opportunity_type,
                "opportunity": opp,
                "detected_at": datetime.utcnow()
            })
        
        # Keep only last 1000 historical opportunities
        if len(self.opportunity_history) > 1000:
            self.opportunity_history = self.opportunity_history[-1000:]
    
    def get_opportunities(self, opportunity_type: str, filters: Dict[str, Any] = None) -> List[Any]:
        """Get opportunities with optional filtering."""
        if opportunity_type not in self.opportunities:
            return []
        
        opportunities = self.opportunities[opportunity_type]
        
        if not filters:
            return opportunities
        
        # Apply filters
        filtered_opportunities = []
        for opp in opportunities:
            include = True
            
            # Filter by symbol
            if "symbol" in filters and hasattr(opp, "symbol"):
                if opp.symbol != filters["symbol"]:
                    include = False
            
            # Filter by minimum profit
            if "min_profit_pct" in filters:
                profit_pct = getattr(opp, "profit_pct", getattr(opp, "annualized_profit_pct", 0))
                if profit_pct < filters["min_profit_pct"]:
                    include = False
            
            # Filter by maximum risk
            if "max_risk_score" in filters and hasattr(opp, "risk_score"):
                if opp.risk_score > filters["max_risk_score"]:
                    include = False
            
            # Filter by expiry
            if hasattr(opp, "expires_at"):
                if opp.expires_at <= datetime.utcnow():
                    include = False
            
            if include:
                filtered_opportunities.append(opp)
        
        return filtered_opportunities
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

# Global opportunity store
opportunity_store = ArbitrageOpportunityStore()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service_info": {
            "version": "1.0.0",
            "active_opportunities": {
                "simple": len(arbitrage_opportunities["simple"]),
                "triangular": len(arbitrage_opportunities["triangular"]),
                "funding": len(arbitrage_opportunities["funding"])
            },
            "scanning_status": {
                "simple": "simple_arbitrage" in scanning_tasks and not scanning_tasks["simple_arbitrage"].done(),
                "triangular": "triangular_arbitrage" in scanning_tasks and not scanning_tasks["triangular_arbitrage"].done(),
                "funding": "funding_arbitrage" in scanning_tasks and not scanning_tasks["funding_arbitrage"].done()
            }
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(generate_latest(), media_type="text/plain")

# Arbitrage API endpoints
@app.get("/api/opportunities/simple")
async def get_simple_arbitrage_opportunities(
    symbol: Optional[str] = None,
    min_profit_pct: Optional[float] = None,
    max_risk_score: Optional[float] = None
):
    """Get simple arbitrage opportunities."""
    filters = {}
    if symbol:
        filters["symbol"] = symbol
    if min_profit_pct is not None:
        filters["min_profit_pct"] = min_profit_pct
    if max_risk_score is not None:
        filters["max_risk_score"] = max_risk_score
    
    opportunities = opportunity_store.get_opportunities("simple", filters)
    
    return {
        "opportunities": opportunities,
        "count": len(opportunities),
        "timestamp": datetime.utcnow().isoformat(),
        "performance_stats": opportunity_store.get_performance_stats()
    }

@app.get("/api/opportunities/triangular")
async def get_triangular_arbitrage_opportunities(
    exchange: Optional[str] = None,
    base_currency: Optional[str] = None,
    min_profit_pct: Optional[float] = None,
    max_risk_score: Optional[float] = None
):
    """Get triangular arbitrage opportunities."""
    opportunities = arbitrage_opportunities["triangular"]
    
    # Apply filters
    if exchange:
        opportunities = [opp for opp in opportunities if opp.exchange == exchange]
    
    if base_currency:
        opportunities = [opp for opp in opportunities if opp.base_currency == base_currency]
    
    if min_profit_pct is not None:
        opportunities = [opp for opp in opportunities if opp.profit_pct >= min_profit_pct]
    
    if max_risk_score is not None:
        opportunities = [opp for opp in opportunities if opp.risk_score <= max_risk_score]
    
    # Remove expired opportunities
    current_time = datetime.utcnow()
    opportunities = [opp for opp in opportunities if opp.expires_at > current_time]
    
    return {
        "opportunities": opportunities,
        "count": len(opportunities),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/opportunities/funding")
async def get_funding_arbitrage_opportunities(
    symbol: Optional[str] = None,
    min_profit_pct: Optional[float] = None,
    max_risk_score: Optional[float] = None
):
    """Get funding rate arbitrage opportunities."""
    opportunities = arbitrage_opportunities["funding"]
    
    # Apply filters
    if symbol:
        opportunities = [opp for opp in opportunities if opp.symbol == symbol]
    
    if min_profit_pct is not None:
        opportunities = [opp for opp in opportunities if opp.annualized_profit_pct >= min_profit_pct]
    
    if max_risk_score is not None:
        opportunities = [opp for opp in opportunities if opp.risk_score <= max_risk_score]
    
    # Remove expired opportunities
    current_time = datetime.utcnow()
    opportunities = [opp for opp in opportunities if opp.expires_at > current_time]
    
    return {
        "opportunities": opportunities,
        "count": len(opportunities),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/opportunities/all")
async def get_all_arbitrage_opportunities():
    """Get all arbitrage opportunities."""
    current_time = datetime.utcnow()
    
    # Filter expired opportunities
    simple_opps = [opp for opp in arbitrage_opportunities["simple"] if opp.expires_at > current_time]
    triangular_opps = [opp for opp in arbitrage_opportunities["triangular"] if opp.expires_at > current_time]
    funding_opps = [opp for opp in arbitrage_opportunities["funding"] if opp.expires_at > current_time]
    
    return {
        "simple_arbitrage": {
            "opportunities": simple_opps,
            "count": len(simple_opps)
        },
        "triangular_arbitrage": {
            "opportunities": triangular_opps,
            "count": len(triangular_opps)
        },
        "funding_arbitrage": {
            "opportunities": funding_opps,
            "count": len(funding_opps)
        },
        "total_opportunities": len(simple_opps) + len(triangular_opps) + len(funding_opps),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/scan/simple")
async def trigger_simple_arbitrage_scan():
    """Manually trigger simple arbitrage scan."""
    try:
        result = await arbitrage_detector.scan_simple_arbitrage()
        return result
    except Exception as e:
        logger.error("Manual simple arbitrage scan failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.post("/api/scan/triangular")
async def trigger_triangular_arbitrage_scan():
    """Manually trigger triangular arbitrage scan."""
    try:
        result = await arbitrage_detector.scan_triangular_arbitrage()
        return result
    except Exception as e:
        logger.error("Manual triangular arbitrage scan failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.post("/api/scan/funding")
async def trigger_funding_arbitrage_scan():
    """Manually trigger funding arbitrage scan."""
    try:
        result = await arbitrage_detector.scan_funding_arbitrage()
        return result
    except Exception as e:
        logger.error("Manual funding arbitrage scan failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.post("/api/scan/all")
async def trigger_all_arbitrage_scans():
    """Manually trigger all arbitrage scans."""
    try:
        results = {}
        
        # Run all scans concurrently
        simple_task = arbitrage_detector.scan_simple_arbitrage()
        triangular_task = arbitrage_detector.scan_triangular_arbitrage()
        funding_task = arbitrage_detector.scan_funding_arbitrage()
        
        simple_result, triangular_result, funding_result = await asyncio.gather(
            simple_task, triangular_task, funding_task, return_exceptions=True
        )
        
        results["simple"] = simple_result if not isinstance(simple_result, Exception) else {"error": str(simple_result)}
        results["triangular"] = triangular_result if not isinstance(triangular_result, Exception) else {"error": str(triangular_result)}
        results["funding"] = funding_result if not isinstance(funding_result, Exception) else {"error": str(funding_result)}
        
        return {
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Manual arbitrage scan failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.get("/api/analytics/performance")
async def get_arbitrage_performance():
    """Get arbitrage detection performance analytics."""
    return {
        "performance_stats": opportunity_store.get_performance_stats(),
        "active_opportunities": {
            "simple": len(opportunity_store.get_opportunities("simple")),
            "triangular": len(opportunity_store.get_opportunities("triangular")),
            "funding": len(opportunity_store.get_opportunities("funding"))
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/analytics/volatility")
async def get_market_volatility():
    """Get market volatility analytics."""
    volatility_data = {}
    
    for symbol in config.supported_symbols:
        symbol_volatility = {}
        # Get available exchanges
        try:
            exchanges_url = f"{config.exchange_gateway_url}/api/exchanges"
            async with arbitrage_detector.exchange_client.get(exchanges_url) as response:
                if response.status == 200:
                    exchanges_data = await response.json()
                    available_exchanges = list(exchanges_data.get("exchanges", {}).keys())
                    
                    for exchange in available_exchanges:
                        volatility_score = price_monitor.get_volatility_score(exchange, symbol)
                        if volatility_score > 0:
                            symbol_volatility[exchange] = volatility_score
        except Exception as e:
            logger.warning("Failed to get volatility data", symbol=symbol, error=str(e))
        
        if symbol_volatility:
            volatility_data[symbol] = symbol_volatility
    
    return {
        "volatility_data": volatility_data,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/analytics/market_efficiency")
async def get_market_efficiency():
    """Get market efficiency metrics based on arbitrage opportunities."""
    simple_opps = opportunity_store.get_opportunities("simple")
    triangular_opps = opportunity_store.get_opportunities("triangular")
    funding_opps = opportunity_store.get_opportunities("funding")
    
    # Calculate market efficiency metrics
    total_opportunities = len(simple_opps) + len(triangular_opps) + len(funding_opps)
    
    # Average profit margins (lower = more efficient market)
    avg_simple_profit = sum(opp.profit_pct for opp in simple_opps) / len(simple_opps) if simple_opps else 0
    avg_triangular_profit = sum(opp.profit_pct for opp in triangular_opps) / len(triangular_opps) if triangular_opps else 0
    avg_funding_profit = sum(opp.annualized_profit_pct for opp in funding_opps) / len(funding_opps) if funding_opps else 0
    
    # Market efficiency score (0-100, higher = more efficient)
    efficiency_score = max(0, 100 - (avg_simple_profit * 10 + avg_triangular_profit * 15 + avg_funding_profit * 5))
    
    return {
        "market_efficiency": {
            "efficiency_score": efficiency_score,
            "total_opportunities": total_opportunities,
            "average_profits": {
                "simple_arbitrage": avg_simple_profit,
                "triangular_arbitrage": avg_triangular_profit,
                "funding_arbitrage": avg_funding_profit
            },
            "opportunity_distribution": {
                "simple": len(simple_opps),
                "triangular": len(triangular_opps),
                "funding": len(funding_opps)
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Background scanning tasks
async def simple_arbitrage_scanner():
    """Background task for simple arbitrage scanning."""
    while True:
        try:
            await arbitrage_detector.scan_simple_arbitrage()
            await asyncio.sleep(config.simple_arbitrage_scan_interval)
        except Exception as e:
            logger.error("Simple arbitrage scanner error", error=str(e))
            await asyncio.sleep(config.simple_arbitrage_scan_interval)

async def triangular_arbitrage_scanner():
    """Background task for triangular arbitrage scanning."""
    while True:
        try:
            await arbitrage_detector.scan_triangular_arbitrage()
            await asyncio.sleep(config.triangular_arbitrage_scan_interval)
        except Exception as e:
            logger.error("Triangular arbitrage scanner error", error=str(e))
            await asyncio.sleep(config.triangular_arbitrage_scan_interval)

async def funding_arbitrage_scanner():
    """Background task for funding arbitrage scanning."""
    while True:
        try:
            await arbitrage_detector.scan_funding_arbitrage()
            await asyncio.sleep(config.funding_arbitrage_scan_interval)
        except Exception as e:
            logger.error("Funding arbitrage scanner error", error=str(e))
            await asyncio.sleep(config.funding_arbitrage_scan_interval)

async def price_monitor_cleanup():
    """Background task for price monitor cleanup."""
    while True:
        try:
            price_monitor.cleanup_old_data()
            await asyncio.sleep(300)  # Clean up every 5 minutes
        except Exception as e:
            logger.error("Price monitor cleanup error", error=str(e))
            await asyncio.sleep(300)

async def market_data_collector():
    """Background task for collecting market data for price monitoring."""
    while True:
        try:
            # Get available exchanges
            exchanges_url = f"{config.exchange_gateway_url}/api/exchanges"
            async with arbitrage_detector.exchange_client.get(exchanges_url) as response:
                if response.status == 200:
                    exchanges_data = await response.json()
                    available_exchanges = list(exchanges_data.get("exchanges", {}).keys())
                    
                    # Collect price data for monitoring
                    for exchange in available_exchanges:
                        for symbol in config.supported_symbols:
                            try:
                                data = await arbitrage_detector.get_exchange_data(exchange, symbol)
                                if data and "ticker" in data:
                                    price = data["ticker"].get("last")
                                    if price:
                                        price_monitor.update_price(exchange, symbol, price)
                            except Exception as e:
                                logger.warning("Failed to collect price data", 
                                             exchange=exchange, symbol=symbol, error=str(e))
            
            await asyncio.sleep(10)  # Collect data every 10 seconds
        except Exception as e:
            logger.error("Market data collector error", error=str(e))
            await asyncio.sleep(10)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Arbitrage Detection Engine starting up...")
    
    # Initialize database connections
    from database import initialize_databases
    await initialize_databases()
    
    # Initialize arbitrage detector
    await arbitrage_detector.initialize()
    
    # Start background scanning tasks
    scanning_tasks["simple_arbitrage"] = asyncio.create_task(simple_arbitrage_scanner())
    scanning_tasks["triangular_arbitrage"] = asyncio.create_task(triangular_arbitrage_scanner())
    scanning_tasks["funding_arbitrage"] = asyncio.create_task(funding_arbitrage_scanner())
    scanning_tasks["price_monitor_cleanup"] = asyncio.create_task(price_monitor_cleanup())
    scanning_tasks["market_data_collector"] = asyncio.create_task(market_data_collector())
    
    logger.info("Arbitrage Detection Engine started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Arbitrage Detection Engine shutting down...")
    
    # Cancel background tasks
    for task_name, task in scanning_tasks.items():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Cancelled {task_name} task")
    
    # Cleanup arbitrage detector
    await arbitrage_detector.cleanup()
    
    # Close database connections
    from database import cleanup_databases
    await cleanup_databases()
    
    logger.info("Arbitrage Detection Engine shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info"
    )