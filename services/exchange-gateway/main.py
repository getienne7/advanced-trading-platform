"""
Exchange Gateway Service for Advanced Trading Platform.
Provides unified interface to multiple cryptocurrency exchanges.
"""
import asyncio
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from database import get_postgres_session, get_redis
from utils import setup_logging

# Import exchange connectors
from exchange_abstraction import ExchangeManager, OrderSide, OrderType, OrderStatus
from exchanges import BinanceConnector, CoinbaseConnector, KrakenConnector

# Configure logging
logger = setup_logging("exchange-gateway")

# Metrics
EXCHANGE_REQUESTS = Counter('exchange_requests_total', 'Total exchange requests', ['exchange', 'method'])
EXCHANGE_ERRORS = Counter('exchange_errors_total', 'Total exchange errors', ['exchange', 'error_type'])
EXCHANGE_LATENCY = Histogram('exchange_latency_seconds', 'Exchange request latency', ['exchange', 'method'])

# Configuration
class ExchangeConfig:
    """Exchange gateway configuration."""
    
    def __init__(self):
        # Binance configuration
        self.binance_api_key = os.getenv("BINANCE_API_KEY", "")
        self.binance_secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        self.binance_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        # Coinbase configuration
        self.coinbase_api_key = os.getenv("COINBASE_API_KEY", "")
        self.coinbase_secret_key = os.getenv("COINBASE_SECRET_KEY", "")
        self.coinbase_passphrase = os.getenv("COINBASE_PASSPHRASE", "")
        self.coinbase_sandbox = os.getenv("COINBASE_SANDBOX", "true").lower() == "true"
        
        # Kraken configuration
        self.kraken_api_key = os.getenv("KRAKEN_API_KEY", "")
        self.kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY", "")
        
        # General configuration
        self.max_retries = int(os.getenv("EXCHANGE_MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("EXCHANGE_REQUEST_TIMEOUT", "30"))
        self.rate_limit_buffer = float(os.getenv("RATE_LIMIT_BUFFER", "0.1"))

config = ExchangeConfig()

# Pydantic models
class OrderRequest(BaseModel):
    """Order request model."""
    exchange: str = Field(..., description="Exchange name")
    symbol: str = Field(..., description="Trading pair symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    type: str = Field(..., description="Order type (market/limit)")
    amount: float = Field(..., description="Order amount")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    time_in_force: Optional[str] = Field("GTC", description="Time in force")

class OrderResponse(BaseModel):
    """Order response model."""
    order_id: str
    exchange: str
    symbol: str
    side: str
    type: str
    amount: float
    price: Optional[float]
    status: str
    timestamp: datetime
    fees: Optional[Dict[str, float]] = None

class BalanceResponse(BaseModel):
    """Balance response model."""
    exchange: str
    balances: Dict[str, Dict[str, float]]  # {currency: {free: x, locked: y}}
    timestamp: datetime

class TickerResponse(BaseModel):
    """Ticker response model."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime

class OrderBookResponse(BaseModel):
    """Order book response model."""
    exchange: str
    symbol: str
    bids: List[List[float]]  # [[price, amount], ...]
    asks: List[List[float]]  # [[price, amount], ...]
    timestamp: datetime

# FastAPI app
app = FastAPI(
    title="Exchange Gateway Service",
    description="Unified interface to multiple cryptocurrency exchanges",
    version="1.0.0"
)

# Exchange manager
exchange_manager = ExchangeManager()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    exchange_status = {}
    overall_status = "healthy"
    
    for name, exchange in exchanges.items():
        try:
            status = await exchange.get_status()
            exchange_status[name] = status
            
            # Check if exchange is having issues
            if status.get('status') != 'online':
                overall_status = "degraded"
                
        except Exception as e:
            exchange_status[name] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "exchanges": exchange_status,
        "service_info": {
            "version": "1.0.0",
            "uptime": "calculated_uptime",
            "active_exchanges": len([e for e in exchange_status.values() if e.get('status') == 'online']),
            "total_exchanges": len(exchange_status)
        }
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with performance metrics."""
    exchange_details = {}
    
    for name, exchange in exchanges.items():
        try:
            start_time = datetime.utcnow()
            
            # Test basic connectivity
            status = await exchange.get_status()
            
            # Test market data
            try:
                ticker = await exchange.get_ticker("BTC/USDT")
                market_data_ok = True
                market_data_latency = (datetime.utcnow() - start_time).total_seconds()
            except:
                market_data_ok = False
                market_data_latency = None
            
            # Test account access (if credentials available)
            try:
                balances = await exchange.get_balances()
                account_access_ok = True
            except:
                account_access_ok = False
            
            exchange_details[name] = {
                "status": status,
                "market_data_ok": market_data_ok,
                "market_data_latency_ms": market_data_latency * 1000 if market_data_latency else None,
                "account_access_ok": account_access_ok,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            exchange_details[name] = {
                "status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "exchanges": exchange_details
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(generate_latest(), media_type="text/plain")

# Exchange Management Endpoints
@app.get("/api/exchanges")
async def list_exchanges():
    """List available exchanges."""
    exchange_info = {}
    
    for name, exchange in exchanges.items():
        try:
            info = await exchange.get_exchange_info()
            exchange_info[name] = info
        except Exception as e:
            exchange_info[name] = {"error": str(e)}
    
    return {
        "exchanges": exchange_info,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/exchanges/{exchange_name}/status")
async def get_exchange_status(exchange_name: str):
    """Get status of a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        status = await exchanges[exchange_name].get_status()
        return status
    except Exception as e:
        logger.error("Failed to get exchange status", exchange=exchange_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

# Market Data Endpoints
@app.get("/api/ticker/{exchange_name}/{symbol}", response_model=TickerResponse)
async def get_ticker(exchange_name: str, symbol: str):
    """Get ticker data for a symbol from a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="get_ticker").time():
            ticker = await exchanges[exchange_name].get_ticker(symbol)
            EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="get_ticker").inc()
            
            return TickerResponse(
                exchange=exchange_name,
                symbol=symbol,
                **ticker
            )
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="ticker_error").inc()
        logger.error("Failed to get ticker", exchange=exchange_name, symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get ticker: {str(e)}")

@app.get("/api/orderbook/{exchange_name}/{symbol}", response_model=OrderBookResponse)
async def get_order_book(exchange_name: str, symbol: str, limit: int = 100):
    """Get order book for a symbol from a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="get_order_book").time():
            order_book = await exchanges[exchange_name].get_order_book(symbol, limit)
            EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="get_order_book").inc()
            
            return OrderBookResponse(
                exchange=exchange_name,
                symbol=symbol,
                **order_book
            )
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="orderbook_error").inc()
        logger.error("Failed to get order book", exchange=exchange_name, symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get order book: {str(e)}")

@app.get("/api/tickers/all/{symbol}")
async def get_all_tickers(symbol: str):
    """Get ticker data for a symbol from all exchanges."""
    tickers = {}
    
    for exchange_name, exchange in exchanges.items():
        try:
            with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="get_ticker").time():
                ticker = await exchange.get_ticker(symbol)
                EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="get_ticker").inc()
                
                tickers[exchange_name] = {
                    "exchange": exchange_name,
                    "symbol": symbol,
                    **ticker
                }
        except Exception as e:
            EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="ticker_error").inc()
            logger.warning("Failed to get ticker from exchange", 
                         exchange=exchange_name, symbol=symbol, error=str(e))
            tickers[exchange_name] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "tickers": tickers,
        "timestamp": datetime.utcnow().isoformat()
    }

# Trading Endpoints
@app.get("/api/balances/{exchange_name}", response_model=BalanceResponse)
async def get_balances(exchange_name: str):
    """Get account balances from a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="get_balances").time():
            balances = await exchanges[exchange_name].get_balances()
            EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="get_balances").inc()
            
            return BalanceResponse(
                exchange=exchange_name,
                balances=balances,
                timestamp=datetime.utcnow()
            )
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="balance_error").inc()
        logger.error("Failed to get balances", exchange=exchange_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get balances: {str(e)}")

@app.post("/api/orders", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    """Place an order on a specific exchange."""
    if order.exchange not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {order.exchange} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=order.exchange, method="place_order").time():
            result = await exchanges[order.exchange].place_order(
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                amount=order.amount,
                price=order.price,
                time_in_force=order.time_in_force
            )
            EXCHANGE_REQUESTS.labels(exchange=order.exchange, method="place_order").inc()
            
            return OrderResponse(
                exchange=order.exchange,
                **result
            )
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=order.exchange, error_type="order_error").inc()
        logger.error("Failed to place order", exchange=order.exchange, order=order.dict(), error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to place order: {str(e)}")

@app.get("/api/orders/{exchange_name}/{order_id}")
async def get_order(exchange_name: str, order_id: str):
    """Get order status from a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="get_order").time():
            order = await exchanges[exchange_name].get_order(order_id)
            EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="get_order").inc()
            
            return {
                "exchange": exchange_name,
                **order
            }
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="order_error").inc()
        logger.error("Failed to get order", exchange=exchange_name, order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get order: {str(e)}")

@app.delete("/api/orders/{exchange_name}/{order_id}")
async def cancel_order(exchange_name: str, order_id: str):
    """Cancel an order on a specific exchange."""
    if exchange_name not in exchanges:
        raise HTTPException(status_code=404, detail=f"Exchange {exchange_name} not found")
    
    try:
        with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="cancel_order").time():
            result = await exchanges[exchange_name].cancel_order(order_id)
            EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="cancel_order").inc()
            
            return {
                "exchange": exchange_name,
                "order_id": order_id,
                "status": "cancelled",
                **result
            }
    except Exception as e:
        EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="cancel_error").inc()
        logger.error("Failed to cancel order", exchange=exchange_name, order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cancel order: {str(e)}")

# Smart Order Routing
@app.post("/api/smart-order")
async def place_smart_order(order: OrderRequest):
    """Place an order using smart routing to find best execution."""
    try:
        # Get best price across all exchanges
        best_exchange = await find_best_execution_exchange(order.symbol, order.side, order.amount)
        
        if not best_exchange:
            raise HTTPException(status_code=400, detail="No suitable exchange found for execution")
        
        # Place order on best exchange
        order.exchange = best_exchange['name']
        
        with EXCHANGE_LATENCY.labels(exchange=order.exchange, method="place_order").time():
            result = await exchanges[order.exchange].place_order(
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                amount=order.amount,
                price=order.price,
                time_in_force=order.time_in_force
            )
            EXCHANGE_REQUESTS.labels(exchange=order.exchange, method="place_order").inc()
            
            return OrderResponse(
                exchange=order.exchange,
                **result,
                routing_info={
                    'selected_exchange': best_exchange['name'],
                    'reason': best_exchange['reason'],
                    'expected_price': best_exchange['price'],
                    'liquidity_score': best_exchange['liquidity_score']
                }
            )
    except Exception as e:
        logger.error("Failed to place smart order", order=order.dict(), error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to place smart order: {str(e)}")

async def find_best_execution_exchange(symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
    """Find the best exchange for order execution."""
    candidates = []
    
    for exchange_name, exchange in exchanges.items():
        try:
            # Get ticker and order book
            ticker = await exchange.get_ticker(symbol)
            order_book = await exchange.get_order_book(symbol, limit=20)
            
            # Calculate execution price and liquidity
            if side.lower() == 'buy':
                price = ticker['ask']
                liquidity = sum(ask[1] for ask in order_book['asks'][:10])
            else:
                price = ticker['bid']
                liquidity = sum(bid[1] for bid in order_book['bids'][:10])
            
            # Calculate liquidity score
            liquidity_score = min(liquidity / amount, 10.0)  # Cap at 10x
            
            # Check if exchange can handle the order size
            if liquidity >= amount * 0.5:  # At least 50% liquidity available
                candidates.append({
                    'name': exchange_name,
                    'price': price,
                    'liquidity_score': liquidity_score,
                    'spread': ticker['ask'] - ticker['bid'],
                    'reason': f"Good liquidity ({liquidity:.2f}) and price ({price:.6f})"
                })
                
        except Exception as e:
            logger.warning("Failed to evaluate exchange for routing", 
                         exchange=exchange_name, symbol=symbol, error=str(e))
    
    if not candidates:
        return None
    
    # Select best exchange based on price and liquidity
    if side.lower() == 'buy':
        # For buy orders, prefer lower ask price and higher liquidity
        best = min(candidates, key=lambda x: (x['price'], -x['liquidity_score']))
    else:
        # For sell orders, prefer higher bid price and higher liquidity
        best = max(candidates, key=lambda x: (x['price'], x['liquidity_score']))
    
    return best

@app.post("/api/orders/failover")
async def place_order_with_failover(order: OrderRequest, backup_exchanges: List[str] = None):
    """Place an order with automatic failover to backup exchanges."""
    exchanges_to_try = [order.exchange]
    if backup_exchanges:
        exchanges_to_try.extend(backup_exchanges)
    else:
        # Use all available exchanges as backup
        exchanges_to_try.extend([name for name in exchanges.keys() if name != order.exchange])
    
    last_error = None
    
    for exchange_name in exchanges_to_try:
        if exchange_name not in exchanges:
            continue
            
        try:
            logger.info("Attempting order placement", exchange=exchange_name, attempt=exchanges_to_try.index(exchange_name) + 1)
            
            with EXCHANGE_LATENCY.labels(exchange=exchange_name, method="place_order").time():
                result = await exchanges[exchange_name].place_order(
                    symbol=order.symbol,
                    side=order.side,
                    type=order.type,
                    amount=order.amount,
                    price=order.price,
                    time_in_force=order.time_in_force
                )
                EXCHANGE_REQUESTS.labels(exchange=exchange_name, method="place_order").inc()
                
                return OrderResponse(
                    exchange=exchange_name,
                    **result,
                    failover_info={
                        'primary_exchange': order.exchange,
                        'executed_on': exchange_name,
                        'attempts': exchanges_to_try.index(exchange_name) + 1,
                        'failed_exchanges': exchanges_to_try[:exchanges_to_try.index(exchange_name)]
                    }
                )
        except Exception as e:
            EXCHANGE_ERRORS.labels(exchange=exchange_name, error_type="failover_error").inc()
            logger.warning("Order placement failed, trying next exchange", 
                         exchange=exchange_name, error=str(e))
            last_error = e
            continue
    
    # All exchanges failed
    logger.error("All exchanges failed for order placement", 
                order=order.dict(), last_error=str(last_error))
    raise HTTPException(status_code=500, 
                       detail=f"Order placement failed on all exchanges. Last error: {str(last_error)}")

# Arbitrage Endpoints
@app.get("/api/arbitrage/opportunities/{symbol}")
async def get_arbitrage_opportunities(symbol: str, min_profit_pct: float = 0.5):
    """Find arbitrage opportunities for a symbol across exchanges."""
    try:
        # Get tickers from all exchanges
        tickers = {}
        for exchange_name, exchange in exchanges.items():
            try:
                ticker = await exchange.get_ticker(symbol)
                tickers[exchange_name] = ticker
            except Exception as e:
                logger.warning("Failed to get ticker for arbitrage", 
                             exchange=exchange_name, symbol=symbol, error=str(e))
        
        if len(tickers) < 2:
            return {
                "symbol": symbol,
                "opportunities": [],
                "message": "Need at least 2 exchanges with data"
            }
        
        # Find arbitrage opportunities
        opportunities = []
        exchanges_list = list(tickers.keys())
        
        for i in range(len(exchanges_list)):
            for j in range(i + 1, len(exchanges_list)):
                exchange_a = exchanges_list[i]
                exchange_b = exchanges_list[j]
                
                ticker_a = tickers[exchange_a]
                ticker_b = tickers[exchange_b]
                
                # Check A -> B arbitrage (buy on A, sell on B)
                profit_pct_ab = ((ticker_b['bid'] - ticker_a['ask']) / ticker_a['ask']) * 100
                if profit_pct_ab >= min_profit_pct:
                    opportunities.append({
                        "buy_exchange": exchange_a,
                        "sell_exchange": exchange_b,
                        "buy_price": ticker_a['ask'],
                        "sell_price": ticker_b['bid'],
                        "profit_pct": round(profit_pct_ab, 4),
                        "profit_per_unit": round(ticker_b['bid'] - ticker_a['ask'], 8)
                    })
                
                # Check B -> A arbitrage (buy on B, sell on A)
                profit_pct_ba = ((ticker_a['bid'] - ticker_b['ask']) / ticker_b['ask']) * 100
                if profit_pct_ba >= min_profit_pct:
                    opportunities.append({
                        "buy_exchange": exchange_b,
                        "sell_exchange": exchange_a,
                        "buy_price": ticker_b['ask'],
                        "sell_price": ticker_a['bid'],
                        "profit_pct": round(profit_pct_ba, 4),
                        "profit_per_unit": round(ticker_a['bid'] - ticker_b['ask'], 8)
                    })
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        
        return {
            "symbol": symbol,
            "opportunities": opportunities,
            "tickers": tickers,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to find arbitrage opportunities", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to find opportunities: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Exchange Gateway service starting up...")
    
    # Initialize database connections
    from database import initialize_databases
    await initialize_databases()
    
    # Initialize exchange connectors
    await initialize_exchanges()
    
    logger.info("Exchange Gateway service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Exchange Gateway service shutting down...")
    
    # Cleanup exchange connections
    for exchange in exchanges.values():
        try:
            await exchange.close()
        except Exception as e:
            logger.warning("Error closing exchange connection", error=str(e))
    
    # Close database connections
    from database import cleanup_databases
    await cleanup_databases()
    
    logger.info("Exchange Gateway service shutdown complete")

async def initialize_exchanges():
    """Initialize exchange connectors."""
    global exchange_manager
    
    # Initialize Binance
    if config.binance_api_key and config.binance_secret_key:
        try:
            binance = BinanceConnector(
                api_key=config.binance_api_key,
                api_secret=config.binance_secret_key,
                sandbox=config.binance_testnet
            )
            exchange_manager.add_exchange("binance", binance)
            await exchange_manager.connect_exchange("binance")
            logger.info("Binance connector initialized", testnet=config.binance_testnet)
        except Exception as e:
            logger.error("Failed to initialize Binance connector", error=str(e))
    else:
        logger.warning("Binance API credentials not provided")
    
    # Initialize Coinbase
    if config.coinbase_api_key and config.coinbase_secret_key and config.coinbase_passphrase:
        try:
            coinbase = CoinbaseConnector(
                api_key=config.coinbase_api_key,
                api_secret=config.coinbase_secret_key,
                passphrase=config.coinbase_passphrase,
                sandbox=config.coinbase_sandbox
            )
            exchange_manager.add_exchange("coinbase", coinbase)
            await exchange_manager.connect_exchange("coinbase")
            logger.info("Coinbase connector initialized", sandbox=config.coinbase_sandbox)
        except Exception as e:
            logger.error("Failed to initialize Coinbase connector", error=str(e))
    else:
        logger.warning("Coinbase API credentials not provided")
    
    # Initialize Kraken
    if config.kraken_api_key and config.kraken_secret_key:
        try:
            kraken = KrakenConnector(
                api_key=config.kraken_api_key,
                api_secret=config.kraken_secret_key
            )
            exchange_manager.add_exchange("kraken", kraken)
            await exchange_manager.connect_exchange("kraken")
            logger.info("Kraken connector initialized")
        except Exception as e:
            logger.error("Failed to initialize Kraken connector", error=str(e))
    else:
        logger.warning("Kraken API credentials not provided")
    
    active_exchanges = exchange_manager.get_active_exchanges()
    if not active_exchanges:
        logger.warning("No exchange connectors initialized - service will have limited functionality")
    else:
        logger.info(f"Initialized {len(active_exchanges)} exchange connectors: {', '.join(active_exchanges)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info"
    )