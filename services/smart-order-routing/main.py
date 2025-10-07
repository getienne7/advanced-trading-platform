"""
Smart Order Routing Service for Advanced Trading Platform.
Implements best execution algorithm, latency optimization, and automatic failover.
"""
import asyncio
import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from database import get_postgres_session, get_redis
from utils import setup_logging
from models import OrderSide, OrderType, ExchangeName

# Configure logging
logger = setup_logging("smart-order-routing")

# Metrics
ROUTING_DECISIONS = Counter('routing_decisions_total', 'Total routing decisions', ['exchange', 'reason'])
EXECUTION_LATENCY = Histogram('execution_latency_seconds', 'Order execution latency', ['exchange'])
FAILOVER_EVENTS = Counter('failover_events_total', 'Total failover events', ['from_exchange', 'to_exchange'])
EXCHANGE_HEALTH_SCORE = Gauge('exchange_health_score', 'Exchange health score', ['exchange'])
BEST_EXECUTION_SAVINGS = Histogram('best_execution_savings_bps', 'Best execution savings in basis points')

class RoutingReason(Enum):
    """Reasons for routing decisions."""
    BEST_PRICE = "best_price"
    BEST_LIQUIDITY = "best_liquidity"
    LOWEST_LATENCY = "lowest_latency"
    FAILOVER = "failover"
    LOAD_BALANCING = "load_balancing"
    COST_OPTIMIZATION = "cost_optimization"

@dataclass
class ExchangeMetrics:
    """Real-time exchange performance metrics."""
    name: str
    latency_ms: float = 0.0
    success_rate: float = 1.0
    liquidity_score: float = 0.0
    spread_bps: float = 0.0
    uptime_pct: float = 100.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0
    is_healthy: bool = True
    
    def update_latency(self, latency_ms: float):
        """Update latency with exponential moving average."""
        alpha = 0.3  # Smoothing factor
        self.latency_ms = alpha * latency_ms + (1 - alpha) * self.latency_ms
        self.last_update = datetime.utcnow()
    
    def update_success_rate(self, success: bool):
        """Update success rate with exponential moving average."""
        alpha = 0.1
        new_rate = 1.0 if success else 0.0
        self.success_rate = alpha * new_rate + (1 - alpha) * self.success_rate
        
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Mark as unhealthy if too many consecutive failures
        self.is_healthy = self.consecutive_failures < 3 and self.success_rate > 0.8
        self.last_update = datetime.utcnow()

@dataclass
class RoutingDecision:
    """Smart routing decision result."""
    selected_exchange: str
    reason: RoutingReason
    expected_price: float
    expected_slippage_bps: float
    liquidity_score: float
    latency_ms: float
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class OrderRequest(BaseModel):
    """Order request for smart routing."""
    symbol: str = Field(..., description="Trading pair symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    type: str = Field(..., description="Order type (market/limit)")
    amount: float = Field(..., description="Order amount")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    time_in_force: Optional[str] = Field("GTC", description="Time in force")
    max_slippage_bps: Optional[float] = Field(50, description="Maximum acceptable slippage in basis points")
    preferred_exchanges: Optional[List[str]] = Field(None, description="Preferred exchanges in order")
    execution_strategy: Optional[str] = Field("best_execution", description="Execution strategy")

class SmartOrderRouter:
    """Smart order routing engine with best execution and failover."""
    
    def __init__(self):
        self.exchange_metrics: Dict[str, ExchangeMetrics] = {}
        self.exchange_clients = {}
        self.routing_cache = {}
        self.cache_ttl_seconds = 5  # Cache routing decisions for 5 seconds
        
        # Configuration
        self.max_latency_ms = 1000  # Maximum acceptable latency
        self.min_liquidity_score = 2.0  # Minimum liquidity score
        self.max_slippage_bps = 100  # Maximum acceptable slippage
        self.health_check_interval = 30  # Health check interval in seconds
        
        # Latency optimization
        self.connection_pools = {}
        self.request_queues = {}
        self.latency_thresholds = {
            'excellent': 50,   # < 50ms
            'good': 100,       # < 100ms  
            'acceptable': 200, # < 200ms
            'poor': 500        # < 500ms
        }
        
    async def initialize(self, exchange_clients: Dict[str, Any]):
        """Initialize the smart order router."""
        self.exchange_clients = exchange_clients
        
        # Initialize metrics for each exchange
        for exchange_name in exchange_clients.keys():
            self.exchange_metrics[exchange_name] = ExchangeMetrics(name=exchange_name)
            self.request_queues[exchange_name] = asyncio.Queue(maxsize=1000)
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._latency_monitor())
        
        logger.info("Smart order router initialized", exchanges=list(exchange_clients.keys()))
    
    async def route_order(self, order: OrderRequest) -> RoutingDecision:
        """Route order to best exchange using smart execution algorithm."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{order.symbol}_{order.side}_{order.amount}_{order.execution_strategy}"
            cached_decision = self._get_cached_decision(cache_key)
            if cached_decision:
                logger.debug("Using cached routing decision", cache_key=cache_key)
                return cached_decision
            
            # Get market data from all healthy exchanges
            market_data = await self._gather_market_data(order.symbol)
            
            if not market_data:
                raise HTTPException(status_code=503, detail="No healthy exchanges available")
            
            # Apply execution strategy
            decision = await self._apply_execution_strategy(order, market_data)
            
            # Cache the decision
            self._cache_decision(cache_key, decision)
            
            # Record metrics
            ROUTING_DECISIONS.labels(
                exchange=decision.selected_exchange,
                reason=decision.reason.value
            ).inc()
            
            execution_time_ms = (time.time() - start_time) * 1000
            logger.info("Order routed successfully",
                       exchange=decision.selected_exchange,
                       reason=decision.reason.value,
                       execution_time_ms=execution_time_ms,
                       expected_price=decision.expected_price)
            
            return decision
            
        except Exception as e:
            logger.error("Failed to route order", error=str(e), order=order.dict())
            raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")
    
    async def _apply_execution_strategy(self, order: OrderRequest, market_data: Dict[str, Any]) -> RoutingDecision:
        """Apply the selected execution strategy."""
        strategy = order.execution_strategy or "best_execution"
        
        if strategy == "best_execution":
            return await self._best_execution_strategy(order, market_data)
        elif strategy == "lowest_latency":
            return await self._lowest_latency_strategy(order, market_data)
        elif strategy == "best_liquidity":
            return await self._best_liquidity_strategy(order, market_data)
        elif strategy == "cost_optimization":
            return await self._cost_optimization_strategy(order, market_data)
        else:
            # Default to best execution
            return await self._best_execution_strategy(order, market_data)
    
    async def _apply_preferred_exchanges(self, order: OrderRequest, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply preferred exchange filtering if specified."""
        if not order.preferred_exchanges:
            return candidates
        
        # Filter candidates to preferred exchanges first
        preferred_candidates = [c for c in candidates if c['exchange'] in order.preferred_exchanges]
        
        if preferred_candidates:
            # Sort by preference order
            preference_order = {exchange: i for i, exchange in enumerate(order.preferred_exchanges)}
            preferred_candidates.sort(key=lambda x: preference_order.get(x['exchange'], 999))
            
            # Add non-preferred as backup
            non_preferred = [c for c in candidates if c['exchange'] not in order.preferred_exchanges]
            return preferred_candidates + non_preferred
        
        return candidates

    async def _best_execution_strategy(self, order: OrderRequest, market_data: Dict[str, Any]) -> RoutingDecision:
        """Best execution strategy considering price, liquidity, and costs."""
        candidates = []
        
        for exchange_name, data in market_data.items():
            metrics = self.exchange_metrics[exchange_name]
            
            if not metrics.is_healthy:
                continue
            
            # Calculate execution quality score
            ticker = data['ticker']
            order_book = data['order_book']
            
            # Get execution cost analysis
            execution_cost = self._calculate_execution_cost(order_book, order.side, order.amount)
            
            if execution_cost['fillable_amount'] < order.amount * 0.8:  # Need at least 80% fillable
                continue
            
            # Calculate composite score
            price_score = self._calculate_price_score(ticker, order.side)
            liquidity_score = execution_cost['liquidity_score']
            latency_score = max(0, 100 - metrics.latency_ms) / 100  # Normalize latency
            reliability_score = metrics.success_rate
            
            # Weighted composite score
            composite_score = (
                price_score * 0.4 +
                liquidity_score * 0.3 +
                latency_score * 0.2 +
                reliability_score * 0.1
            )
            
            candidates.append({
                'exchange': exchange_name,
                'score': composite_score,
                'price': execution_cost['average_price'],
                'slippage_bps': execution_cost['slippage_bps'],
                'liquidity_score': liquidity_score,
                'latency_ms': metrics.latency_ms,
                'execution_cost': execution_cost
            })
        
        if not candidates:
            raise Exception("No suitable exchanges found for execution")
        
        # Apply preferred exchange filtering
        candidates = await self._apply_preferred_exchanges(order, candidates)
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x['score'])
        
        # Check slippage limit
        if best_candidate['slippage_bps'] > order.max_slippage_bps:
            # Try to find alternative with acceptable slippage
            acceptable_candidates = [c for c in candidates if c['slippage_bps'] <= order.max_slippage_bps]
            if acceptable_candidates:
                best_candidate = max(acceptable_candidates, key=lambda x: x['score'])
            else:
                logger.warning("All exchanges exceed slippage limit",
                             max_slippage=order.max_slippage_bps,
                             best_slippage=best_candidate['slippage_bps'])
        
        return RoutingDecision(
            selected_exchange=best_candidate['exchange'],
            reason=RoutingReason.BEST_PRICE,
            expected_price=best_candidate['price'],
            expected_slippage_bps=best_candidate['slippage_bps'],
            liquidity_score=best_candidate['liquidity_score'],
            latency_ms=best_candidate['latency_ms'],
            confidence=best_candidate['score'],
            alternatives=[c for c in candidates if c['exchange'] != best_candidate['exchange']],
            metadata={'execution_cost': best_candidate['execution_cost']}
        )
    
    async def _optimize_for_hft(self, order: OrderRequest, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize routing for high-frequency trading requirements."""
        # Filter for ultra-low latency exchanges only
        hft_candidates = []
        
        for candidate in candidates:
            exchange_name = candidate['exchange']
            metrics = self.exchange_metrics[exchange_name]
            
            # HFT requirements: < 100ms latency, > 95% success rate
            if (metrics.latency_ms < 100 and 
                metrics.success_rate > 0.95 and 
                metrics.consecutive_failures == 0):
                
                # Add HFT score based on latency and reliability
                hft_score = (1.0 - metrics.latency_ms / 100) * metrics.success_rate
                candidate['hft_score'] = hft_score
                hft_candidates.append(candidate)
        
        # Sort by HFT score
        hft_candidates.sort(key=lambda x: x.get('hft_score', 0), reverse=True)
        
        return hft_candidates if hft_candidates else candidates

    async def _lowest_latency_strategy(self, order: OrderRequest, market_data: Dict[str, Any]) -> RoutingDecision:
        """Strategy optimized for lowest latency execution."""
        candidates = []
        
        for exchange_name, data in market_data.items():
            metrics = self.exchange_metrics[exchange_name]
            
            if not metrics.is_healthy or metrics.latency_ms > self.max_latency_ms:
                continue
            
            ticker = data['ticker']
            order_book = data['order_book']
            execution_cost = self._calculate_execution_cost(order_book, order.side, order.amount)
            
            if execution_cost['fillable_amount'] >= order.amount * 0.9:  # Need 90% fillable for latency strategy
                candidates.append({
                    'exchange': exchange_name,
                    'latency_ms': metrics.latency_ms,
                    'price': execution_cost['average_price'],
                    'slippage_bps': execution_cost['slippage_bps'],
                    'liquidity_score': execution_cost['liquidity_score']
                })
        
        if not candidates:
            # Fallback to best execution if no low-latency options
            return await self._best_execution_strategy(order, market_data)
        
        # Apply HFT optimizations if order amount suggests high-frequency trading
        if order.amount < 1.0:  # Small orders likely HFT
            candidates = await self._optimize_for_hft(order, candidates)
        
        # Apply preferred exchange filtering
        candidates = await self._apply_preferred_exchanges(order, candidates)
        
        # Select lowest latency
        best_candidate = min(candidates, key=lambda x: x['latency_ms'])
        
        return RoutingDecision(
            selected_exchange=best_candidate['exchange'],
            reason=RoutingReason.LOWEST_LATENCY,
            expected_price=best_candidate['price'],
            expected_slippage_bps=best_candidate['slippage_bps'],
            liquidity_score=best_candidate['liquidity_score'],
            latency_ms=best_candidate['latency_ms'],
            confidence=0.9,  # High confidence for latency-optimized routing
            alternatives=[c for c in candidates if c['exchange'] != best_candidate['exchange']]
        )
    
    async def _best_liquidity_strategy(self, order: OrderRequest, market_data: Dict[str, Any]) -> RoutingDecision:
        """Strategy optimized for best liquidity and minimal market impact."""
        candidates = []
        
        for exchange_name, data in market_data.items():
            metrics = self.exchange_metrics[exchange_name]
            
            if not metrics.is_healthy:
                continue
            
            order_book = data['order_book']
            execution_cost = self._calculate_execution_cost(order_book, order.side, order.amount)
            
            candidates.append({
                'exchange': exchange_name,
                'liquidity_score': execution_cost['liquidity_score'],
                'price': execution_cost['average_price'],
                'slippage_bps': execution_cost['slippage_bps'],
                'latency_ms': metrics.latency_ms,
                'fillable_pct': execution_cost['fillable_amount'] / order.amount * 100
            })
        
        if not candidates:
            raise Exception("No exchanges available for liquidity strategy")
        
        # Select best liquidity
        best_candidate = max(candidates, key=lambda x: (x['liquidity_score'], x['fillable_pct']))
        
        return RoutingDecision(
            selected_exchange=best_candidate['exchange'],
            reason=RoutingReason.BEST_LIQUIDITY,
            expected_price=best_candidate['price'],
            expected_slippage_bps=best_candidate['slippage_bps'],
            liquidity_score=best_candidate['liquidity_score'],
            latency_ms=best_candidate['latency_ms'],
            confidence=0.85,
            alternatives=[c for c in candidates if c['exchange'] != best_candidate['exchange']]
        )
    
    async def _cost_optimization_strategy(self, order: OrderRequest, market_data: Dict[str, Any]) -> RoutingDecision:
        """Strategy optimized for total cost including fees and slippage."""
        candidates = []
        
        for exchange_name, data in market_data.items():
            metrics = self.exchange_metrics[exchange_name]
            
            if not metrics.is_healthy:
                continue
            
            ticker = data['ticker']
            order_book = data['order_book']
            execution_cost = self._calculate_execution_cost(order_book, order.side, order.amount)
            
            # Estimate total cost including fees
            base_price = ticker['bid'] if order.side.lower() == 'sell' else ticker['ask']
            slippage_cost = execution_cost['slippage_bps'] / 10000 * base_price * order.amount
            
            # Assume 0.1% trading fee (should be configurable per exchange)
            trading_fee = 0.001 * base_price * order.amount
            
            total_cost = slippage_cost + trading_fee
            
            candidates.append({
                'exchange': exchange_name,
                'total_cost': total_cost,
                'price': execution_cost['average_price'],
                'slippage_bps': execution_cost['slippage_bps'],
                'liquidity_score': execution_cost['liquidity_score'],
                'latency_ms': metrics.latency_ms
            })
        
        if not candidates:
            raise Exception("No exchanges available for cost optimization")
        
        # Select lowest total cost
        best_candidate = min(candidates, key=lambda x: x['total_cost'])
        
        return RoutingDecision(
            selected_exchange=best_candidate['exchange'],
            reason=RoutingReason.COST_OPTIMIZATION,
            expected_price=best_candidate['price'],
            expected_slippage_bps=best_candidate['slippage_bps'],
            liquidity_score=best_candidate['liquidity_score'],
            latency_ms=best_candidate['latency_ms'],
            confidence=0.8,
            alternatives=[c for c in candidates if c['exchange'] != best_candidate['exchange']],
            metadata={'total_cost': best_candidate['total_cost']}
        )
    
    async def _emergency_failover(self, order: OrderRequest) -> RoutingDecision:
        """Emergency failover when all preferred exchanges fail."""
        logger.warning("Initiating emergency failover", symbol=order.symbol)
        
        # Find any healthy exchange
        healthy_exchanges = [
            name for name, metrics in self.exchange_metrics.items() 
            if metrics.is_healthy and metrics.consecutive_failures < 5
        ]
        
        if not healthy_exchanges:
            raise Exception("No healthy exchanges available for emergency failover")
        
        # Select exchange with best recent performance
        best_exchange = min(healthy_exchanges, 
                          key=lambda x: self.exchange_metrics[x].latency_ms)
        
        return RoutingDecision(
            selected_exchange=best_exchange,
            reason=RoutingReason.FAILOVER,
            expected_price=0.0,  # Will be determined at execution
            expected_slippage_bps=0.0,
            liquidity_score=self.exchange_metrics[best_exchange].liquidity_score,
            latency_ms=self.exchange_metrics[best_exchange].latency_ms,
            confidence=0.6,  # Lower confidence for emergency failover
            alternatives=[],
            metadata={'emergency_failover': True}
        )

    async def execute_with_failover(self, order: OrderRequest, routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute order with automatic failover to backup exchanges."""
        primary_exchange = routing_decision.selected_exchange
        backup_exchanges = [alt['exchange'] for alt in routing_decision.alternatives]
        
        # Try primary exchange first
        try:
            result = await self._execute_order_on_exchange(primary_exchange, order)
            
            # Update success metrics
            self.exchange_metrics[primary_exchange].update_success_rate(True)
            
            return {
                'success': True,
                'exchange': primary_exchange,
                'result': result,
                'failover_used': False
            }
            
        except Exception as primary_error:
            logger.warning("Primary exchange failed, attempting failover",
                         primary_exchange=primary_exchange,
                         error=str(primary_error))
            
            # Update failure metrics
            self.exchange_metrics[primary_exchange].update_success_rate(False)
            FAILOVER_EVENTS.labels(from_exchange=primary_exchange, to_exchange="attempting").inc()
            
            # Try backup exchanges
            for backup_exchange in backup_exchanges:
                try:
                    # Re-route to backup exchange
                    backup_decision = await self._get_backup_routing(order, backup_exchange)
                    result = await self._execute_order_on_exchange(backup_exchange, order)
                    
                    # Update success metrics
                    self.exchange_metrics[backup_exchange].update_success_rate(True)
                    FAILOVER_EVENTS.labels(from_exchange=primary_exchange, to_exchange=backup_exchange).inc()
                    
                    logger.info("Failover successful",
                              primary_exchange=primary_exchange,
                              backup_exchange=backup_exchange)
                    
                    return {
                        'success': True,
                        'exchange': backup_exchange,
                        'result': result,
                        'failover_used': True,
                        'primary_exchange': primary_exchange,
                        'primary_error': str(primary_error)
                    }
                    
                except Exception as backup_error:
                    logger.warning("Backup exchange failed",
                                 backup_exchange=backup_exchange,
                                 error=str(backup_error))
                    
                    # Update failure metrics
                    self.exchange_metrics[backup_exchange].update_success_rate(False)
                    continue
            
            # Try emergency failover as last resort
            try:
                logger.warning("All backup exchanges failed, attempting emergency failover")
                emergency_decision = await self._emergency_failover(order)
                result = await self._execute_order_on_exchange(emergency_decision.selected_exchange, order)
                
                self.exchange_metrics[emergency_decision.selected_exchange].update_success_rate(True)
                FAILOVER_EVENTS.labels(from_exchange=primary_exchange, to_exchange=emergency_decision.selected_exchange).inc()
                
                logger.info("Emergency failover successful",
                          primary_exchange=primary_exchange,
                          emergency_exchange=emergency_decision.selected_exchange)
                
                return {
                    'success': True,
                    'exchange': emergency_decision.selected_exchange,
                    'result': result,
                    'failover_used': True,
                    'emergency_failover': True,
                    'primary_exchange': primary_exchange,
                    'primary_error': str(primary_error)
                }
                
            except Exception as emergency_error:
                logger.error("Emergency failover also failed", error=str(emergency_error))
                # All exchanges failed
                raise Exception(f"All exchanges failed including emergency failover. Primary error: {str(primary_error)}, Emergency error: {str(emergency_error)}")
    
    async def _execute_order_on_exchange(self, exchange_name: str, order: OrderRequest) -> Dict[str, Any]:
        """Execute order on specific exchange with latency tracking."""
        start_time = time.time()
        
        try:
            exchange_client = self.exchange_clients[exchange_name]
            
            # Execute the order
            result = await exchange_client.place_order(
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                amount=order.amount,
                price=order.price,
                time_in_force=order.time_in_force
            )
            
            # Track execution latency
            execution_time_ms = (time.time() - start_time) * 1000
            self.exchange_metrics[exchange_name].update_latency(execution_time_ms)
            EXECUTION_LATENCY.labels(exchange=exchange_name).observe(execution_time_ms / 1000)
            
            return result
            
        except Exception as e:
            # Track failed execution latency
            execution_time_ms = (time.time() - start_time) * 1000
            self.exchange_metrics[exchange_name].update_latency(execution_time_ms)
            raise e
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """Gather market data from all healthy exchanges concurrently."""
        tasks = []
        
        for exchange_name, client in self.exchange_clients.items():
            if self.exchange_metrics[exchange_name].is_healthy:
                task = asyncio.create_task(self._get_exchange_market_data(exchange_name, client, symbol))
                tasks.append((exchange_name, task))
        
        market_data = {}
        
        # Wait for all tasks with timeout
        for exchange_name, task in tasks:
            try:
                data = await asyncio.wait_for(task, timeout=2.0)  # 2 second timeout
                market_data[exchange_name] = data
            except asyncio.TimeoutError:
                logger.warning("Market data timeout", exchange=exchange_name)
                self.exchange_metrics[exchange_name].update_success_rate(False)
            except Exception as e:
                logger.warning("Failed to get market data", exchange=exchange_name, error=str(e))
                self.exchange_metrics[exchange_name].update_success_rate(False)
        
        return market_data
    
    async def _get_exchange_market_data(self, exchange_name: str, client: Any, symbol: str) -> Dict[str, Any]:
        """Get market data from a specific exchange."""
        start_time = time.time()
        
        try:
            # Get ticker and order book concurrently
            ticker_task = asyncio.create_task(client.get_ticker(symbol))
            orderbook_task = asyncio.create_task(client.get_order_book(symbol, limit=20))
            
            ticker, order_book = await asyncio.gather(ticker_task, orderbook_task)
            
            # Update latency metrics
            latency_ms = (time.time() - start_time) * 1000
            self.exchange_metrics[exchange_name].update_latency(latency_ms)
            self.exchange_metrics[exchange_name].update_success_rate(True)
            
            return {
                'ticker': ticker,
                'order_book': order_book,
                'latency_ms': latency_ms
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.exchange_metrics[exchange_name].update_latency(latency_ms)
            self.exchange_metrics[exchange_name].update_success_rate(False)
            raise e
    
    def _calculate_execution_cost(self, order_book: Dict[str, Any], side: str, amount: float) -> Dict[str, Any]:
        """Calculate execution cost and slippage for an order."""
        try:
            if side.lower() == 'buy':
                levels = order_book.get('asks', [])
            else:
                levels = order_book.get('bids', [])
            
            if not levels:
                return {
                    'average_price': 0,
                    'best_price': 0,
                    'slippage_bps': 10000,  # 100% slippage (no liquidity)
                    'fillable_amount': 0,
                    'liquidity_score': 0
                }
            
            remaining_amount = amount
            total_cost = 0.0
            
            for price, size in levels:
                if remaining_amount <= 0:
                    break
                
                fill_amount = min(remaining_amount, size)
                total_cost += fill_amount * price
                remaining_amount -= fill_amount
            
            fillable_amount = amount - remaining_amount
            
            if fillable_amount > 0:
                avg_price = total_cost / fillable_amount
                best_price = levels[0][0]
                slippage = abs(avg_price - best_price) / best_price if best_price > 0 else 0
                
                # Calculate liquidity score
                total_liquidity = sum(level[1] for level in levels[:10])
                liquidity_score = min(total_liquidity / amount, 10.0) if amount > 0 else 0
                
                return {
                    'average_price': avg_price,
                    'best_price': best_price,
                    'slippage_bps': slippage * 10000,
                    'fillable_amount': fillable_amount,
                    'liquidity_score': liquidity_score
                }
            
            return {
                'average_price': 0,
                'best_price': levels[0][0] if levels else 0,
                'slippage_bps': 10000,
                'fillable_amount': 0,
                'liquidity_score': 0
            }
            
        except Exception as e:
            logger.error("Failed to calculate execution cost", error=str(e))
            return {
                'average_price': 0,
                'best_price': 0,
                'slippage_bps': 10000,
                'fillable_amount': 0,
                'liquidity_score': 0
            }
    
    def _calculate_price_score(self, ticker: Dict[str, Any], side: str) -> float:
        """Calculate price competitiveness score."""
        try:
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            
            if bid <= 0 or ask <= 0:
                return 0.0
            
            spread = ask - bid
            mid_price = (bid + ask) / 2
            spread_pct = spread / mid_price if mid_price > 0 else 1.0
            
            # Lower spread = higher score
            price_score = max(0, 1.0 - spread_pct * 100)  # Normalize spread percentage
            
            return price_score
            
        except Exception:
            return 0.0
    
    async def _get_backup_routing(self, order: OrderRequest, backup_exchange: str) -> RoutingDecision:
        """Get routing decision for backup exchange."""
        # Simplified routing for failover - just use the backup exchange
        metrics = self.exchange_metrics[backup_exchange]
        
        return RoutingDecision(
            selected_exchange=backup_exchange,
            reason=RoutingReason.FAILOVER,
            expected_price=0.0,  # Will be determined at execution
            expected_slippage_bps=0.0,
            liquidity_score=metrics.liquidity_score,
            latency_ms=metrics.latency_ms,
            confidence=0.7,  # Lower confidence for failover
            alternatives=[]
        )
    
    def _get_cached_decision(self, cache_key: str) -> Optional[RoutingDecision]:
        """Get cached routing decision if still valid."""
        if cache_key in self.routing_cache:
            cached_data = self.routing_cache[cache_key]
            if datetime.utcnow() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl_seconds):
                return cached_data['decision']
        return None
    
    def _cache_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision."""
        self.routing_cache[cache_key] = {
            'decision': decision,
            'timestamp': datetime.utcnow()
        }
        
        # Clean old cache entries
        if len(self.routing_cache) > 1000:
            # Remove oldest 20% of entries
            sorted_items = sorted(self.routing_cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_items[:200]:
                del self.routing_cache[key]
    
    async def _health_monitor(self):
        """Background task to monitor exchange health."""
        while True:
            try:
                for exchange_name, client in self.exchange_clients.items():
                    try:
                        # Simple health check
                        start_time = time.time()
                        await client.get_status()
                        
                        latency_ms = (time.time() - start_time) * 1000
                        self.exchange_metrics[exchange_name].update_latency(latency_ms)
                        self.exchange_metrics[exchange_name].update_success_rate(True)
                        
                        # Update Prometheus metrics
                        health_score = self._calculate_health_score(exchange_name)
                        EXCHANGE_HEALTH_SCORE.labels(exchange=exchange_name).set(health_score)
                        
                    except Exception as e:
                        logger.warning("Health check failed", exchange=exchange_name, error=str(e))
                        self.exchange_metrics[exchange_name].update_success_rate(False)
                        EXCHANGE_HEALTH_SCORE.labels(exchange=exchange_name).set(0)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(5)
    
    async def _latency_monitor(self):
        """Background task to continuously monitor latency."""
        while True:
            try:
                # Ping all exchanges with lightweight requests
                for exchange_name, client in self.exchange_clients.items():
                    if self.exchange_metrics[exchange_name].is_healthy:
                        try:
                            start_time = time.time()
                            await client.get_status()
                            latency_ms = (time.time() - start_time) * 1000
                            self.exchange_metrics[exchange_name].update_latency(latency_ms)
                        except:
                            pass  # Ignore errors in latency monitoring
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Latency monitor error", error=str(e))
                await asyncio.sleep(5)
    
    def _categorize_exchange_latency(self, exchange_name: str) -> str:
        """Categorize exchange by latency performance."""
        latency = self.exchange_metrics[exchange_name].latency_ms
        
        if latency < self.latency_thresholds['excellent']:
            return 'excellent'
        elif latency < self.latency_thresholds['good']:
            return 'good'
        elif latency < self.latency_thresholds['acceptable']:
            return 'acceptable'
        elif latency < self.latency_thresholds['poor']:
            return 'poor'
        else:
            return 'unacceptable'

    def _calculate_health_score(self, exchange_name: str) -> float:
        """Calculate overall health score for an exchange."""
        metrics = self.exchange_metrics[exchange_name]
        
        # Normalize metrics to 0-1 scale
        latency_score = max(0, 1 - metrics.latency_ms / 1000)  # 1000ms = 0 score
        success_score = metrics.success_rate
        uptime_score = metrics.uptime_pct / 100
        
        # Weighted average
        health_score = (
            latency_score * 0.3 +
            success_score * 0.5 +
            uptime_score * 0.2
        )
        
        return health_score
    
    def get_exchange_status(self) -> Dict[str, Any]:
        """Get current status of all exchanges."""
        status = {}
        
        for exchange_name, metrics in self.exchange_metrics.items():
            status[exchange_name] = {
                'name': exchange_name,
                'is_healthy': metrics.is_healthy,
                'latency_ms': metrics.latency_ms,
                'latency_category': self._categorize_exchange_latency(exchange_name),
                'success_rate': metrics.success_rate,
                'consecutive_failures': metrics.consecutive_failures,
                'last_update': metrics.last_update.isoformat(),
                'health_score': self._calculate_health_score(exchange_name),
                'liquidity_score': metrics.liquidity_score,
                'uptime_pct': metrics.uptime_pct
            }
        
        return status

# Global router instance
smart_router = SmartOrderRouter()

# FastAPI app
app = FastAPI(
    title="Smart Order Routing Service",
    description="Intelligent order routing with best execution and failover",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "smart-order-routing",
        "version": "1.0.0"
    }

@app.get("/api/exchanges/status")
async def get_exchanges_status():
    """Get status of all exchanges."""
    return {
        "exchanges": smart_router.get_exchange_status(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/route")
async def route_order(order: OrderRequest) -> RoutingDecision:
    """Route order to best exchange."""
    return await smart_router.route_order(order)

@app.get("/api/routing/stats")
async def get_routing_statistics():
    """Get routing performance statistics."""
    stats = {
        "total_exchanges": len(smart_router.exchange_clients),
        "healthy_exchanges": sum(1 for m in smart_router.exchange_metrics.values() if m.is_healthy),
        "average_latency_ms": statistics.mean([m.latency_ms for m in smart_router.exchange_metrics.values()]) if smart_router.exchange_metrics else 0,
        "cache_size": len(smart_router.routing_cache),
        "exchange_performance": {}
    }
    
    for name, metrics in smart_router.exchange_metrics.items():
        stats["exchange_performance"][name] = {
            "latency_ms": metrics.latency_ms,
            "success_rate": metrics.success_rate,
            "health_score": smart_router._calculate_health_score(name),
            "latency_category": smart_router._categorize_exchange_latency(name)
        }
    
    return stats

@app.post("/api/execute")
async def execute_order_with_routing(order: OrderRequest):
    """Route and execute order with failover."""
    try:
        # Get routing decision
        routing_decision = await smart_router.route_order(order)
        
        # Execute with failover
        result = await smart_router.execute_with_failover(order, routing_decision)
        
        return {
            "success": True,
            "routing_decision": routing_decision,
            "execution_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to execute order", error=str(e), order=order.dict())
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(generate_latest(), media_type="text/plain")

@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Smart Order Routing service starting up...")
    
    # Initialize database connections
    from database import initialize_databases
    await initialize_databases()
    
    # Initialize exchange clients (will be set by main application)
    # smart_router will be initialized when exchange clients are available
    
    logger.info("Smart Order Routing service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Smart Order Routing service shutting down...")
    
    # Cleanup
    from database import cleanup_databases
    await cleanup_databases()
    
    logger.info("Smart Order Routing service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info"
    )
# Import advanced order types
from advanced_order_types import (
    AdvancedOrderExecutor, AdvancedOrderRequest, AdvancedOrderType,
    IcebergOrderConfig, TWAPConfig, SmartSplitConfig,
    ExecutionStatus, create_advanced_order_executor
)

# Global advanced order executor
advanced_order_executor: Optional[AdvancedOrderExecutor] = None

# Advanced Order Endpoints

@app.post("/advanced-orders/submit")
async def submit_advanced_order(order_request: AdvancedOrderRequest):
    """Submit an advanced order for execution."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        result = await advanced_order_executor.submit_advanced_order(order_request)
        return result
    except Exception as e:
        logger.error("Failed to submit advanced order", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/advanced-orders/{order_id}/status")
async def get_advanced_order_status(order_id: str):
    """Get status of an advanced order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        result = await advanced_order_executor.get_order_status(order_id)
        if not result['success']:
            raise HTTPException(status_code=404, detail=result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get order status", order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanced-orders/{order_id}/cancel")
async def cancel_advanced_order(order_id: str):
    """Cancel an active advanced order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        result = await advanced_order_executor.cancel_order(order_id)
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel order", order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/advanced-orders/active")
async def list_active_advanced_orders():
    """List all active advanced orders."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        result = await advanced_order_executor.list_active_orders()
        return result
    except Exception as e:
        logger.error("Failed to list active orders", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/advanced-orders/{order_id}/metrics")
async def get_execution_metrics(order_id: str):
    """Get detailed execution metrics for an order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        result = await advanced_order_executor.get_execution_metrics(order_id)
        if not result['success']:
            raise HTTPException(status_code=404, detail=result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get execution metrics", order_id=order_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Example endpoints for creating different order types

@app.post("/advanced-orders/iceberg")
async def create_iceberg_order(
    order_id: str,
    symbol: str,
    side: str,
    total_quantity: float,
    visible_quantity: float,
    user_id: str,
    strategy_id: Optional[str] = None,
    price_improvement_threshold_bps: float = 5.0,
    max_slices: int = 100,
    slice_interval_seconds: float = 30.0,
    randomize_timing: bool = True,
    randomize_quantity: bool = True
):
    """Create and submit an iceberg order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        config = IcebergOrderConfig(
            total_quantity=Decimal(str(total_quantity)),
            visible_quantity=Decimal(str(visible_quantity)),
            price_improvement_threshold_bps=price_improvement_threshold_bps,
            max_slices=max_slices,
            slice_interval_seconds=slice_interval_seconds,
            randomize_timing=randomize_timing,
            randomize_quantity=randomize_quantity
        )
        
        order_request = AdvancedOrderRequest(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=AdvancedOrderType.ICEBERG,
            config=config,
            user_id=user_id,
            strategy_id=strategy_id
        )
        
        result = await advanced_order_executor.submit_advanced_order(order_request)
        return result
    except Exception as e:
        logger.error("Failed to create iceberg order", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanced-orders/twap")
async def create_twap_order(
    order_id: str,
    symbol: str,
    side: str,
    total_quantity: float,
    duration_minutes: int,
    user_id: str,
    strategy_id: Optional[str] = None,
    slice_interval_minutes: float = 5.0,
    participation_rate: float = 0.1,
    price_limit: Optional[float] = None,
    adaptive_sizing: bool = True,
    market_impact_threshold_bps: float = 20.0
):
    """Create and submit a TWAP order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        config = TWAPConfig(
            total_quantity=Decimal(str(total_quantity)),
            duration_minutes=duration_minutes,
            slice_interval_minutes=slice_interval_minutes,
            participation_rate=participation_rate,
            price_limit=Decimal(str(price_limit)) if price_limit else None,
            adaptive_sizing=adaptive_sizing,
            market_impact_threshold_bps=market_impact_threshold_bps
        )
        
        order_request = AdvancedOrderRequest(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=AdvancedOrderType.TWAP,
            config=config,
            user_id=user_id,
            strategy_id=strategy_id
        )
        
        result = await advanced_order_executor.submit_advanced_order(order_request)
        return result
    except Exception as e:
        logger.error("Failed to create TWAP order", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanced-orders/smart-split")
async def create_smart_split_order(
    order_id: str,
    symbol: str,
    side: str,
    total_quantity: float,
    user_id: str,
    strategy_id: Optional[str] = None,
    max_exchanges: int = 3,
    min_slice_size: float = 0.001,
    liquidity_threshold: float = 0.05,
    rebalance_interval_seconds: float = 60.0,
    cost_optimization: bool = True
):
    """Create and submit a smart split order."""
    if not advanced_order_executor:
        raise HTTPException(status_code=503, detail="Advanced order executor not initialized")
    
    try:
        config = SmartSplitConfig(
            total_quantity=Decimal(str(total_quantity)),
            max_exchanges=max_exchanges,
            min_slice_size=Decimal(str(min_slice_size)),
            liquidity_threshold=liquidity_threshold,
            rebalance_interval_seconds=rebalance_interval_seconds,
            cost_optimization=cost_optimization
        )
        
        order_request = AdvancedOrderRequest(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=AdvancedOrderType.SMART_SPLIT,
            config=config,
            user_id=user_id,
            strategy_id=strategy_id
        )
        
        result = await advanced_order_executor.submit_advanced_order(order_request)
        return result
    except Exception as e:
        logger.error("Failed to create smart split order", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Update the startup event to initialize advanced order executor
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global smart_router, market_data_service, advanced_order_executor
    
    try:
        # Initialize exchange clients (mock for now)
        exchange_clients = {
            'binance': MockExchangeClient('binance'),
            'coinbase': MockExchangeClient('coinbase'),
            'kraken': MockExchangeClient('kraken')
        }
        
        # Initialize smart router
        smart_router = SmartOrderRouter()
        await smart_router.initialize(exchange_clients)
        
        # Initialize market data service (mock for now)
        market_data_service = MockMarketDataService()
        
        # Initialize advanced order executor
        advanced_order_executor = create_advanced_order_executor(
            smart_router=smart_router,
            market_data_service=market_data_service,
            exchange_clients=exchange_clients
        )
        
        logger.info("Smart Order Routing Service started successfully")
        
    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise

# Add health check endpoint that includes advanced order executor status
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including advanced order executor."""
    global smart_router, advanced_order_executor
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'smart_router': 'healthy' if smart_router else 'not_initialized',
            'advanced_order_executor': 'healthy' if advanced_order_executor else 'not_initialized'
        }
    }
    
    if advanced_order_executor:
        try:
            active_orders = await advanced_order_executor.list_active_orders()
            health_status['services']['advanced_order_executor'] = {
                'status': 'healthy',
                'active_orders_count': active_orders.get('count', 0)
            }
        except Exception as e:
            health_status['services']['advanced_order_executor'] = {
                'status': 'error',
                'error': str(e)
            }
    
    if smart_router:
        health_status['services']['smart_router'] = {
            'status': 'healthy',
            'exchange_count': len(smart_router.exchange_clients),
            'healthy_exchanges': len([
                name for name, metrics in smart_router.exchange_metrics.items()
                if metrics.is_healthy
            ])
        }
    
    return health_status