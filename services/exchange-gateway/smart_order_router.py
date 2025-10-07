"""Smart Order Routing System
Implements intelligent order routing for optimal execution across multiple exchanges.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
from collections import defaultdict, deque

try:
    from exchange_abstraction import ExchangeManager, OrderSide, OrderType, OrderStatus, Order, OrderBook, Ticker
except ImportError:
    # Mock classes for testing
    class ExchangeManager:
        pass
    class OrderSide:
        BUY = "buy"
        SELL = "sell"
    class OrderType:
        MARKET = "market"
        LIMIT = "limit"
    class OrderStatus:
        PENDING = "pending"
        FILLED = "filled"
    class Order:
        pass
    class OrderBook:
        pass
    class Ticker:
        pass

logger = structlog.get_logger("smart-order-router")


class RoutingStrategy(str, Enum):
    """Order routing strategies."""
    BEST_PRICE = "best_price"  # Route to exchange with best price
    LOWEST_COST = "lowest_cost"  # Route considering fees and slippage
    FASTEST_EXECUTION = "fastest_execution"  # Route to fastest exchange
    LIQUIDITY_SEEKING = "liquidity_seeking"  # Route to exchange with most liquidity
    VOLUME_WEIGHTED = "volume_weighted"  # Split order based on volume
    TIME_WEIGHTED = "time_weighted"  # TWAP execution
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # Minimize market impact
    SMART_SPLIT = "smart_split"  # Intelligently split across exchanges


class ExecutionAlgorithm(str, Enum):
    """Execution algorithms."""
    IMMEDIATE = "immediate"  # Execute immediately
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"  # Percentage of Volume
    ICEBERG = "iceberg"  # Iceberg orders
    SNIPER = "sniper"  # Snipe liquidity
    STEALTH = "stealth"  # Minimize market impact


@dataclass
class ExecutionVenue:
    """Represents an execution venue (exchange)."""
    name: str
    is_active: bool
    latency_ms: float
    success_rate: float
    avg_fill_rate: float
    maker_fee: Decimal
    taker_fee: Decimal
    min_order_size: Decimal
    max_order_size: Decimal
    supported_order_types: List[OrderType]
    
    # Performance metrics
    avg_execution_time: float = 0.0
    recent_failures: int = 0
    last_failure_time: Optional[datetime] = None
    
    @property
    def health_score(self) -> float:
        """Calculate venue health score (0-100)."""
        base_score = self.success_rate * 100
        
        # Penalize for recent failures
        if self.recent_failures > 0:
            failure_penalty = min(self.recent_failures * 5, 30)
            base_score -= failure_penalty
        
        # Penalize for high latency
        if self.latency_ms > 100:
            latency_penalty = min((self.latency_ms - 100) / 10, 20)
            base_score -= latency_penalty
        
        return max(0, min(100, base_score))
    
    @property
    def is_healthy(self) -> bool:
        """Check if venue is healthy for routing."""
        return self.is_active and self.health_score >= 70


@dataclass
class MarketImpactModel:
    """Market impact estimation model."""
    symbol: str
    exchange: str
    linear_coefficient: float = 0.1  # Linear impact coefficient
    sqrt_coefficient: float = 0.05   # Square root impact coefficient
    permanent_impact: float = 0.3    # Permanent vs temporary impact ratio
    
    def estimate_impact(self, order_size: Decimal, avg_volume: Decimal) -> Decimal:
        """Estimate market impact for an order."""
        if avg_volume <= 0:
            return Decimal('1.0')  # High impact if no volume data
        
        participation_rate = float(order_size / avg_volume)
        
        # Market impact model: linear + square root components
        linear_impact = self.linear_coefficient * participation_rate
        sqrt_impact = self.sqrt_coefficient * np.sqrt(participation_rate)
        
        total_impact = linear_impact + sqrt_impact
        return Decimal(str(total_impact * 100))  # Return as percentage


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    venue: str
    quantity: Decimal
    price: Optional[Decimal]
    order_type: OrderType
    expected_fill_time: float
    expected_cost: Decimal
    confidence_score: float
    reasoning: str


@dataclass
class SmartOrderRequest:
    """Smart order request."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE
    max_participation_rate: float = 0.2  # Max 20% of volume
    time_horizon_minutes: int = 5  # Execution time horizon
    urgency: float = 0.5  # 0 = patient, 1 = urgent
    allow_partial_fills: bool = True
    min_fill_size: Optional[Decimal] = None
    max_slippage_pct: Optional[Decimal] = None
    client_order_id: Optional[str] = None


@dataclass
class ExecutionReport:
    """Execution report."""
    request_id: str
    symbol: str
    side: OrderSide
    requested_quantity: Decimal
    filled_quantity: Decimal
    avg_fill_price: Decimal
    total_cost: Decimal
    total_fees: Decimal
    execution_time: float
    venues_used: List[str]
    routing_decisions: List[RoutingDecision]
    slippage_pct: Decimal
    implementation_shortfall: Decimal
    status: str
    timestamp: datetime


class VenueSelector:
    """Selects optimal venues for order execution."""
    
    def __init__(self):
        self.venues: Dict[str, ExecutionVenue] = {}
        self.market_impact_models: Dict[str, Dict[str, MarketImpactModel]] = {}  # {exchange: {symbol: model}}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_venue(self, venue: ExecutionVenue):
        """Add an execution venue."""
        self.venues[venue.name] = venue
        logger.info(f"Added execution venue: {venue.name}")
    
    def update_venue_performance(self, venue_name: str, execution_time: float, success: bool):
        """Update venue performance metrics."""
        if venue_name not in self.venues:
            return
        
        venue = self.venues[venue_name]
        
        # Update execution time
        if venue.avg_execution_time == 0:
            venue.avg_execution_time = execution_time
        else:
            venue.avg_execution_time = (venue.avg_execution_time * 0.9) + (execution_time * 0.1)
        
        # Update success rate
        self.performance_history[venue_name].append(success)
        recent_performance = list(self.performance_history[venue_name])
        venue.success_rate = sum(recent_performance) / len(recent_performance)
        
        # Update failure tracking
        if not success:
            venue.recent_failures += 1
            venue.last_failure_time = datetime.now()
        else:
            # Decay recent failures over time
            if venue.last_failure_time:
                time_since_failure = (datetime.now() - venue.last_failure_time).total_seconds()
                if time_since_failure > 300:  # 5 minutes
                    venue.recent_failures = max(0, venue.recent_failures - 1)
    
    def get_healthy_venues(self) -> List[ExecutionVenue]:
        """Get list of healthy venues."""
        return [venue for venue in self.venues.values() if venue.is_healthy]
    
    def select_venues_for_strategy(self, 
                                  request: SmartOrderRequest,
                                  market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Select venues based on routing strategy."""
        healthy_venues = self.get_healthy_venues()
        
        if not healthy_venues:
            raise Exception("No healthy venues available for routing")
        
        if request.strategy == RoutingStrategy.BEST_PRICE:
            return self._select_best_price(request, healthy_venues, market_data)
        elif request.strategy == RoutingStrategy.LOWEST_COST:
            return self._select_lowest_cost(request, healthy_venues, market_data)
        elif request.strategy == RoutingStrategy.FASTEST_EXECUTION:
            return self._select_fastest_execution(request, healthy_venues, market_data)
        elif request.strategy == RoutingStrategy.LIQUIDITY_SEEKING:
            return self._select_liquidity_seeking(request, healthy_venues, market_data)
        elif request.strategy == RoutingStrategy.SMART_SPLIT:
            return self._select_smart_split(request, healthy_venues, market_data)
        else:
            # Default to best price
            return self._select_best_price(request, healthy_venues, market_data)
    
    def _select_best_price(self, 
                          request: SmartOrderRequest,
                          venues: List[ExecutionVenue],
                          market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Select venue with best price."""
        best_venue = None
        best_price = None
        
        for venue in venues:
            if venue.name not in market_data:
                continue
            
            ticker = market_data[venue.name].get('ticker')
            if not ticker:
                continue
            
            # Get relevant price based on order side
            if request.side == OrderSide.BUY:
                price = ticker.get('ask_price')
                if best_price is None or (price and price < best_price):
                    best_price = price
                    best_venue = venue
            else:
                price = ticker.get('bid_price')
                if best_price is None or (price and price > best_price):
                    best_price = price
                    best_venue = venue
        
        if not best_venue:
            raise Exception("No venue with valid pricing found")
        
        return [RoutingDecision(
            venue=best_venue.name,
            quantity=request.quantity,
            price=best_price,
            order_type=request.order_type,
            expected_fill_time=best_venue.avg_execution_time,
            expected_cost=self._calculate_expected_cost(request, best_venue, best_price),
            confidence_score=best_venue.health_score / 100,
            reasoning=f"Best price: {best_price}"
        )]
    
    def _select_lowest_cost(self,
                           request: SmartOrderRequest,
                           venues: List[ExecutionVenue],
                           market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Select venue with lowest total cost (price + fees + slippage)."""
        best_venue = None
        lowest_cost = None
        
        for venue in venues:
            if venue.name not in market_data:
                continue
            
            ticker = market_data[venue.name].get('ticker')
            order_book = market_data[venue.name].get('order_book')
            if not ticker:
                continue
            
            # Calculate total cost
            if request.side == OrderSide.BUY:
                base_price = ticker.get('ask_price', Decimal('0'))
            else:
                base_price = ticker.get('bid_price', Decimal('0'))
            
            if base_price <= 0:
                continue
            
            # Add fees
            fee_rate = venue.taker_fee if request.order_type == OrderType.MARKET else venue.maker_fee
            fee_cost = base_price * request.quantity * fee_rate / 100
            
            # Estimate slippage
            slippage_cost = self._estimate_slippage_cost(request, venue, order_book)
            
            total_cost = base_price * request.quantity + fee_cost + slippage_cost
            
            if lowest_cost is None or total_cost < lowest_cost:
                lowest_cost = total_cost
                best_venue = venue
        
        if not best_venue:
            raise Exception("No venue with valid cost calculation found")
        
        return [RoutingDecision(
            venue=best_venue.name,
            quantity=request.quantity,
            price=None,  # Market order
            order_type=request.order_type,
            expected_fill_time=best_venue.avg_execution_time,
            expected_cost=lowest_cost,
            confidence_score=best_venue.health_score / 100,
            reasoning=f"Lowest total cost: {lowest_cost}"
        )]
    
    def _select_fastest_execution(self,
                                 request: SmartOrderRequest,
                                 venues: List[ExecutionVenue],
                                 market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Select venue with fastest execution."""
        fastest_venue = min(venues, key=lambda v: v.latency_ms + v.avg_execution_time * 1000)
        
        # Get price from market data
        price = None
        if fastest_venue.name in market_data:
            ticker = market_data[fastest_venue.name].get('ticker')
            if ticker:
                if request.side == OrderSide.BUY:
                    price = ticker.get('ask_price')
                else:
                    price = ticker.get('bid_price')
        
        return [RoutingDecision(
            venue=fastest_venue.name,
            quantity=request.quantity,
            price=price,
            order_type=request.order_type,
            expected_fill_time=fastest_venue.avg_execution_time,
            expected_cost=self._calculate_expected_cost(request, fastest_venue, price),
            confidence_score=fastest_venue.health_score / 100,
            reasoning=f"Fastest execution: {fastest_venue.latency_ms}ms latency"
        )]
    
    def _select_liquidity_seeking(self,
                                 request: SmartOrderRequest,
                                 venues: List[ExecutionVenue],
                                 market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Select venue with most liquidity."""
        best_venue = None
        best_liquidity = Decimal('0')
        
        for venue in venues:
            if venue.name not in market_data:
                continue
            
            order_book = market_data[venue.name].get('order_book')
            if not order_book:
                continue
            
            # Calculate available liquidity
            if request.side == OrderSide.BUY:
                liquidity = sum(qty for _, qty in order_book.get('asks', [])[:10])
            else:
                liquidity = sum(qty for _, qty in order_book.get('bids', [])[:10])
            
            if liquidity > best_liquidity:
                best_liquidity = liquidity
                best_venue = venue
        
        if not best_venue:
            raise Exception("No venue with sufficient liquidity found")
        
        # Get price
        price = None
        if best_venue.name in market_data:
            ticker = market_data[best_venue.name].get('ticker')
            if ticker:
                if request.side == OrderSide.BUY:
                    price = ticker.get('ask_price')
                else:
                    price = ticker.get('bid_price')
        
        return [RoutingDecision(
            venue=best_venue.name,
            quantity=request.quantity,
            price=price,
            order_type=request.order_type,
            expected_fill_time=best_venue.avg_execution_time,
            expected_cost=self._calculate_expected_cost(request, best_venue, price),
            confidence_score=best_venue.health_score / 100,
            reasoning=f"Best liquidity: {best_liquidity} available"
        )]
    
    def _select_smart_split(self,
                           request: SmartOrderRequest,
                           venues: List[ExecutionVenue],
                           market_data: Dict[str, Dict[str, Any]]) -> List[RoutingDecision]:
        """Intelligently split order across multiple venues."""
        decisions = []
        remaining_quantity = request.quantity
        
        # Score venues based on multiple factors
        venue_scores = []
        for venue in venues:
            if venue.name not in market_data:
                continue
            
            ticker = market_data[venue.name].get('ticker')
            order_book = market_data[venue.name].get('order_book')
            if not ticker or not order_book:
                continue
            
            # Calculate composite score
            price_score = self._calculate_price_score(request, ticker)
            liquidity_score = self._calculate_liquidity_score(request, order_book)
            cost_score = self._calculate_cost_score(request, venue, ticker)
            speed_score = self._calculate_speed_score(venue)
            
            composite_score = (
                price_score * 0.3 +
                liquidity_score * 0.3 +
                cost_score * 0.2 +
                speed_score * 0.2
            )
            
            venue_scores.append((venue, composite_score, ticker, order_book))
        
        # Sort by score (descending)
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate quantity based on scores and liquidity
        total_score = sum(score for _, score, _, _ in venue_scores)
        
        for venue, score, ticker, order_book in venue_scores:
            if remaining_quantity <= 0:
                break
            
            # Calculate allocation based on score and available liquidity
            score_allocation = (score / total_score) * request.quantity
            
            # Get available liquidity
            if request.side == OrderSide.BUY:
                available_liquidity = sum(qty for _, qty in order_book.get('asks', [])[:5])
                price = ticker.get('ask_price')
            else:
                available_liquidity = sum(qty for _, qty in order_book.get('bids', [])[:5])
                price = ticker.get('bid_price')
            
            # Limit allocation to available liquidity and remaining quantity
            allocation = min(score_allocation, available_liquidity * Decimal('0.5'), remaining_quantity)
            
            if allocation > venue.min_order_size:
                decisions.append(RoutingDecision(
                    venue=venue.name,
                    quantity=allocation,
                    price=price,
                    order_type=request.order_type,
                    expected_fill_time=venue.avg_execution_time,
                    expected_cost=self._calculate_expected_cost(request, venue, price, allocation),
                    confidence_score=venue.health_score / 100,
                    reasoning=f"Smart split: {allocation} of {request.quantity} (score: {score:.2f})"
                ))
                remaining_quantity -= allocation
        
        if not decisions:
            # Fallback to best price if smart split fails
            return self._select_best_price(request, venues, market_data)
        
        return decisions
    
    def _calculate_price_score(self, request: SmartOrderRequest, ticker: Dict[str, Any]) -> float:
        """Calculate price score (0-100)."""
        if request.side == OrderSide.BUY:
            price = ticker.get('ask_price', float('inf'))
            # Lower ask price = higher score
            return max(0, 100 - float(price) / 1000)  # Simplified scoring
        else:
            price = ticker.get('bid_price', 0)
            # Higher bid price = higher score
            return min(100, float(price) / 1000)  # Simplified scoring
    
    def _calculate_liquidity_score(self, request: SmartOrderRequest, order_book: Dict[str, Any]) -> float:
        """Calculate liquidity score (0-100)."""
        if request.side == OrderSide.BUY:
            liquidity = sum(float(qty) for _, qty in order_book.get('asks', [])[:10])
        else:
            liquidity = sum(float(qty) for _, qty in order_book.get('bids', [])[:10])
        
        # Normalize liquidity score
        return min(100, liquidity / float(request.quantity) * 20)
    
    def _calculate_cost_score(self, request: SmartOrderRequest, venue: ExecutionVenue, ticker: Dict[str, Any]) -> float:
        """Calculate cost score (0-100)."""
        fee_rate = venue.taker_fee if request.order_type == OrderType.MARKET else venue.maker_fee
        # Lower fees = higher score
        return max(0, 100 - float(fee_rate) * 10)
    
    def _calculate_speed_score(self, venue: ExecutionVenue) -> float:
        """Calculate speed score (0-100)."""
        # Lower latency = higher score
        return max(0, 100 - venue.latency_ms / 10)
    
    def _calculate_expected_cost(self, 
                               request: SmartOrderRequest, 
                               venue: ExecutionVenue, 
                               price: Optional[Decimal],
                               quantity: Optional[Decimal] = None) -> Decimal:
        """Calculate expected total cost."""
        if not price:
            return Decimal('0')
        
        qty = quantity or request.quantity
        base_cost = price * qty
        
        # Add fees
        fee_rate = venue.taker_fee if request.order_type == OrderType.MARKET else venue.maker_fee
        fee_cost = base_cost * fee_rate / 100
        
        return base_cost + fee_cost
    
    def _estimate_slippage_cost(self, 
                              request: SmartOrderRequest, 
                              venue: ExecutionVenue, 
                              order_book: Optional[Dict[str, Any]]) -> Decimal:
        """Estimate slippage cost."""
        if not order_book:
            return Decimal('0')
        
        # Simple slippage estimation based on order book depth
        if request.side == OrderSide.BUY:
            levels = order_book.get('asks', [])
        else:
            levels = order_book.get('bids', [])
        
        if not levels:
            return Decimal('0')
        
        # Calculate weighted average price for the order size
        remaining_qty = request.quantity
        total_cost = Decimal('0')
        
        for price, qty in levels[:10]:  # Look at top 10 levels
            if remaining_qty <= 0:
                break
            
            fill_qty = min(remaining_qty, qty)
            total_cost += price * fill_qty
            remaining_qty -= fill_qty
        
        if remaining_qty > 0:
            # Not enough liquidity, high slippage
            return request.quantity * Decimal('0.01')  # 1% slippage penalty
        
        avg_price = total_cost / request.quantity
        best_price = levels[0][0]
        
        slippage = abs(avg_price - best_price)
        return slippage * request.quantity


class SmartOrderRouter:
    """Main smart order routing system."""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.venue_selector = VenueSelector()
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # Initialize venues from exchange manager
        self._initialize_venues()
    
    def _initialize_venues(self):
        """Initialize execution venues from exchange manager."""
        # This would be populated from actual exchange configurations
        default_venues = [
            ExecutionVenue(
                name="binance",
                is_active=True,
                latency_ms=50.0,
                success_rate=0.98,
                avg_fill_rate=0.95,
                maker_fee=Decimal('0.1'),
                taker_fee=Decimal('0.1'),
                min_order_size=Decimal('10'),
                max_order_size=Decimal('1000000'),
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT]
            ),
            ExecutionVenue(
                name="coinbase",
                is_active=True,
                latency_ms=80.0,
                success_rate=0.96,
                avg_fill_rate=0.92,
                maker_fee=Decimal('0.5'),
                taker_fee=Decimal('0.5'),
                min_order_size=Decimal('5'),
                max_order_size=Decimal('500000'),
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT]
            ),
            ExecutionVenue(
                name="kraken",
                is_active=True,
                latency_ms=120.0,
                success_rate=0.94,
                avg_fill_rate=0.90,
                maker_fee=Decimal('0.16'),
                taker_fee=Decimal('0.26'),
                min_order_size=Decimal('20'),
                max_order_size=Decimal('200000'),
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT]
            )
        ]
        
        for venue in default_venues:
            self.venue_selector.add_venue(venue)
    
    async def route_order(self, request: SmartOrderRequest) -> ExecutionReport:
        """Route and execute a smart order."""
        start_time = time.time()
        request_id = f"order_{int(time.time() * 1000)}"
        
        try:
            # Get market data
            market_data = await self._get_market_data(request.symbol)
            
            # Select venues
            routing_decisions = self.venue_selector.select_venues_for_strategy(
                request, market_data
            )
            
            if not routing_decisions:
                raise Exception("No routing decisions generated")
            
            # Execute orders
            execution_results = await self._execute_routing_decisions(
                request, routing_decisions
            )
            
            # Generate execution report
            report = self._generate_execution_report(
                request_id, request, routing_decisions, execution_results, start_time
            )
            
            self.execution_reports.append(report)
            return report
            
        except Exception as e:
            logger.error(f"Order routing failed: {str(e)}")
            
            # Generate failure report
            report = ExecutionReport(
                request_id=request_id,
                symbol=request.symbol,
                side=request.side,
                requested_quantity=request.quantity,
                filled_quantity=Decimal('0'),
                avg_fill_price=Decimal('0'),
                total_cost=Decimal('0'),
                total_fees=Decimal('0'),
                execution_time=time.time() - start_time,
                venues_used=[],
                routing_decisions=[],
                slippage_pct=Decimal('0'),
                implementation_shortfall=Decimal('0'),
                status=f"FAILED: {str(e)}",
                timestamp=datetime.now()
            )
            
            self.execution_reports.append(report)
            return report
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Get market data for routing decisions."""
        market_data = {}
        
        # Get tickers and order books from all active exchanges
        active_exchanges = self.exchange_manager.get_active_exchanges()
        
        for exchange in active_exchanges:
            try:
                # This would call the actual exchange manager methods
                # For now, we'll create mock data
                market_data[exchange] = {
                    'ticker': {
                        'bid_price': Decimal('50000'),
                        'ask_price': Decimal('50100'),
                        'volume': Decimal('100')
                    },
                    'order_book': {
                        'bids': [(Decimal('50000'), Decimal('1')), (Decimal('49990'), Decimal('2'))],
                        'asks': [(Decimal('50100'), Decimal('1.5')), (Decimal('50110'), Decimal('2.5'))]
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to get market data from {exchange}: {str(e)}")
        
        return market_data
    
    async def _execute_routing_decisions(self, 
                                       request: SmartOrderRequest,
                                       decisions: List[RoutingDecision]) -> List[Dict[str, Any]]:
        """Execute routing decisions."""
        results = []
        
        for decision in decisions:
            try:
                # This would call the actual exchange connector
                # For now, we'll simulate execution
                execution_result = {
                    'venue': decision.venue,
                    'quantity': decision.quantity,
                    'filled_quantity': decision.quantity,  # Assume full fill
                    'avg_price': decision.price or Decimal('50050'),
                    'fees': decision.expected_cost * Decimal('0.001'),  # 0.1% fees
                    'execution_time': decision.expected_fill_time,
                    'success': True
                }
                
                results.append(execution_result)
                
                # Update venue performance
                self.venue_selector.update_venue_performance(
                    decision.venue, decision.expected_fill_time, True
                )
                
            except Exception as e:
                logger.error(f"Execution failed on {decision.venue}: {str(e)}")
                
                execution_result = {
                    'venue': decision.venue,
                    'quantity': decision.quantity,
                    'filled_quantity': Decimal('0'),
                    'avg_price': Decimal('0'),
                    'fees': Decimal('0'),
                    'execution_time': 0,
                    'success': False,
                    'error': str(e)
                }
                
                results.append(execution_result)
                
                # Update venue performance
                self.venue_selector.update_venue_performance(
                    decision.venue, 0, False
                )
        
        return results
    
    def _generate_execution_report(self,
                                 request_id: str,
                                 request: SmartOrderRequest,
                                 decisions: List[RoutingDecision],
                                 results: List[Dict[str, Any]],
                                 start_time: float) -> ExecutionReport:
        """Generate execution report."""
        total_filled = sum(result['filled_quantity'] for result in results)
        total_cost = sum(result['avg_price'] * result['filled_quantity'] for result in results if result['success'])
        total_fees = sum(result['fees'] for result in results)
        
        avg_fill_price = total_cost / total_filled if total_filled > 0 else Decimal('0')
        
        # Calculate slippage (simplified)
        if request.limit_price and avg_fill_price > 0:
            slippage_pct = abs(avg_fill_price - request.limit_price) / request.limit_price * 100
        else:
            slippage_pct = Decimal('0')
        
        venues_used = [result['venue'] for result in results if result['success']]
        
        return ExecutionReport(
            request_id=request_id,
            symbol=request.symbol,
            side=request.side,
            requested_quantity=request.quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            total_cost=total_cost,
            total_fees=total_fees,
            execution_time=time.time() - start_time,
            venues_used=venues_used,
            routing_decisions=decisions,
            slippage_pct=slippage_pct,
            implementation_shortfall=Decimal('0'),  # Would calculate properly
            status="COMPLETED" if total_filled == request.quantity else "PARTIAL",
            timestamp=datetime.now()
        )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_reports:
            return {}
        
        successful_reports = [r for r in self.execution_reports if r.status == "COMPLETED"]
        
        return {
            'total_orders': len(self.execution_reports),
            'successful_orders': len(successful_reports),
            'success_rate': len(successful_reports) / len(self.execution_reports) * 100,
            'avg_execution_time': sum(r.execution_time for r in successful_reports) / len(successful_reports) if successful_reports else 0,
            'avg_slippage_pct': sum(r.slippage_pct for r in successful_reports) / len(successful_reports) if successful_reports else 0,
            'total_volume': sum(r.filled_quantity for r in successful_reports),
            'total_fees': sum(r.total_fees for r in successful_reports),
            'venue_usage': self._calculate_venue_usage(),
            'last_update': datetime.now().isoformat()
        }
    
    def _calculate_venue_usage(self) -> Dict[str, int]:
        """Calculate venue usage statistics."""
        venue_usage = defaultdict(int)
        
        for report in self.execution_reports:
            for venue in report.venues_used:
                venue_usage[venue] += 1
        
        return dict(venue_usage)