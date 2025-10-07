"""
Advanced Order Types and Execution Engine.
Implements iceberg orders, TWAP execution, and smart order splitting.
"""
import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import statistics
import random

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("advanced-order-types")

class AdvancedOrderType(str, Enum):
    """Advanced order types supported by the system."""
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    SMART_SPLIT = "SMART_SPLIT"
    ADAPTIVE = "ADAPTIVE"

class ExecutionStatus(str, Enum):
    """Execution status for advanced orders."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"

class SliceStatus(str, Enum):
    """Status of individual order slices."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

@dataclass
class OrderSlice:
    """Individual slice of a larger order."""
    id: str
    parent_order_id: str
    exchange: str
    symbol: str
    side: str
    quantity: Decimal
    price: Optional[Decimal] = None
    status: SliceStatus = SliceStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    exchange_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fees: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """Metrics for order execution performance."""
    total_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Decimal
    total_fees: Decimal
    execution_time_seconds: float
    slices_count: int
    successful_slices: int
    failed_slices: int
    market_impact_bps: float
    implementation_shortfall_bps: float
    participation_rate: float

class IcebergOrderConfig(BaseModel):
    """Configuration for iceberg orders."""
    total_quantity: Decimal = Field(..., description="Total order quantity")
    visible_quantity: Decimal = Field(..., description="Visible quantity per slice")
    price_improvement_threshold_bps: float = Field(5.0, description="Price improvement threshold in basis points")
    max_slices: int = Field(100, description="Maximum number of slices")
    slice_interval_seconds: float = Field(30.0, description="Interval between slices in seconds")
    randomize_timing: bool = Field(True, description="Randomize slice timing")
    randomize_quantity: bool = Field(True, description="Randomize slice quantities")

class TWAPConfig(BaseModel):
    """Configuration for TWAP (Time-Weighted Average Price) execution."""
    total_quantity: Decimal = Field(..., description="Total order quantity")
    duration_minutes: int = Field(..., description="Execution duration in minutes")
    slice_interval_minutes: float = Field(5.0, description="Interval between slices in minutes")
    participation_rate: float = Field(0.1, description="Maximum participation rate (0.0 to 1.0)")
    price_limit: Optional[Decimal] = Field(None, description="Price limit for execution")
    adaptive_sizing: bool = Field(True, description="Adapt slice sizes based on market conditions")
    market_impact_threshold_bps: float = Field(20.0, description="Market impact threshold in basis points")

class SmartSplitConfig(BaseModel):
    """Configuration for smart order splitting."""
    total_quantity: Decimal = Field(..., description="Total order quantity")
    max_exchanges: int = Field(3, description="Maximum number of exchanges to use")
    min_slice_size: Decimal = Field(Decimal('0.001'), description="Minimum slice size")
    liquidity_threshold: float = Field(0.05, description="Minimum liquidity threshold")
    rebalance_interval_seconds: float = Field(60.0, description="Rebalancing interval")
    cost_optimization: bool = Field(True, description="Optimize for total execution cost")

class AdvancedOrderRequest(BaseModel):
    """Request for advanced order execution."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading pair symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    order_type: AdvancedOrderType = Field(..., description="Advanced order type")
    config: Union[IcebergOrderConfig, TWAPConfig, SmartSplitConfig] = Field(..., description="Order configuration")
    user_id: str = Field(..., description="User identifier")
    strategy_id: Optional[str] = Field(None, description="Strategy identifier")
    priority: int = Field(5, description="Execution priority (1-10)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AdvancedOrderExecutor:
    """Advanced order execution engine."""
    
    def __init__(self, smart_router, market_data_service, exchange_clients):
        self.smart_router = smart_router
        self.market_data_service = market_data_service
        self.exchange_clients = exchange_clients
        
        # Active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.execution_metrics: Dict[str, ExecutionMetrics] = {}
        
        # Configuration
        self.max_concurrent_executions = 50
        self.slice_timeout_seconds = 300  # 5 minutes
        self.market_data_refresh_seconds = 10
        
    async def submit_advanced_order(self, order_request: AdvancedOrderRequest) -> Dict[str, Any]:
        """Submit an advanced order for execution."""
        order_id = order_request.order_id
        
        # Validate order
        validation_result = await self._validate_order(order_request)
        if not validation_result['valid']:
            return {
                'success': False,
                'order_id': order_id,
                'error': validation_result['error'],
                'status': ExecutionStatus.FAILED
            }
        
        # Check capacity
        if len(self.active_executions) >= self.max_concurrent_executions:
            return {
                'success': False,
                'order_id': order_id,
                'error': 'Maximum concurrent executions reached',
                'status': ExecutionStatus.FAILED
            }
        
        # Initialize execution state
        execution_state = {
            'order_request': order_request,
            'status': ExecutionStatus.PENDING,
            'slices': [],
            'filled_quantity': Decimal('0'),
            'total_fees': Decimal('0'),
            'start_time': datetime.utcnow(),
            'last_update': datetime.utcnow(),
            'metrics': None
        }
        
        self.active_executions[order_id] = execution_state
        
        # Start execution task based on order type
        if order_request.order_type == AdvancedOrderType.ICEBERG:
            task = asyncio.create_task(self._execute_iceberg_order(order_request))
        elif order_request.order_type == AdvancedOrderType.TWAP:
            task = asyncio.create_task(self._execute_twap_order(order_request))
        elif order_request.order_type == AdvancedOrderType.SMART_SPLIT:
            task = asyncio.create_task(self._execute_smart_split_order(order_request))
        else:
            return {
                'success': False,
                'order_id': order_id,
                'error': f'Unsupported order type: {order_request.order_type}',
                'status': ExecutionStatus.FAILED
            }
        
        self.execution_tasks[order_id] = task
        
        # Update status to active
        execution_state['status'] = ExecutionStatus.ACTIVE
        
        logger.info("Advanced order submitted",
                   order_id=order_id,
                   order_type=order_request.order_type.value,
                   symbol=order_request.symbol)
        
        return {
            'success': True,
            'order_id': order_id,
            'status': ExecutionStatus.ACTIVE,
            'message': f'{order_request.order_type.value} order execution started'
        }    

    async def _execute_iceberg_order(self, order_request: AdvancedOrderRequest) -> None:
        """Execute an iceberg order with hidden quantity management."""
        order_id = order_request.order_id
        config: IcebergOrderConfig = order_request.config
        execution_state = self.active_executions[order_id]
        
        logger.info("Starting iceberg order execution",
                   order_id=order_id,
                   total_quantity=float(config.total_quantity),
                   visible_quantity=float(config.visible_quantity))
        
        try:
            remaining_quantity = config.total_quantity
            slice_count = 0
            
            while remaining_quantity > 0 and slice_count < config.max_slices:
                # Calculate slice size
                slice_size = min(config.visible_quantity, remaining_quantity)
                
                # Apply randomization if enabled
                if config.randomize_quantity and slice_count > 0:
                    # Randomize ±20% of visible quantity
                    randomization_factor = 1.0 + random.uniform(-0.2, 0.2)
                    slice_size = min(
                        config.visible_quantity * Decimal(str(randomization_factor)),
                        remaining_quantity
                    )
                
                # Get current market conditions
                market_data = await self._get_market_data(order_request.symbol)
                if not market_data:
                    logger.warning("No market data available, pausing execution", order_id=order_id)
                    await asyncio.sleep(30)
                    continue
                
                # Check for price improvement opportunity
                current_price = self._get_current_price(market_data, order_request.side)
                if await self._should_wait_for_price_improvement(order_request, current_price, config):
                    logger.debug("Waiting for price improvement", order_id=order_id)
                    await asyncio.sleep(config.slice_interval_seconds / 2)
                    continue
                
                # Route and execute slice
                slice_result = await self._execute_order_slice(
                    order_request, slice_size, slice_count, market_data
                )
                
                if slice_result['success']:
                    filled_qty = slice_result['filled_quantity']
                    remaining_quantity -= filled_qty
                    execution_state['filled_quantity'] += filled_qty
                    execution_state['total_fees'] += slice_result.get('fees', Decimal('0'))
                    
                    logger.info("Iceberg slice executed",
                               order_id=order_id,
                               slice_count=slice_count,
                               filled_quantity=float(filled_qty),
                               remaining_quantity=float(remaining_quantity))
                else:
                    logger.warning("Iceberg slice failed",
                                 order_id=order_id,
                                 slice_count=slice_count,
                                 error=slice_result.get('error'))
                
                slice_count += 1
                
                # Wait before next slice (with randomization if enabled)
                wait_time = config.slice_interval_seconds
                if config.randomize_timing:
                    wait_time *= random.uniform(0.8, 1.2)
                
                if remaining_quantity > 0:
                    await asyncio.sleep(wait_time)
            
            # Update final status
            if remaining_quantity == 0:
                execution_state['status'] = ExecutionStatus.COMPLETED
                logger.info("Iceberg order completed", order_id=order_id)
            else:
                execution_state['status'] = ExecutionStatus.PARTIALLY_FILLED
                logger.info("Iceberg order partially filled", 
                           order_id=order_id,
                           remaining_quantity=float(remaining_quantity))
        
        except Exception as e:
            logger.error("Iceberg order execution failed", order_id=order_id, error=str(e))
            execution_state['status'] = ExecutionStatus.FAILED
        
        finally:
            execution_state['last_update'] = datetime.utcnow()
            await self._calculate_execution_metrics(order_id) 
   
    async def _execute_twap_order(self, order_request: AdvancedOrderRequest) -> None:
        """Execute a TWAP (Time-Weighted Average Price) order."""
        order_id = order_request.order_id
        config: TWAPConfig = order_request.config
        execution_state = self.active_executions[order_id]
        
        logger.info("Starting TWAP order execution",
                   order_id=order_id,
                   total_quantity=float(config.total_quantity),
                   duration_minutes=config.duration_minutes)
        
        try:
            # Calculate execution schedule
            total_slices = int(config.duration_minutes / config.slice_interval_minutes)
            base_slice_size = config.total_quantity / total_slices
            
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=config.duration_minutes)
            
            remaining_quantity = config.total_quantity
            slice_count = 0
            
            while datetime.utcnow() < end_time and remaining_quantity > 0:
                # Calculate time-based slice size
                time_remaining_pct = (end_time - datetime.utcnow()).total_seconds() / (config.duration_minutes * 60)
                target_slice_size = min(base_slice_size, remaining_quantity)
                
                # Adaptive sizing based on market conditions
                if config.adaptive_sizing:
                    market_data = await self._get_market_data(order_request.symbol)
                    if market_data:
                        volume_adjustment = await self._calculate_volume_adjustment(
                            market_data, config.participation_rate
                        )
                        target_slice_size = min(target_slice_size * volume_adjustment, remaining_quantity)
                
                # Check market impact
                market_impact = await self._estimate_market_impact(
                    order_request.symbol, order_request.side, target_slice_size
                )
                
                if market_impact > config.market_impact_threshold_bps:
                    # Reduce slice size to limit market impact
                    impact_adjustment = config.market_impact_threshold_bps / market_impact
                    target_slice_size *= Decimal(str(impact_adjustment))
                    logger.debug("Reducing slice size due to market impact",
                               order_id=order_id,
                               original_size=float(base_slice_size),
                               adjusted_size=float(target_slice_size))
                
                # Execute slice
                if target_slice_size > 0:
                    slice_result = await self._execute_order_slice(
                        order_request, target_slice_size, slice_count, market_data
                    )
                    
                    if slice_result['success']:
                        filled_qty = slice_result['filled_quantity']
                        remaining_quantity -= filled_qty
                        execution_state['filled_quantity'] += filled_qty
                        execution_state['total_fees'] += slice_result.get('fees', Decimal('0'))
                        
                        logger.info("TWAP slice executed",
                                   order_id=order_id,
                                   slice_count=slice_count,
                                   filled_quantity=float(filled_qty),
                                   remaining_quantity=float(remaining_quantity))
                
                slice_count += 1
                
                # Wait for next slice interval
                await asyncio.sleep(config.slice_interval_minutes * 60)
            
            # Update final status
            if remaining_quantity == 0:
                execution_state['status'] = ExecutionStatus.COMPLETED
            else:
                execution_state['status'] = ExecutionStatus.PARTIALLY_FILLED
            
            logger.info("TWAP order execution finished",
                       order_id=order_id,
                       status=execution_state['status'].value,
                       filled_quantity=float(execution_state['filled_quantity']))
        
        except Exception as e:
            logger.error("TWAP order execution failed", order_id=order_id, error=str(e))
            execution_state['status'] = ExecutionStatus.FAILED
        
        finally:
            execution_state['last_update'] = datetime.utcnow()
            await self._calculate_execution_metrics(order_id) 
   
    async def _execute_smart_split_order(self, order_request: AdvancedOrderRequest) -> None:
        """Execute smart order splitting across multiple exchanges."""
        order_id = order_request.order_id
        config: SmartSplitConfig = order_request.config
        execution_state = self.active_executions[order_id]
        
        logger.info("Starting smart split order execution",
                   order_id=order_id,
                   total_quantity=float(config.total_quantity),
                   max_exchanges=config.max_exchanges)
        
        try:
            remaining_quantity = config.total_quantity
            
            while remaining_quantity > 0:
                # Get market data from all exchanges
                all_market_data = await self._get_multi_exchange_market_data(order_request.symbol)
                
                if not all_market_data:
                    logger.warning("No market data available", order_id=order_id)
                    await asyncio.sleep(30)
                    continue
                
                # Calculate optimal allocation across exchanges
                allocation = await self._calculate_optimal_allocation(
                    order_request, remaining_quantity, all_market_data, config
                )
                
                if not allocation:
                    logger.warning("No suitable exchanges for allocation", order_id=order_id)
                    break
                
                # Execute slices concurrently across selected exchanges
                slice_tasks = []
                for exchange, slice_info in allocation.items():
                    if slice_info['quantity'] >= config.min_slice_size:
                        task = asyncio.create_task(
                            self._execute_exchange_slice(
                                order_request, exchange, slice_info, all_market_data[exchange]
                            )
                        )
                        slice_tasks.append((exchange, task))
                
                # Wait for all slices to complete
                slice_results = {}
                for exchange, task in slice_tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=self.slice_timeout_seconds)
                        slice_results[exchange] = result
                    except asyncio.TimeoutError:
                        logger.warning("Slice execution timeout", order_id=order_id, exchange=exchange)
                        slice_results[exchange] = {'success': False, 'error': 'timeout'}
                
                # Process results
                total_filled = Decimal('0')
                for exchange, result in slice_results.items():
                    if result['success']:
                        filled_qty = result['filled_quantity']
                        total_filled += filled_qty
                        execution_state['filled_quantity'] += filled_qty
                        execution_state['total_fees'] += result.get('fees', Decimal('0'))
                        
                        logger.info("Smart split slice executed",
                                   order_id=order_id,
                                   exchange=exchange,
                                   filled_quantity=float(filled_qty))
                
                remaining_quantity -= total_filled
                
                # If no progress made, break to avoid infinite loop
                if total_filled == 0:
                    logger.warning("No progress made in smart split execution", order_id=order_id)
                    break
                
                # Wait before rebalancing (if more quantity remains)
                if remaining_quantity > 0:
                    await asyncio.sleep(config.rebalance_interval_seconds)
            
            # Update final status
            if remaining_quantity == 0:
                execution_state['status'] = ExecutionStatus.COMPLETED
            else:
                execution_state['status'] = ExecutionStatus.PARTIALLY_FILLED
            
            logger.info("Smart split order execution finished",
                       order_id=order_id,
                       status=execution_state['status'].value,
                       filled_quantity=float(execution_state['filled_quantity']))
        
        except Exception as e:
            logger.error("Smart split order execution failed", order_id=order_id, error=str(e))
            execution_state['status'] = ExecutionStatus.FAILED
        
        finally:
            execution_state['last_update'] = datetime.utcnow()
            await self._calculate_execution_metrics(order_id) 
   
    async def _execute_order_slice(self, order_request: AdvancedOrderRequest, 
                                 slice_size: Decimal, slice_count: int,
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single order slice using smart routing."""
        try:
            # Create routing request
            routing_request = {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'type': 'market',  # Use market orders for advanced execution
                'amount': float(slice_size),
                'execution_strategy': 'best_execution'
            }
            
            # Get routing decision
            routing_decision = await self.smart_router.route_order(routing_request)
            
            # Execute with failover
            execution_result = await self.smart_router.execute_with_failover(
                routing_request, routing_decision
            )
            
            if execution_result['success']:
                # Create order slice record
                order_slice = OrderSlice(
                    id=f"{order_request.order_id}_slice_{slice_count}",
                    parent_order_id=order_request.order_id,
                    exchange=execution_result['exchange'],
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=slice_size,
                    status=SliceStatus.FILLED,
                    filled_quantity=Decimal(str(execution_result['result'].get('filled_quantity', slice_size))),
                    average_fill_price=Decimal(str(execution_result['result'].get('average_price', 0))),
                    exchange_order_id=execution_result['result'].get('order_id'),
                    submitted_at=datetime.utcnow(),
                    filled_at=datetime.utcnow(),
                    fees=Decimal(str(execution_result['result'].get('fees', 0)))
                )
                
                # Add to execution state
                self.active_executions[order_request.order_id]['slices'].append(order_slice)
                
                return {
                    'success': True,
                    'filled_quantity': order_slice.filled_quantity,
                    'average_price': order_slice.average_fill_price,
                    'fees': order_slice.fees,
                    'exchange': order_slice.exchange,
                    'slice': order_slice
                }
            else:
                return {
                    'success': False,
                    'error': execution_result.get('error', 'Unknown execution error'),
                    'filled_quantity': Decimal('0')
                }
        
        except Exception as e:
            logger.error("Order slice execution failed",
                        order_id=order_request.order_id,
                        slice_count=slice_count,
                        error=str(e))
            return {
                'success': False,
                'error': str(e),
                'filled_quantity': Decimal('0')
            }
    
    async def _execute_exchange_slice(self, order_request: AdvancedOrderRequest,
                                    exchange: str, slice_info: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a slice on a specific exchange."""
        try:
            exchange_client = self.exchange_clients[exchange]
            
            # Execute order directly on the exchange
            result = await exchange_client.place_order(
                symbol=order_request.symbol,
                side=order_request.side,
                type='market',
                amount=float(slice_info['quantity']),
                price=slice_info.get('price')
            )
            
            return {
                'success': True,
                'filled_quantity': Decimal(str(result.get('filled_quantity', slice_info['quantity']))),
                'average_price': Decimal(str(result.get('average_price', 0))),
                'fees': Decimal(str(result.get('fees', 0))),
                'exchange_order_id': result.get('order_id')
            }
        
        except Exception as e:
            logger.error("Exchange slice execution failed",
                        order_id=order_request.order_id,
                        exchange=exchange,
                        error=str(e))
            return {
                'success': False,
                'error': str(e),
                'filled_quantity': Decimal('0')
            }    
  
  # Helper Methods
    
    async def _validate_order(self, order_request: AdvancedOrderRequest) -> Dict[str, Any]:
        """Validate an advanced order request."""
        try:
            # Basic validation
            if not order_request.order_id:
                return {'valid': False, 'error': 'Order ID is required'}
            
            if not order_request.symbol:
                return {'valid': False, 'error': 'Symbol is required'}
            
            if order_request.side not in ['buy', 'sell']:
                return {'valid': False, 'error': 'Invalid order side'}
            
            # Type-specific validation
            if order_request.order_type == AdvancedOrderType.ICEBERG:
                config: IcebergOrderConfig = order_request.config
                if config.total_quantity <= 0:
                    return {'valid': False, 'error': 'Total quantity must be positive'}
                if config.visible_quantity <= 0:
                    return {'valid': False, 'error': 'Visible quantity must be positive'}
                if config.visible_quantity > config.total_quantity:
                    return {'valid': False, 'error': 'Visible quantity cannot exceed total quantity'}
            
            elif order_request.order_type == AdvancedOrderType.TWAP:
                config: TWAPConfig = order_request.config
                if config.total_quantity <= 0:
                    return {'valid': False, 'error': 'Total quantity must be positive'}
                if config.duration_minutes <= 0:
                    return {'valid': False, 'error': 'Duration must be positive'}
                if config.participation_rate <= 0 or config.participation_rate > 1:
                    return {'valid': False, 'error': 'Participation rate must be between 0 and 1'}
            
            elif order_request.order_type == AdvancedOrderType.SMART_SPLIT:
                config: SmartSplitConfig = order_request.config
                if config.total_quantity <= 0:
                    return {'valid': False, 'error': 'Total quantity must be positive'}
                if config.max_exchanges <= 0:
                    return {'valid': False, 'error': 'Max exchanges must be positive'}
            
            return {'valid': True}
        
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol."""
        try:
            if not self.market_data_service:
                return None
            
            # Get ticker data
            ticker = await self.market_data_service.get_ticker(symbol)
            
            # Get order book data
            order_book = await self.market_data_service.get_order_book(symbol, limit=20)
            
            # Get recent trades
            trades = await self.market_data_service.get_recent_trades(symbol, limit=100)
            
            return {
                'ticker': ticker,
                'order_book': order_book,
                'trades': trades,
                'timestamp': datetime.utcnow()
            }
        
        except Exception as e:
            logger.error("Failed to get market data", symbol=symbol, error=str(e))
            return None
    
    def _get_current_price(self, market_data: Dict[str, Any], side: str) -> Optional[Decimal]:
        """Get current price for order side."""
        try:
            ticker = market_data.get('ticker', {})
            if side == 'buy':
                return Decimal(str(ticker.get('ask', 0)))
            else:
                return Decimal(str(ticker.get('bid', 0)))
        except Exception:
            return None
    
    async def _should_wait_for_price_improvement(self, order_request: AdvancedOrderRequest,
                                               current_price: Decimal,
                                               config: IcebergOrderConfig) -> bool:
        """Check if we should wait for better price."""
        try:
            if not hasattr(config, 'price_improvement_threshold_bps'):
                return False
            
            # Get historical price data for comparison
            market_data = await self._get_market_data(order_request.symbol)
            if not market_data:
                return False
            
            trades = market_data.get('trades', [])
            if not trades:
                return False
            
            # Calculate recent average price
            recent_prices = [Decimal(str(trade.get('price', 0))) for trade in trades[-10:]]
            if not recent_prices:
                return False
            
            avg_price = sum(recent_prices) / len(recent_prices)
            
            # Calculate price improvement threshold
            threshold = config.price_improvement_threshold_bps / 10000  # Convert bps to decimal
            
            if order_request.side == 'buy':
                # For buy orders, wait if current price is significantly higher than average
                return (current_price - avg_price) / avg_price > threshold
            else:
                # For sell orders, wait if current price is significantly lower than average
                return (avg_price - current_price) / avg_price > threshold
        
        except Exception as e:
            logger.error("Error checking price improvement", error=str(e))
            return False
    
    async def _calculate_volume_adjustment(self, market_data: Dict[str, Any],
                                         participation_rate: float) -> Decimal:
        """Calculate volume adjustment factor based on market conditions."""
        try:
            trades = market_data.get('trades', [])
            if not trades:
                return Decimal('1.0')
            
            # Calculate recent volume
            recent_volume = sum(float(trade.get('amount', 0)) for trade in trades[-20:])
            
            # Calculate average volume per trade
            avg_volume_per_trade = recent_volume / len(trades[-20:]) if trades else 0
            
            # Adjust based on participation rate
            if avg_volume_per_trade > 0:
                max_volume = avg_volume_per_trade * participation_rate
                return Decimal(str(min(2.0, max(0.1, max_volume / avg_volume_per_trade))))
            
            return Decimal('1.0')
        
        except Exception as e:
            logger.error("Error calculating volume adjustment", error=str(e))
            return Decimal('1.0')
    
    async def _estimate_market_impact(self, symbol: str, side: str, quantity: Decimal) -> float:
        """Estimate market impact in basis points."""
        try:
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return 0.0
            
            order_book = market_data.get('order_book', {})
            if side == 'buy':
                asks = order_book.get('asks', [])
                if not asks:
                    return 0.0
                
                # Calculate how much of the order book we would consume
                cumulative_quantity = Decimal('0')
                weighted_price = Decimal('0')
                total_value = Decimal('0')
                
                for price_str, qty_str in asks[:10]:  # Look at top 10 levels
                    price = Decimal(str(price_str))
                    qty = Decimal(str(qty_str))
                    
                    if cumulative_quantity >= quantity:
                        break
                    
                    fill_qty = min(qty, quantity - cumulative_quantity)
                    value = fill_qty * price
                    total_value += value
                    cumulative_quantity += fill_qty
                
                if cumulative_quantity > 0:
                    avg_fill_price = total_value / cumulative_quantity
                    best_price = Decimal(str(asks[0][0]))
                    impact = ((avg_fill_price - best_price) / best_price) * 10000  # Convert to bps
                    return float(impact)
            
            else:  # sell
                bids = order_book.get('bids', [])
                if not bids:
                    return 0.0
                
                cumulative_quantity = Decimal('0')
                total_value = Decimal('0')
                
                for price_str, qty_str in bids[:10]:
                    price = Decimal(str(price_str))
                    qty = Decimal(str(qty_str))
                    
                    if cumulative_quantity >= quantity:
                        break
                    
                    fill_qty = min(qty, quantity - cumulative_quantity)
                    value = fill_qty * price
                    total_value += value
                    cumulative_quantity += fill_qty
                
                if cumulative_quantity > 0:
                    avg_fill_price = total_value / cumulative_quantity
                    best_price = Decimal(str(bids[0][0]))
                    impact = ((best_price - avg_fill_price) / best_price) * 10000  # Convert to bps
                    return float(impact)
            
            return 0.0
        
        except Exception as e:
            logger.error("Error estimating market impact", symbol=symbol, error=str(e))
            return 0.0
    
    async def _get_multi_exchange_market_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Get market data from all available exchanges."""
        try:
            all_data = {}
            
            for exchange_name, exchange_client in self.exchange_clients.items():
                try:
                    # Get ticker
                    ticker = await exchange_client.fetch_ticker(symbol)
                    
                    # Get order book
                    order_book = await exchange_client.fetch_order_book(symbol, limit=10)
                    
                    all_data[exchange_name] = {
                        'ticker': ticker,
                        'order_book': order_book,
                        'timestamp': datetime.utcnow()
                    }
                
                except Exception as e:
                    logger.warning("Failed to get market data from exchange",
                                 exchange=exchange_name, symbol=symbol, error=str(e))
                    continue
            
            return all_data
        
        except Exception as e:
            logger.error("Error getting multi-exchange market data", symbol=symbol, error=str(e))
            return {}
    
    async def _calculate_optimal_allocation(self, order_request: AdvancedOrderRequest,
                                          remaining_quantity: Decimal,
                                          all_market_data: Dict[str, Dict[str, Any]],
                                          config: SmartSplitConfig) -> Dict[str, Dict[str, Any]]:
        """Calculate optimal allocation across exchanges."""
        try:
            allocations = {}
            
            # Score each exchange
            exchange_scores = {}
            for exchange, market_data in all_market_data.items():
                score = await self._score_exchange(exchange, market_data, order_request, config)
                if score > 0:
                    exchange_scores[exchange] = score
            
            if not exchange_scores:
                return {}
            
            # Sort exchanges by score
            sorted_exchanges = sorted(exchange_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to max exchanges
            selected_exchanges = sorted_exchanges[:config.max_exchanges]
            
            # Calculate allocation weights
            total_score = sum(score for _, score in selected_exchanges)
            
            for exchange, score in selected_exchanges:
                weight = score / total_score
                allocation_quantity = remaining_quantity * Decimal(str(weight))
                
                # Ensure minimum slice size
                if allocation_quantity >= config.min_slice_size:
                    market_data = all_market_data[exchange]
                    ticker = market_data.get('ticker', {})
                    
                    allocations[exchange] = {
                        'quantity': allocation_quantity,
                        'weight': weight,
                        'score': score,
                        'price': ticker.get('ask' if order_request.side == 'buy' else 'bid')
                    }
            
            return allocations
        
        except Exception as e:
            logger.error("Error calculating optimal allocation", error=str(e))
            return {}
    
    async def _score_exchange(self, exchange: str, market_data: Dict[str, Any],
                            order_request: AdvancedOrderRequest,
                            config: SmartSplitConfig) -> float:
        """Score an exchange for allocation."""
        try:
            ticker = market_data.get('ticker', {})
            order_book = market_data.get('order_book', {})
            
            if not ticker or not order_book:
                return 0.0
            
            score = 0.0
            
            # Price competitiveness (40% weight)
            if order_request.side == 'buy':
                price = ticker.get('ask', 0)
                asks = order_book.get('asks', [])
                if asks and price > 0:
                    # Lower ask price is better for buy orders
                    price_score = 1.0 / price if price > 0 else 0
                    score += price_score * 0.4
            else:
                price = ticker.get('bid', 0)
                bids = order_book.get('bids', [])
                if bids and price > 0:
                    # Higher bid price is better for sell orders
                    score += price * 0.4
            
            # Liquidity (30% weight)
            if order_request.side == 'buy':
                asks = order_book.get('asks', [])
                liquidity = sum(float(qty) for _, qty in asks[:5]) if asks else 0
            else:
                bids = order_book.get('bids', [])
                liquidity = sum(float(qty) for _, qty in bids[:5]) if bids else 0
            
            if liquidity >= config.liquidity_threshold:
                score += min(liquidity / 1000, 1.0) * 0.3  # Normalize liquidity score
            
            # Spread (20% weight)
            ask = ticker.get('ask', 0)
            bid = ticker.get('bid', 0)
            if ask > 0 and bid > 0:
                spread = (ask - bid) / ((ask + bid) / 2)
                spread_score = max(0, 1.0 - spread * 100)  # Lower spread is better
                score += spread_score * 0.2
            
            # Volume (10% weight)
            volume = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0)
            if volume > 0:
                volume_score = min(volume / 1000000, 1.0)  # Normalize volume
                score += volume_score * 0.1
            
            return score
        
        except Exception as e:
            logger.error("Error scoring exchange", exchange=exchange, error=str(e))
            return 0.0
    
    async def _calculate_execution_metrics(self, order_id: str) -> None:
        """Calculate execution metrics for completed order."""
        try:
            execution_state = self.active_executions.get(order_id)
            if not execution_state:
                return
            
            slices = execution_state.get('slices', [])
            if not slices:
                return
            
            # Calculate metrics
            total_quantity = execution_state['order_request'].config.total_quantity
            filled_quantity = execution_state['filled_quantity']
            remaining_quantity = total_quantity - filled_quantity
            
            # Calculate average price
            total_value = sum(slice.filled_quantity * (slice.average_fill_price or Decimal('0')) 
                            for slice in slices if slice.average_fill_price)
            average_price = total_value / filled_quantity if filled_quantity > 0 else Decimal('0')
            
            # Calculate total fees
            total_fees = sum(slice.fees for slice in slices)
            
            # Calculate execution time
            start_time = execution_state['start_time']
            end_time = execution_state['last_update']
            execution_time_seconds = (end_time - start_time).total_seconds()
            
            # Count slices
            slices_count = len(slices)
            successful_slices = len([s for s in slices if s.status == SliceStatus.FILLED])
            failed_slices = slices_count - successful_slices
            
            # Estimate market impact and implementation shortfall
            market_impact_bps = 0.0  # Simplified - would need benchmark price
            implementation_shortfall_bps = 0.0  # Simplified
            participation_rate = 0.0  # Simplified
            
            metrics = ExecutionMetrics(
                total_quantity=total_quantity,
                filled_quantity=filled_quantity,
                remaining_quantity=remaining_quantity,
                average_price=average_price,
                total_fees=total_fees,
                execution_time_seconds=execution_time_seconds,
                slices_count=slices_count,
                successful_slices=successful_slices,
                failed_slices=failed_slices,
                market_impact_bps=market_impact_bps,
                implementation_shortfall_bps=implementation_shortfall_bps,
                participation_rate=participation_rate
            )
            
            execution_state['metrics'] = metrics
            self.execution_metrics[order_id] = metrics
            
            logger.info("Execution metrics calculated",
                       order_id=order_id,
                       filled_quantity=float(filled_quantity),
                       average_price=float(average_price),
                       total_fees=float(total_fees),
                       execution_time=execution_time_seconds)
        
        except Exception as e:
            logger.error("Error calculating execution metrics", order_id=order_id, error=str(e))
    
    # Management Methods
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an advanced order."""
        execution_state = self.active_executions.get(order_id)
        if not execution_state:
            return {
                'success': False,
                'error': 'Order not found',
                'order_id': order_id
            }
        
        return {
            'success': True,
            'order_id': order_id,
            'status': execution_state['status'].value,
            'filled_quantity': float(execution_state['filled_quantity']),
            'total_fees': float(execution_state['total_fees']),
            'slices_count': len(execution_state['slices']),
            'start_time': execution_state['start_time'].isoformat(),
            'last_update': execution_state['last_update'].isoformat(),
            'metrics': execution_state.get('metrics')
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an active advanced order."""
        try:
            execution_state = self.active_executions.get(order_id)
            if not execution_state:
                return {
                    'success': False,
                    'error': 'Order not found',
                    'order_id': order_id
                }
            
            if execution_state['status'] in [ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED, ExecutionStatus.FAILED]:
                return {
                    'success': False,
                    'error': f'Order already in final state: {execution_state["status"].value}',
                    'order_id': order_id
                }
            
            # Cancel the execution task
            task = self.execution_tasks.get(order_id)
            if task and not task.done():
                task.cancel()
            
            # Update status
            execution_state['status'] = ExecutionStatus.CANCELLED
            execution_state['last_update'] = datetime.utcnow()
            
            # Calculate final metrics
            await self._calculate_execution_metrics(order_id)
            
            logger.info("Advanced order cancelled", order_id=order_id)
            
            return {
                'success': True,
                'order_id': order_id,
                'status': ExecutionStatus.CANCELLED.value,
                'message': 'Order cancelled successfully'
            }
        
        except Exception as e:
            logger.error("Error cancelling order", order_id=order_id, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def list_active_orders(self) -> Dict[str, Any]:
        """List all active advanced orders."""
        try:
            active_orders = []
            
            for order_id, execution_state in self.active_executions.items():
                if execution_state['status'] in [ExecutionStatus.ACTIVE, ExecutionStatus.PARTIALLY_FILLED]:
                    order_info = {
                        'order_id': order_id,
                        'symbol': execution_state['order_request'].symbol,
                        'side': execution_state['order_request'].side,
                        'order_type': execution_state['order_request'].order_type.value,
                        'status': execution_state['status'].value,
                        'filled_quantity': float(execution_state['filled_quantity']),
                        'start_time': execution_state['start_time'].isoformat(),
                        'slices_count': len(execution_state['slices'])
                    }
                    active_orders.append(order_info)
            
            return {
                'success': True,
                'active_orders': active_orders,
                'count': len(active_orders)
            }
        
        except Exception as e:
            logger.error("Error listing active orders", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'active_orders': [],
                'count': 0
            }
    
    async def get_execution_metrics(self, order_id: str) -> Dict[str, Any]:
        """Get detailed execution metrics for an order."""
        try:
            metrics = self.execution_metrics.get(order_id)
            if not metrics:
                return {
                    'success': False,
                    'error': 'Metrics not found for order',
                    'order_id': order_id
                }
            
            return {
                'success': True,
                'order_id': order_id,
                'metrics': {
                    'total_quantity': float(metrics.total_quantity),
                    'filled_quantity': float(metrics.filled_quantity),
                    'remaining_quantity': float(metrics.remaining_quantity),
                    'average_price': float(metrics.average_price),
                    'total_fees': float(metrics.total_fees),
                    'execution_time_seconds': metrics.execution_time_seconds,
                    'slices_count': metrics.slices_count,
                    'successful_slices': metrics.successful_slices,
                    'failed_slices': metrics.failed_slices,
                    'market_impact_bps': metrics.market_impact_bps,
                    'implementation_shortfall_bps': metrics.implementation_shortfall_bps,
                    'participation_rate': metrics.participation_rate
                }
            }
        
        except Exception as e:
            logger.error("Error getting execution metrics", order_id=order_id, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }


# Factory function for creating advanced order executor
def create_advanced_order_executor(smart_router, market_data_service, exchange_clients):
    """Create an advanced order executor instance."""
    return AdvancedOrderExecutor(smart_router, market_data_service, exchange_clients)


# Example usage and testing functions
async def example_iceberg_order():
    """Example of creating an iceberg order."""
    config = IcebergOrderConfig(
        total_quantity=Decimal('10.0'),
        visible_quantity=Decimal('1.0'),
        price_improvement_threshold_bps=5.0,
        max_slices=20,
        slice_interval_seconds=60.0,
        randomize_timing=True,
        randomize_quantity=True
    )
    
    order_request = AdvancedOrderRequest(
        order_id="iceberg_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.ICEBERG,
        config=config,
        user_id="user_123",
        strategy_id="strategy_456"
    )
    
    return order_request


async def example_twap_order():
    """Example of creating a TWAP order."""
    config = TWAPConfig(
        total_quantity=Decimal('5.0'),
        duration_minutes=120,  # 2 hours
        slice_interval_minutes=10.0,
        participation_rate=0.15,
        adaptive_sizing=True,
        market_impact_threshold_bps=25.0
    )
    
    order_request = AdvancedOrderRequest(
        order_id="twap_001",
        symbol="ETH/USDT",
        side="sell",
        order_type=AdvancedOrderType.TWAP,
        config=config,
        user_id="user_123",
        strategy_id="strategy_789"
    )
    
    return order_request


async def example_smart_split_order():
    """Example of creating a smart split order."""
    config = SmartSplitConfig(
        total_quantity=Decimal('20.0'),
        max_exchanges=3,
        min_slice_size=Decimal('0.1'),
        liquidity_threshold=0.05,
        rebalance_interval_seconds=120.0,
        cost_optimization=True
    )
    
    order_request = AdvancedOrderRequest(
        order_id="split_001",
        symbol="BTC/USDT",
        side="buy",
        order_type=AdvancedOrderType.SMART_SPLIT,
        config=config,
        user_id="user_123",
        strategy_id="strategy_101"
    )
    
    return order_request


if __name__ == "__main__":
    # This would be used for testing the module
    print("Advanced Order Types and Execution Engine")
    print("Supports: Iceberg Orders, TWAP Execution, Smart Order Splitting")