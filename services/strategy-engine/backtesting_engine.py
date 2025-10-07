"""
Advanced Backtesting Engine for Trading Strategy Framework.
Implements historical data replay, realistic execution simulation, and comprehensive performance metrics.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import math
import warnings
from abc import ABC, abstractmethod
import copy
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("backtesting-engine")

class OrderType(str, Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ExecutionModel(str, Enum):
    """Execution models for backtesting."""
    PERFECT = "perfect"  # No slippage, instant execution
    REALISTIC = "realistic"  # With slippage and fees
    CONSERVATIVE = "conservative"  # Higher slippage estimates
    AGGRESSIVE = "aggressive"  # Lower slippage estimates

@dataclass
class MarketData:
    """Market data point for backtesting."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

@dataclass
class Order:
    """Order representation for backtesting."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Executed trade representation."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fees: float
    slippage: float
    order_id: str
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position tracking for backtesting."""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_price: float
    timestamp: datetime
    trades: List[Trade] = field(default_factory=list)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    ulcer_index: float
    var_95: float
    expected_shortfall: float
    kelly_criterion: float
    start_date: datetime
    end_date: datetime
    duration_days: int

class BacktestingConfig(BaseModel):
    """Configuration for backtesting engine."""
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate (0.1%)")
    slippage_model: str = Field(default="linear", description="Slippage model")
    slippage_rate: float = Field(default=0.0005, description="Base slippage rate")
    execution_model: ExecutionModel = Field(default=ExecutionModel.REALISTIC, description="Execution model")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate for Sharpe ratio")
    benchmark_symbol: Optional[str] = Field(default=None, description="Benchmark symbol")
    max_position_size: float = Field(default=1.0, description="Maximum position size (fraction of capital)")
    margin_requirement: float = Field(default=1.0, description="Margin requirement")
    interest_rate: float = Field(default=0.05, description="Interest rate for margin")

class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.signals = []
    
    @abstractmethod
    async def generate_signals(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Generate trading signals based on market data."""
        pass
    
    @abstractmethod
    async def on_market_data(self, data: MarketData) -> Optional[Order]:
        """Process market data and potentially generate orders."""
        pass
    
    async def on_order_filled(self, order: Order, trade: Trade) -> None:
        """Handle order fill events."""
        pass
    
    async def on_position_update(self, position: Position) -> None:
        """Handle position updates."""
        pass

class BacktestingEngine:
    """Advanced backtesting engine with realistic execution simulation."""
    
    def __init__(self, config: BacktestingConfig = None):
        self.config = config or BacktestingConfig()
        self.current_time = None
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.benchmark_curve = []
        self.order_id_counter = 0
        self.trade_id_counter = 0
        
        logger.info("Backtesting Engine initialized", config=self.config.model_dump())
    
    async def run_backtest(self, 
                          strategy: TradingStrategy,
                          market_data: List[MarketData],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Run comprehensive backtest with realistic execution simulation."""
        try:
            # Filter data by date range
            if start_date or end_date:
                market_data = self._filter_data_by_date(market_data, start_date, end_date)
            
            if not market_data:
                raise ValueError("No market data available for backtesting")
            
            # Initialize backtest
            self._initialize_backtest(strategy, market_data[0].timestamp)
            
            # Process each data point
            for i, data_point in enumerate(market_data):
                await self._process_market_data(strategy, data_point, i)
            
            # Finalize backtest
            await self._finalize_backtest(strategy, market_data[-1])
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                market_data[0].timestamp, market_data[-1].timestamp
            )
            
            # Generate backtest report
            backtest_report = {
                'strategy_name': strategy.name,
                'parameters': strategy.parameters,
                'performance_metrics': performance_metrics,
                'equity_curve': self.equity_curve,
                'drawdown_curve': self.drawdown_curve,
                'trades': [self._trade_to_dict(trade) for trade in self.trades],
                'positions': {symbol: self._position_to_dict(pos) for symbol, pos in self.positions.items()},
                'total_orders': len(self.orders),
                'total_trades': len(self.trades),
                'final_capital': self.current_capital,
                'config': self.config.model_dump()
            }
            
            logger.info("Backtest completed successfully",
                       strategy=strategy.name,
                       total_return=performance_metrics.total_return,
                       sharpe_ratio=performance_metrics.sharpe_ratio,
                       max_drawdown=performance_metrics.max_drawdown,
                       total_trades=performance_metrics.total_trades)
            
            return backtest_report
            
        except Exception as e:
            logger.error("Backtest failed", error=str(e), strategy=strategy.name)
            raise
    
    async def _process_market_data(self, strategy: TradingStrategy, data: MarketData, index: int):
        """Process single market data point."""
        self.current_time = data.timestamp
        
        # Update positions with current prices
        await self._update_positions(data)
        
        # Process pending orders
        await self._process_pending_orders(data)
        
        # Generate new signals/orders from strategy
        new_order = await strategy.on_market_data(data)
        if new_order:
            await self._submit_order(new_order)
        
        # Update equity curve
        current_equity = await self._calculate_current_equity()
        self.equity_curve.append({
            'timestamp': data.timestamp,
            'equity': current_equity,
            'cash': self.current_capital,
            'positions_value': current_equity - self.current_capital
        })
        
        # Update drawdown curve
        if self.equity_curve:
            peak_equity = max(point['equity'] for point in self.equity_curve)
            current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            self.drawdown_curve.append({
                'timestamp': data.timestamp,
                'drawdown': current_drawdown,
                'peak_equity': peak_equity
            })
    
    async def _process_pending_orders(self, data: MarketData):
        """Process pending orders against current market data."""
        filled_orders = []
        
        for order in self.orders:
            if order.status == OrderStatus.PENDING and order.symbol == data.symbol:
                fill_result = await self._try_fill_order(order, data)
                if fill_result:
                    filled_orders.append(order)
        
        # Remove filled orders
        for order in filled_orders:
            if order in self.orders:
                self.orders.remove(order)
    
    async def _try_fill_order(self, order: Order, data: MarketData) -> bool:
        """Attempt to fill an order based on market data."""
        try:
            fill_price = None
            
            # Determine fill price based on order type
            if order.order_type == OrderType.MARKET:
                fill_price = self._get_market_fill_price(order, data)
            elif order.order_type == OrderType.LIMIT:
                fill_price = self._get_limit_fill_price(order, data)
            elif order.order_type == OrderType.STOP:
                fill_price = self._get_stop_fill_price(order, data)
            elif order.order_type == OrderType.STOP_LIMIT:
                fill_price = self._get_stop_limit_fill_price(order, data)
            
            if fill_price is not None:
                # Apply execution model (slippage and fees)
                execution_result = await self._apply_execution_model(order, fill_price, data)
                
                # Create trade
                trade = Trade(
                    id=f"trade_{self.trade_id_counter}",
                    timestamp=data.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    price=execution_result['final_price'],
                    fees=execution_result['fees'],
                    slippage=execution_result['slippage'],
                    order_id=order.id
                )
                self.trade_id_counter += 1
                
                # Update order status
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.filled_price = execution_result['final_price']
                order.fees = execution_result['fees']
                order.slippage = execution_result['slippage']
                
                # Execute trade
                await self._execute_trade(trade)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("Order fill failed", error=str(e), order_id=order.id)
            order.status = OrderStatus.REJECTED
            return False
    
    def _get_market_fill_price(self, order: Order, data: MarketData) -> float:
        """Get fill price for market order."""
        if order.side == OrderSide.BUY:
            return data.ask if data.ask else data.close
        else:
            return data.bid if data.bid else data.close
    
    def _get_limit_fill_price(self, order: Order, data: MarketData) -> Optional[float]:
        """Get fill price for limit order."""
        if order.side == OrderSide.BUY:
            # Buy limit: fill if market price <= limit price
            if data.low <= order.price:
                return min(order.price, data.open)
        else:
            # Sell limit: fill if market price >= limit price
            if data.high >= order.price:
                return max(order.price, data.open)
        return None
    
    def _get_stop_fill_price(self, order: Order, data: MarketData) -> Optional[float]:
        """Get fill price for stop order."""
        if order.side == OrderSide.BUY:
            # Buy stop: fill if market price >= stop price
            if data.high >= order.stop_price:
                return max(order.stop_price, data.open)
        else:
            # Sell stop: fill if market price <= stop price
            if data.low <= order.stop_price:
                return min(order.stop_price, data.open)
        return None
    
    def _get_stop_limit_fill_price(self, order: Order, data: MarketData) -> Optional[float]:
        """Get fill price for stop-limit order."""
        # First check if stop is triggered
        stop_triggered = False
        if order.side == OrderSide.BUY and data.high >= order.stop_price:
            stop_triggered = True
        elif order.side == OrderSide.SELL and data.low <= order.stop_price:
            stop_triggered = True
        
        if stop_triggered:
            # Then check if limit can be filled
            return self._get_limit_fill_price(order, data)
        
        return None
    
    async def _apply_execution_model(self, order: Order, base_price: float, data: MarketData) -> Dict[str, float]:
        """Apply execution model to calculate final price, slippage, and fees."""
        if self.config.execution_model == ExecutionModel.PERFECT:
            return {
                'final_price': base_price,
                'slippage': 0.0,
                'fees': 0.0
            }
        
        # Calculate slippage
        slippage = await self._calculate_slippage(order, base_price, data)
        
        # Apply slippage to price
        if order.side == OrderSide.BUY:
            final_price = base_price * (1 + slippage)
        else:
            final_price = base_price * (1 - slippage)
        
        # Calculate fees
        fees = self._calculate_fees(order.quantity, final_price)
        
        return {
            'final_price': final_price,
            'slippage': slippage,
            'fees': fees
        }
    
    async def _calculate_slippage(self, order: Order, price: float, data: MarketData) -> float:
        """Calculate slippage based on order size and market conditions."""
        base_slippage = self.config.slippage_rate
        
        # Adjust for execution model
        if self.config.execution_model == ExecutionModel.CONSERVATIVE:
            base_slippage *= 2.0
        elif self.config.execution_model == ExecutionModel.AGGRESSIVE:
            base_slippage *= 0.5
        
        # Adjust for order size (larger orders have more slippage)
        order_value = order.quantity * price
        size_multiplier = 1.0 + (order_value / self.current_capital) * 0.5
        
        # Adjust for volatility (higher volatility = more slippage)
        if hasattr(data, 'volatility'):
            volatility_multiplier = 1.0 + data.volatility * 0.1
        else:
            # Estimate volatility from high-low range
            volatility_estimate = (data.high - data.low) / data.close
            volatility_multiplier = 1.0 + volatility_estimate * 0.1
        
        # Adjust for spread if available
        spread_multiplier = 1.0
        if data.bid and data.ask:
            spread = (data.ask - data.bid) / ((data.ask + data.bid) / 2)
            spread_multiplier = 1.0 + spread * 0.5
        
        total_slippage = base_slippage * size_multiplier * volatility_multiplier * spread_multiplier
        
        return min(total_slippage, 0.01)  # Cap at 1%
    
    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees."""
        trade_value = quantity * price
        return trade_value * self.config.commission_rate
    
    async def _execute_trade(self, trade: Trade):
        """Execute a trade and update positions."""
        # Update capital
        if trade.side == OrderSide.BUY:
            self.current_capital -= (trade.quantity * trade.price + trade.fees)
        else:
            self.current_capital += (trade.quantity * trade.price - trade.fees)
        
        # Update position
        await self._update_position(trade)
        
        # Add to trades list
        self.trades.append(trade)
        
        logger.debug("Trade executed",
                    symbol=trade.symbol,
                    side=trade.side.value,
                    quantity=trade.quantity,
                    price=trade.price,
                    fees=trade.fees)
    
    async def _update_position(self, trade: Trade):
        """Update position based on executed trade."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                last_price=trade.price,
                timestamp=trade.timestamp,
                trades=[]
            )
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Calculate new average price
            total_cost = position.quantity * position.average_price + trade.quantity * trade.price
            new_quantity = position.quantity + trade.quantity
            
            if new_quantity > 0:
                position.average_price = total_cost / new_quantity
            
            position.quantity = new_quantity
        else:
            # Selling - calculate realized P&L
            if position.quantity > 0:
                realized_pnl = trade.quantity * (trade.price - position.average_price)
                position.realized_pnl += realized_pnl
                trade.pnl = realized_pnl
            
            position.quantity -= trade.quantity
            
            # If position is closed, reset average price
            if position.quantity <= 0:
                position.quantity = 0
                position.average_price = 0
        
        position.last_price = trade.price
        position.timestamp = trade.timestamp
        position.trades.append(trade)
    
    async def _update_positions(self, data: MarketData):
        """Update all positions with current market data."""
        if data.symbol in self.positions:
            position = self.positions[data.symbol]
            position.last_price = data.close
            
            # Calculate unrealized P&L
            if position.quantity > 0:
                position.unrealized_pnl = position.quantity * (data.close - position.average_price)
            else:
                position.unrealized_pnl = 0.0
            
            position.timestamp = data.timestamp
    
    async def _submit_order(self, order: Order):
        """Submit an order to the backtesting engine."""
        order.id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1
        
        # Validate order
        if not await self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return
        
        self.orders.append(order)
        logger.debug("Order submitted", order_id=order.id, symbol=order.symbol, side=order.side.value)
    
    async def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        # Check if we have enough capital for buy orders
        if order.side == OrderSide.BUY:
            estimated_cost = order.quantity * (order.price or 0) * 1.01  # Add buffer for slippage/fees
            if estimated_cost > self.current_capital:
                logger.warning("Insufficient capital for order", order_id=order.id, required=estimated_cost, available=self.current_capital)
                return False
        
        # Check if we have enough position for sell orders
        if order.side == OrderSide.SELL:
            current_position = self.positions.get(order.symbol, Position(order.symbol, 0, 0, 0, 0, 0, datetime.now())).quantity
            if order.quantity > current_position:
                logger.warning("Insufficient position for sell order", order_id=order.id, required=order.quantity, available=current_position)
                return False
        
        return True
    
    async def _calculate_current_equity(self) -> float:
        """Calculate current total equity."""
        total_equity = self.current_capital
        
        for position in self.positions.values():
            if position.quantity > 0:
                position_value = position.quantity * position.last_price
                total_equity += position_value
        
        return total_equity
    
    def _filter_data_by_date(self, market_data: List[MarketData], start_date: Optional[datetime], end_date: Optional[datetime]) -> List[MarketData]:
        """Filter market data by date range."""
        filtered_data = market_data
        
        if start_date:
            filtered_data = [d for d in filtered_data if d.timestamp >= start_date]
        
        if end_date:
            filtered_data = [d for d in filtered_data if d.timestamp <= end_date]
        
        return filtered_data
    
    def _initialize_backtest(self, strategy: TradingStrategy, start_time: datetime):
        """Initialize backtest state."""
        self.current_time = start_time
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.order_id_counter = 0
        self.trade_id_counter = 0
        
        # Initialize strategy
        strategy.positions = {}
        strategy.orders = []
        strategy.trades = []
        strategy.equity_curve = []
        strategy.signals = []
    
    async def _finalize_backtest(self, strategy: TradingStrategy, end_data: MarketData):
        """Finalize backtest by closing all positions."""
        # Close all open positions at market price
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                # Create market sell order to close position
                close_order = Order(
                    id=f"close_order_{self.order_id_counter}",
                    timestamp=end_data.timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity
                )
                self.order_id_counter += 1
                
                # Execute the closing trade
                close_trade = Trade(
                    id=f"close_trade_{self.trade_id_counter}",
                    timestamp=end_data.timestamp,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    price=end_data.close,
                    fees=self._calculate_fees(position.quantity, end_data.close),
                    slippage=0.0,  # No slippage for final close
                    order_id=close_order.id
                )
                self.trade_id_counter += 1
                
                await self._execute_trade(close_trade)
    
    async def _calculate_performance_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            raise ValueError("No equity curve data available for performance calculation")
        
        # Extract equity values
        equity_values = [point['equity'] for point in self.equity_curve]
        returns = []
        
        # Calculate returns
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                returns.append((equity_values[i] - equity_values[i-1]) / equity_values[i-1])
            else:
                returns.append(0.0)
        
        returns_array = np.array(returns)
        
        # Basic metrics
        initial_capital = self.config.initial_capital
        final_equity = equity_values[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Time-based metrics
        duration = end_date - start_date
        duration_days = duration.days
        duration_years = duration_days / 365.25
        
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0
        
        # Sharpe ratio
        excess_returns = returns_array - (self.config.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 1 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown metrics
        peak_equity = initial_capital
        max_drawdown = 0.0
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for equity in equity_values:
            if equity > peak_equity:
                peak_equity = equity
                current_drawdown_duration = 0
            else:
                current_drawdown = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, current_drawdown)
                current_drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade-based metrics
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        total_trades = len(self.trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win * winning_trades_count / (avg_loss * losing_trades_count)) if avg_loss > 0 and losing_trades_count > 0 else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in self.trades:
            if trade.pnl and trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif trade.pnl and trade.pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Recovery factor
        total_profit = sum([t.pnl for t in self.trades if t.pnl and t.pnl > 0])
        recovery_factor = total_profit / (max_drawdown * initial_capital) if max_drawdown > 0 else 0
        
        # Ulcer Index
        squared_drawdowns = []
        for equity in equity_values:
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            squared_drawdowns.append(drawdown ** 2)
        
        ulcer_index = np.sqrt(np.mean(squared_drawdowns)) if squared_drawdowns else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns_array[returns_array <= var_95]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0
        
        # Kelly Criterion
        if win_rate > 0 and avg_loss > 0:
            kelly_criterion = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        else:
            kelly_criterion = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades_count,
            losing_trades=losing_trades_count,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            kelly_criterion=kelly_criterion,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days
        )
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'id': trade.id,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'side': trade.side.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'fees': trade.fees,
            'slippage': trade.slippage,
            'order_id': trade.order_id,
            'pnl': trade.pnl,
            'metadata': trade.metadata
        }
    
    def _position_to_dict(self, position: Position) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'symbol': position.symbol,
            'quantity': position.quantity,
            'average_price': position.average_price,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'last_price': position.last_price,
            'timestamp': position.timestamp.isoformat(),
            'trades_count': len(position.trades)
        }


class WalkForwardAnalyzer:
    """Walk-forward analysis for robust strategy testing."""
    
    def __init__(self, 
                 training_period_days: int = 252,  # 1 year
                 testing_period_days: int = 63,    # 3 months
                 step_days: int = 21):             # 3 weeks
        self.training_period_days = training_period_days
        self.testing_period_days = testing_period_days
        self.step_days = step_days
    
    async def run_walk_forward_analysis(self,
                                      strategy_class: type,
                                      strategy_parameters: Dict[str, Any],
                                      market_data: List[MarketData],
                                      config: BacktestingConfig = None) -> Dict[str, Any]:
        """Run walk-forward analysis on a strategy."""
        config = config or BacktestingConfig()
        
        # Sort data by timestamp
        market_data.sort(key=lambda x: x.timestamp)
        
        # Generate analysis periods
        periods = self._generate_periods(market_data)
        
        results = []
        
        for i, period in enumerate(periods):
            logger.info(f"Running walk-forward period {i+1}/{len(periods)}",
                       training_start=period['training_start'].isoformat(),
                       training_end=period['training_end'].isoformat(),
                       testing_start=period['testing_start'].isoformat(),
                       testing_end=period['testing_end'].isoformat())
            
            # Get training and testing data
            training_data = self._filter_data_by_period(
                market_data, period['training_start'], period['training_end']
            )
            testing_data = self._filter_data_by_period(
                market_data, period['testing_start'], period['testing_end']
            )
            
            if not training_data or not testing_data:
                logger.warning(f"Insufficient data for period {i+1}, skipping")
                continue
            
            # Train strategy (if it supports training)
            strategy = strategy_class("walk_forward_strategy", strategy_parameters)
            if hasattr(strategy, 'train'):
                await strategy.train(training_data)
            
            # Test strategy
            backtesting_engine = BacktestingEngine(config)
            backtest_result = await backtesting_engine.run_backtest(
                strategy, testing_data
            )
            
            period_result = {
                'period': i + 1,
                'training_period': {
                    'start': period['training_start'],
                    'end': period['training_end'],
                    'data_points': len(training_data)
                },
                'testing_period': {
                    'start': period['testing_start'],
                    'end': period['testing_end'],
                    'data_points': len(testing_data)
                },
                'performance': backtest_result['performance_metrics'],
                'trades': len(backtest_result['trades']),
                'final_capital': backtest_result['final_capital']
            }
            
            results.append(period_result)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_walk_forward_results(results)
        
        return {
            'strategy_name': strategy_class.__name__,
            'parameters': strategy_parameters,
            'walk_forward_config': {
                'training_period_days': self.training_period_days,
                'testing_period_days': self.testing_period_days,
                'step_days': self.step_days
            },
            'periods': results,
            'aggregated_metrics': aggregated_metrics,
            'total_periods': len(results)
        }
    
    def _generate_periods(self, market_data: List[MarketData]) -> List[Dict[str, datetime]]:
        """Generate walk-forward analysis periods."""
        if not market_data:
            return []
        
        start_date = market_data[0].timestamp
        end_date = market_data[-1].timestamp
        
        periods = []
        current_date = start_date
        
        while current_date + timedelta(days=self.training_period_days + self.testing_period_days) <= end_date:
            training_start = current_date
            training_end = current_date + timedelta(days=self.training_period_days)
            testing_start = training_end
            testing_end = testing_start + timedelta(days=self.testing_period_days)
            
            periods.append({
                'training_start': training_start,
                'training_end': training_end,
                'testing_start': testing_start,
                'testing_end': testing_end
            })
            
            current_date += timedelta(days=self.step_days)
        
        return periods
    
    def _filter_data_by_period(self, market_data: List[MarketData], start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Filter market data by period."""
        return [d for d in market_data if start_date <= d.timestamp <= end_date]
    
    def _aggregate_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results."""
        if not results:
            return {}
        
        # Extract performance metrics
        total_returns = [r['performance'].total_return for r in results]
        sharpe_ratios = [r['performance'].sharpe_ratio for r in results]
        max_drawdowns = [r['performance'].max_drawdown for r in results]
        win_rates = [r['performance'].win_rate for r in results]
        
        return {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': max(max_drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'consistency_score': len([r for r in total_returns if r > 0]) / len(total_returns),
            'total_periods': len(results)
        }


class HistoricalDataReplay:
    """Historical data replay system for backtesting."""
    
    def __init__(self, data_source: str = "file"):
        self.data_source = data_source
        self.data_cache = {}
    
    async def load_historical_data(self, 
                                 symbol: str,
                                 start_date: datetime,
                                 end_date: datetime,
                                 timeframe: str = "1h") -> List[MarketData]:
        """Load historical market data for backtesting."""
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Generate synthetic data for demonstration
        # In production, this would load from database or external API
        data = await self._generate_synthetic_data(symbol, start_date, end_date, timeframe)
        
        self.data_cache[cache_key] = data
        return data
    
    async def _generate_synthetic_data(self,
                                     symbol: str,
                                     start_date: datetime,
                                     end_date: datetime,
                                     timeframe: str) -> List[MarketData]:
        """Generate synthetic market data for testing."""
        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        
        # Generate data points
        data_points = []
        current_time = start_date
        current_price = 50000.0  # Starting price
        
        while current_time <= end_date:
            # Simple random walk with some trend and volatility
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            current_price *= (1 + price_change)
            
            # Generate OHLCV data
            volatility = abs(price_change) * current_price
            high = current_price + np.random.uniform(0, volatility)
            low = current_price - np.random.uniform(0, volatility)
            open_price = current_price + np.random.uniform(-volatility/2, volatility/2)
            close_price = current_price
            volume = np.random.uniform(1000, 10000)
            
            # Calculate bid/ask spread
            spread_pct = 0.001  # 0.1% spread
            bid = close_price * (1 - spread_pct/2)
            ask = close_price * (1 + spread_pct/2)
            
            data_point = MarketData(
                timestamp=current_time,
                symbol=symbol,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                bid=bid,
                ask=ask,
                spread=ask - bid
            )
            
            data_points.append(data_point)
            current_time += timedelta(minutes=timeframe_minutes)
        
        return data_points
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return timeframe_map.get(timeframe, 60)


# Example strategy implementations for testing
class SimpleMovingAverageStrategy(TradingStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.short_window = parameters.get('short_window', 10)
        self.long_window = parameters.get('long_window', 30)
        self.price_history = []
        self.position_size = parameters.get('position_size', 0.1)
    
    async def generate_signals(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Generate signals based on moving average crossover."""
        signals = []
        
        for i, data in enumerate(market_data):
            self.price_history.append(data.close)
            
            if len(self.price_history) >= self.long_window:
                short_ma = np.mean(self.price_history[-self.short_window:])
                long_ma = np.mean(self.price_history[-self.long_window:])
                
                # Previous MAs for crossover detection
                if len(self.price_history) > self.long_window:
                    prev_short_ma = np.mean(self.price_history[-self.short_window-1:-1])
                    prev_long_ma = np.mean(self.price_history[-self.long_window-1:-1])
                    
                    # Bullish crossover
                    if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                        signals.append({
                            'timestamp': data.timestamp,
                            'symbol': data.symbol,
                            'signal': 'BUY',
                            'strength': 1.0,
                            'price': data.close,
                            'metadata': {
                                'short_ma': short_ma,
                                'long_ma': long_ma,
                                'crossover': 'bullish'
                            }
                        })
                    
                    # Bearish crossover
                    elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                        signals.append({
                            'timestamp': data.timestamp,
                            'symbol': data.symbol,
                            'signal': 'SELL',
                            'strength': 1.0,
                            'price': data.close,
                            'metadata': {
                                'short_ma': short_ma,
                                'long_ma': long_ma,
                                'crossover': 'bearish'
                            }
                        })
        
        return signals
    
    async def on_market_data(self, data: MarketData) -> Optional[Order]:
        """Process market data and generate orders."""
        self.price_history.append(data.close)
        
        if len(self.price_history) < self.long_window:
            return None
        
        short_ma = np.mean(self.price_history[-self.short_window:])
        long_ma = np.mean(self.price_history[-self.long_window:])
        
        # Check for crossover
        if len(self.price_history) > self.long_window:
            prev_short_ma = np.mean(self.price_history[-self.short_window-1:-1])
            prev_long_ma = np.mean(self.price_history[-self.long_window-1:-1])
            
            current_position = self.positions.get(data.symbol, Position(data.symbol, 0, 0, 0, 0, 0, datetime.now())).quantity
            
            # Bullish crossover - buy signal
            if short_ma > long_ma and prev_short_ma <= prev_long_ma and current_position <= 0:
                quantity = self.position_size  # Fixed position size
                return Order(
                    id="",  # Will be set by backtesting engine
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    metadata={'signal': 'ma_crossover_bullish'}
                )
            
            # Bearish crossover - sell signal
            elif short_ma < long_ma and prev_short_ma >= prev_long_ma and current_position > 0:
                return Order(
                    id="",  # Will be set by backtesting engine
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=current_position,
                    metadata={'signal': 'ma_crossover_bearish'}
                )
        
        return None