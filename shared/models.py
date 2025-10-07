"""
Shared data models across all microservices.
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


# Enums
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExchangeName(str, Enum):
    BITUNIX = "BITUNIX"
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"


class StrategyType(str, Enum):
    COPY_TRADING = "COPY_TRADING"
    FUTURES_TRADING = "FUTURES_TRADING"
    ARBITRAGE = "ARBITRAGE"
    MARKET_MAKING = "MARKET_MAKING"
    DCA = "DCA"


class SignalType(str, Enum):
    SENTIMENT = "SENTIMENT"
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    ARBITRAGE = "ARBITRAGE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Base Models
class BaseEntity(BaseModel):
    """Base model with common fields"""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# Trading Models
class Symbol(BaseModel):
    """Trading symbol information"""
    symbol: str
    base_asset: str
    quote_asset: str
    exchange: ExchangeName
    min_quantity: Decimal
    max_quantity: Decimal
    quantity_precision: int
    price_precision: int
    active: bool = True


class Ticker(BaseModel):
    """Market ticker data"""
    symbol: str
    exchange: ExchangeName
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    change_24h_pct: Decimal
    timestamp: datetime


class OrderBook(BaseModel):
    """Order book data"""
    symbol: str
    exchange: ExchangeName
    bids: List[List[Decimal]]  # [price, quantity]
    asks: List[List[Decimal]]  # [price, quantity]
    timestamp: datetime


class Kline(BaseModel):
    """Candlestick data"""
    symbol: str
    exchange: ExchangeName
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    trades_count: int


class Order(BaseEntity):
    """Trading order"""
    user_id: UUID
    strategy_id: Optional[UUID] = None
    exchange: ExchangeName
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    fees: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Position(BaseEntity):
    """Trading position"""
    user_id: UUID
    strategy_id: Optional[UUID] = None
    exchange: ExchangeName
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')
    leverage: int = 1
    margin: Decimal
    liquidation_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class Trade(BaseEntity):
    """Executed trade"""
    user_id: UUID
    strategy_id: Optional[UUID] = None
    order_id: UUID
    exchange: ExchangeName
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fees: Decimal
    pnl: Decimal
    exchange_trade_id: str
    executed_at: datetime


# Strategy Models
class Strategy(BaseEntity):
    """Trading strategy"""
    user_id: UUID
    name: str
    description: str
    type: StrategyType
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    max_allocation: Decimal
    current_allocation: Decimal = Decimal('0')
    active: bool = True
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class Signal(BaseEntity):
    """Trading signal"""
    strategy_id: UUID
    symbol: str
    exchange: ExchangeName
    type: SignalType
    side: OrderSide
    strength: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    confidence: float = Field(ge=0.0, le=1.0)  # 0 to 1
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timeframe: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None


# Portfolio Models
class Portfolio(BaseEntity):
    """User portfolio"""
    user_id: UUID
    name: str
    total_value: Decimal
    available_balance: Decimal
    allocated_balance: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    risk_score: float
    strategies: List[UUID] = Field(default_factory=list)


class PortfolioSnapshot(BaseEntity):
    """Portfolio state snapshot"""
    portfolio_id: UUID
    total_value: Decimal
    available_balance: Decimal
    allocated_balance: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    positions_count: int
    active_orders_count: int
    risk_metrics: Dict[str, Any] = Field(default_factory=dict)


# Risk Models
class RiskMetrics(BaseModel):
    """Risk assessment metrics"""
    portfolio_id: UUID
    var_95: Decimal  # Value at Risk 95%
    var_99: Decimal  # Value at Risk 99%
    expected_shortfall: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    concentration_risk: float
    leverage_ratio: float
    calculated_at: datetime


class RiskLimit(BaseEntity):
    """Risk limits configuration"""
    user_id: UUID
    portfolio_id: Optional[UUID] = None
    max_daily_loss_pct: Decimal
    max_position_size_pct: Decimal
    max_leverage: int
    max_correlation: float
    max_concentration_pct: Decimal
    var_limit: Decimal
    active: bool = True


# User Models
class User(BaseEntity):
    """Platform user"""
    email: str
    username: str
    first_name: str
    last_name: str
    is_active: bool = True
    is_verified: bool = False
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    kyc_status: str = "PENDING"
    preferences: Dict[str, Any] = Field(default_factory=dict)


class UserSession(BaseEntity):
    """User session"""
    user_id: UUID
    token: str
    expires_at: datetime
    ip_address: str
    user_agent: str
    active: bool = True


# Analytics Models
class PerformanceMetrics(BaseModel):
    """Performance analysis metrics"""
    entity_id: UUID  # Portfolio, Strategy, or User ID
    entity_type: str  # "portfolio", "strategy", "user"
    period_start: datetime
    period_end: datetime
    total_return: Decimal
    total_return_pct: float
    annualized_return_pct: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: Decimal
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    trades_count: int
    avg_trade_duration_hours: float
    best_trade: Decimal
    worst_trade: Decimal


# Market Data Models
class MarketSentiment(BaseModel):
    """Market sentiment analysis"""
    symbol: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # -1 (bearish) to 1 (bullish)
    confidence: float = Field(ge=0.0, le=1.0)
    news_sentiment: float
    social_sentiment: float
    technical_sentiment: float
    volume_sentiment: float
    sources_count: int
    analyzed_at: datetime


class PricePrediction(BaseModel):
    """AI price prediction"""
    symbol: str
    exchange: ExchangeName
    current_price: Decimal
    predicted_price: Decimal
    prediction_horizon_hours: int
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str
    features_used: List[str]
    predicted_at: datetime


# Arbitrage Models
class ArbitrageOpportunity(BaseEntity):
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    buy_exchange: ExchangeName
    sell_exchange: ExchangeName
    buy_price: Decimal
    sell_price: Decimal
    profit_amount: Decimal
    profit_percentage: float
    required_capital: Decimal
    execution_time_estimate_ms: int
    risk_score: float
    detected_at: datetime
    expires_at: datetime
    executed: bool = False


# Notification Models
class Notification(BaseEntity):
    """User notification"""
    user_id: UUID
    title: str
    message: str
    type: str  # "INFO", "WARNING", "ERROR", "SUCCESS"
    priority: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    read: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# API Models
class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated API response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str
    channel: str
    data: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""
    action: str  # "subscribe" or "unsubscribe"
    channels: List[str]
    symbols: Optional[List[str]] = None
    user_id: Optional[UUID] = None