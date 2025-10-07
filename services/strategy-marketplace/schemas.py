"""
Pydantic schemas for Strategy Marketplace Service
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

class StrategyCreate(BaseModel):
    """Schema for creating a new strategy"""
    name: str = Field(..., min_length=3, max_length=255)
    description: str = Field(..., min_length=10, max_length=2000)
    category: str = Field(..., regex="^(arbitrage|momentum|mean_reversion|scalping|swing|grid|dca)$")
    risk_level: str = Field(..., regex="^(low|medium|high)$")
    min_capital: float = Field(100.0, ge=100.0)
    max_drawdown: Optional[float] = Field(None, ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    subscription_fee: float = Field(0.0, ge=0.0, le=1000.0)
    performance_fee: float = Field(0.0, ge=0.0, le=0.5)
    is_public: bool = True

    @validator('performance_fee')
    def validate_performance_fee(cls, v):
        if v > 0.5:  # Max 50% performance fee
            raise ValueError('Performance fee cannot exceed 50%')
        return v

class StrategyUpdate(BaseModel):
    """Schema for updating strategy details"""
    description: Optional[str] = Field(None, min_length=10, max_length=2000)
    subscription_fee: Optional[float] = Field(None, ge=0.0, le=1000.0)
    performance_fee: Optional[float] = Field(None, ge=0.0, le=0.5)
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None

class StrategyResponse(BaseModel):
    """Schema for strategy response"""
    id: UUID
    name: str
    description: str
    creator_id: UUID
    creator_name: str
    category: str
    risk_level: str
    min_capital: float
    max_drawdown: Optional[float]
    subscription_fee: float
    performance_fee: float
    is_public: bool
    is_active: bool
    total_subscribers: int
    average_rating: float
    total_ratings: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SubscriptionCreate(BaseModel):
    """Schema for creating a strategy subscription"""
    allocation_percentage: float = Field(10.0, ge=1.0, le=100.0)
    max_position_size: Optional[float] = Field(None, gt=0)
    risk_multiplier: float = Field(1.0, ge=0.1, le=5.0)
    auto_trade: bool = True

    @validator('allocation_percentage')
    def validate_allocation(cls, v):
        if v > 100:
            raise ValueError('Allocation cannot exceed 100%')
        return v

class SubscriptionResponse(BaseModel):
    """Schema for subscription response"""
    id: UUID
    strategy_id: UUID
    subscriber_id: UUID
    subscriber_name: str
    allocation_percentage: float
    max_position_size: Optional[float]
    risk_multiplier: float
    is_active: bool
    auto_trade: bool
    total_fees_paid: float
    total_profit_shared: float
    subscribed_at: datetime
    last_trade_at: Optional[datetime]
    strategy: StrategyResponse

    class Config:
        from_attributes = True

class PerformanceMetrics(BaseModel):
    """Schema for strategy performance metrics"""
    strategy_id: UUID
    period_start: datetime
    period_end: datetime
    period_type: str
    total_return: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade_duration: Optional[float]
    var_95: Optional[float]
    expected_shortfall: Optional[float]
    beta: Optional[float]
    alpha: Optional[float]
    total_volume: float
    average_position_size: Optional[float]
    calculated_at: datetime

    class Config:
        from_attributes = True

class RatingCreate(BaseModel):
    """Schema for creating a strategy rating"""
    rating: float = Field(..., ge=1.0, le=5.0)
    review: Optional[str] = Field(None, max_length=1000)
    performance_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    risk_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    communication_rating: Optional[float] = Field(None, ge=1.0, le=5.0)

class RatingResponse(BaseModel):
    """Schema for rating response"""
    id: UUID
    strategy_id: UUID
    user_id: UUID
    user_name: str
    rating: float
    review: Optional[str]
    performance_rating: Optional[float]
    risk_rating: Optional[float]
    communication_rating: Optional[float]
    is_verified_subscriber: bool
    subscription_duration: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MarketplaceStats(BaseModel):
    """Schema for marketplace statistics"""
    total_strategies: int
    active_strategies: int
    total_subscribers: int
    total_volume_traded: float
    average_strategy_rating: float
    top_performing_strategies: List[StrategyResponse]
    categories: Dict[str, int]
    risk_levels: Dict[str, int]

class EarningsResponse(BaseModel):
    """Schema for creator earnings response"""
    strategy_id: UUID
    period_start: datetime
    period_end: datetime
    subscription_revenue: float
    performance_revenue: float
    total_revenue: float
    platform_fee_rate: float
    platform_fee_amount: float
    net_earnings: float
    active_subscribers: int
    new_subscribers: int
    churned_subscribers: int

    class Config:
        from_attributes = True

class LeaderboardEntry(BaseModel):
    """Schema for leaderboard entry"""
    strategy: StrategyResponse
    overall_rank: int
    category_rank: int
    risk_adjusted_rank: int
    ranking_score: float
    risk_adjusted_return: float
    consistency_score: float
    subscriber_growth: float

    class Config:
        from_attributes = True

class StrategySearchFilters(BaseModel):
    """Schema for strategy search filters"""
    category: Optional[str] = None
    risk_level: Optional[str] = None
    min_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    max_subscription_fee: Optional[float] = Field(None, ge=0.0)
    max_performance_fee: Optional[float] = Field(None, ge=0.0, le=0.5)
    min_subscribers: Optional[int] = Field(None, ge=0)
    sort_by: str = Field("performance", regex="^(performance|rating|subscribers|created_at|fees)$")
    sort_order: str = Field("desc", regex="^(asc|desc)$")

class CopyTradeSignal(BaseModel):
    """Schema for copy trading signals"""
    strategy_id: UUID
    signal_type: str = Field(..., regex="^(open|close|modify)$")
    symbol: str
    side: str = Field(..., regex="^(buy|sell)$")
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SubscriptionMetrics(BaseModel):
    """Schema for subscription performance metrics"""
    subscription_id: UUID
    total_return: float
    total_fees_paid: float
    total_profit_shared: float
    trades_copied: int
    successful_trades: int
    failed_trades: int
    average_slippage: float
    correlation_with_strategy: float
    tracking_error: float