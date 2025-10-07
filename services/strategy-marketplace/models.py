"""
Database models for Strategy Marketplace Service
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class Strategy(Base):
    """Strategy model for published trading strategies"""
    __tablename__ = "strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    creator_id = Column(UUID(as_uuid=True), nullable=False)
    creator_name = Column(String(255), nullable=False)
    
    # Strategy details
    category = Column(String(100))  # e.g., "arbitrage", "momentum", "mean_reversion"
    risk_level = Column(String(50))  # "low", "medium", "high"
    min_capital = Column(Float, default=100.0)
    max_drawdown = Column(Float)
    
    # Strategy configuration
    parameters = Column(JSON)  # Strategy parameters as JSON
    code_hash = Column(String(64))  # Hash of strategy code for integrity
    
    # Marketplace info
    is_public = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    subscription_fee = Column(Float, default=0.0)  # Monthly fee
    performance_fee = Column(Float, default=0.0)  # Percentage of profits
    
    # Metrics
    total_subscribers = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    total_ratings = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("StrategySubscription", back_populates="strategy")
    performance_records = relationship("StrategyPerformance", back_populates="strategy")
    ratings = relationship("StrategyRating", back_populates="strategy")

class StrategySubscription(Base):
    """User subscriptions to strategies"""
    __tablename__ = "strategy_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    subscriber_id = Column(UUID(as_uuid=True), nullable=False)
    subscriber_name = Column(String(255), nullable=False)
    
    # Subscription details
    allocation_percentage = Column(Float, default=10.0)  # % of portfolio to allocate
    max_position_size = Column(Float)  # Maximum position size
    risk_multiplier = Column(Float, default=1.0)  # Risk scaling factor
    
    # Status
    is_active = Column(Boolean, default=True)
    auto_trade = Column(Boolean, default=True)
    
    # Financial tracking
    total_fees_paid = Column(Float, default=0.0)
    total_profit_shared = Column(Float, default=0.0)
    
    # Timestamps
    subscribed_at = Column(DateTime, default=datetime.utcnow)
    last_trade_at = Column(DateTime)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="subscriptions")

class StrategyPerformance(Base):
    """Performance tracking for strategies"""
    __tablename__ = "strategy_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20))  # "daily", "weekly", "monthly"
    
    # Performance metrics
    total_return = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    
    # Trading statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    average_trade_duration = Column(Float)  # in hours
    
    # Risk metrics
    var_95 = Column(Float)  # Value at Risk 95%
    expected_shortfall = Column(Float)
    beta = Column(Float)  # Market beta
    alpha = Column(Float)  # Market alpha
    
    # Volume and liquidity
    total_volume = Column(Float, default=0.0)
    average_position_size = Column(Float)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="performance_records")

class StrategyRating(Base):
    """User ratings and reviews for strategies"""
    __tablename__ = "strategy_ratings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    user_name = Column(String(255), nullable=False)
    
    # Rating details
    rating = Column(Float, nullable=False)  # 1.0 to 5.0
    review = Column(Text)
    
    # Rating categories
    performance_rating = Column(Float)  # Performance satisfaction
    risk_rating = Column(Float)  # Risk management satisfaction
    communication_rating = Column(Float)  # Creator communication
    
    # Verification
    is_verified_subscriber = Column(Boolean, default=False)
    subscription_duration = Column(Integer)  # Days subscribed when rated
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="ratings")

class StrategyEarnings(Base):
    """Earnings tracking for strategy creators"""
    __tablename__ = "strategy_earnings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    creator_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Earnings period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Revenue breakdown
    subscription_revenue = Column(Float, default=0.0)
    performance_revenue = Column(Float, default=0.0)
    total_revenue = Column(Float, default=0.0)
    
    # Platform fees
    platform_fee_rate = Column(Float, default=0.3)  # 30% platform fee
    platform_fee_amount = Column(Float, default=0.0)
    net_earnings = Column(Float, default=0.0)
    
    # Subscriber metrics
    active_subscribers = Column(Integer, default=0)
    new_subscribers = Column(Integer, default=0)
    churned_subscribers = Column(Integer, default=0)
    
    # Payment status
    is_paid = Column(Boolean, default=False)
    paid_at = Column(DateTime)
    payment_reference = Column(String(255))
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)

class StrategyLeaderboard(Base):
    """Leaderboard rankings for strategies"""
    __tablename__ = "strategy_leaderboard"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"), nullable=False)
    
    # Ranking period
    period = Column(String(20))  # "daily", "weekly", "monthly", "yearly"
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Rankings
    overall_rank = Column(Integer)
    category_rank = Column(Integer)
    risk_adjusted_rank = Column(Integer)
    
    # Ranking metrics
    ranking_score = Column(Float)  # Composite ranking score
    risk_adjusted_return = Column(Float)
    consistency_score = Column(Float)
    subscriber_growth = Column(Float)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)