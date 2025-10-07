"""
Core services for Strategy Marketplace
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
import httpx
from uuid import UUID, uuid4

from models import (
    Strategy, StrategySubscription, StrategyPerformance, 
    StrategyRating, StrategyEarnings, StrategyLeaderboard
)
from schemas import (
    StrategyCreate, StrategyResponse, SubscriptionCreate, 
    PerformanceMetrics, RatingCreate, MarketplaceStats,
    CopyTradeSignal
)

class StrategyService:
    """Service for managing strategies in the marketplace"""
    
    def __init__(self):
        self.strategy_engine_url = "http://localhost:8003"  # Strategy Engine Service
    
    async def publish_strategy(self, strategy_data: StrategyCreate, creator_id: str, db: Session) -> StrategyResponse:
        """Publish a new strategy to the marketplace"""
        
        # Generate code hash for integrity
        code_hash = hashlib.sha256(
            json.dumps(strategy_data.parameters, sort_keys=True).encode()
        ).hexdigest()
        
        # Create strategy record
        strategy = Strategy(
            name=strategy_data.name,
            description=strategy_data.description,
            creator_id=creator_id,
            creator_name=await self._get_user_name(creator_id),
            category=strategy_data.category,
            risk_level=strategy_data.risk_level,
            min_capital=strategy_data.min_capital,
            max_drawdown=strategy_data.max_drawdown,
            parameters=strategy_data.parameters,
            code_hash=code_hash,
            is_public=strategy_data.is_public,
            subscription_fee=strategy_data.subscription_fee,
            performance_fee=strategy_data.performance_fee
        )
        
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
        
        # Register strategy with strategy engine
        await self._register_strategy_with_engine(strategy)
        
        return StrategyResponse.from_orm(strategy)
    
    async def get_strategies(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        category: Optional[str] = None,
        min_rating: Optional[float] = None,
        sort_by: str = "performance"
    ) -> List[StrategyResponse]:
        """Get list of strategies with filtering and sorting"""
        
        query = db.query(Strategy).filter(Strategy.is_public == True, Strategy.is_active == True)
        
        # Apply filters
        if category:
            query = query.filter(Strategy.category == category)
        
        if min_rating:
            query = query.filter(Strategy.average_rating >= min_rating)
        
        # Apply sorting
        if sort_by == "performance":
            # Join with performance data and sort by total return
            query = query.outerjoin(StrategyPerformance).order_by(
                desc(StrategyPerformance.total_return)
            )
        elif sort_by == "rating":
            query = query.order_by(desc(Strategy.average_rating))
        elif sort_by == "subscribers":
            query = query.order_by(desc(Strategy.total_subscribers))
        elif sort_by == "created_at":
            query = query.order_by(desc(Strategy.created_at))
        
        strategies = query.offset(skip).limit(limit).all()
        return [StrategyResponse.from_orm(strategy) for strategy in strategies]
    
    async def get_strategy_by_id(self, strategy_id: str, db: Session) -> Optional[StrategyResponse]:
        """Get strategy by ID"""
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if strategy:
            return StrategyResponse.from_orm(strategy)
        return None
    
    async def get_user_strategies(self, creator_id: str, db: Session) -> List[StrategyResponse]:
        """Get strategies created by a specific user"""
        strategies = db.query(Strategy).filter(Strategy.creator_id == creator_id).all()
        return [StrategyResponse.from_orm(strategy) for strategy in strategies]
    
    async def rate_strategy(
        self, 
        strategy_id: str, 
        user_id: str, 
        rating_data: RatingCreate, 
        db: Session
    ) -> Dict[str, Any]:
        """Rate and review a strategy"""
        
        # Check if user has already rated this strategy
        existing_rating = db.query(StrategyRating).filter(
            and_(StrategyRating.strategy_id == strategy_id, StrategyRating.user_id == user_id)
        ).first()
        
        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating_data.rating
            existing_rating.review = rating_data.review
            existing_rating.performance_rating = rating_data.performance_rating
            existing_rating.risk_rating = rating_data.risk_rating
            existing_rating.communication_rating = rating_data.communication_rating
            existing_rating.updated_at = datetime.utcnow()
        else:
            # Create new rating
            rating = StrategyRating(
                strategy_id=strategy_id,
                user_id=user_id,
                user_name=await self._get_user_name(user_id),
                rating=rating_data.rating,
                review=rating_data.review,
                performance_rating=rating_data.performance_rating,
                risk_rating=rating_data.risk_rating,
                communication_rating=rating_data.communication_rating,
                is_verified_subscriber=await self._is_verified_subscriber(strategy_id, user_id, db)
            )
            db.add(rating)
        
        db.commit()
        
        # Update strategy average rating
        await self._update_strategy_rating(strategy_id, db)
        
        return {"message": "Rating submitted successfully"}
    
    async def get_marketplace_stats(self, db: Session) -> MarketplaceStats:
        """Get overall marketplace statistics"""
        
        total_strategies = db.query(Strategy).filter(Strategy.is_public == True).count()
        active_strategies = db.query(Strategy).filter(
            and_(Strategy.is_public == True, Strategy.is_active == True)
        ).count()
        
        total_subscribers = db.query(func.sum(Strategy.total_subscribers)).scalar() or 0
        
        # Get category distribution
        categories = db.query(
            Strategy.category, func.count(Strategy.id)
        ).filter(Strategy.is_public == True).group_by(Strategy.category).all()
        
        # Get risk level distribution
        risk_levels = db.query(
            Strategy.risk_level, func.count(Strategy.id)
        ).filter(Strategy.is_public == True).group_by(Strategy.risk_level).all()
        
        # Get top performing strategies
        top_strategies = db.query(Strategy).filter(
            and_(Strategy.is_public == True, Strategy.is_active == True)
        ).order_by(desc(Strategy.average_rating)).limit(10).all()
        
        return MarketplaceStats(
            total_strategies=total_strategies,
            active_strategies=active_strategies,
            total_subscribers=total_subscribers,
            total_volume_traded=0.0,  # Would be calculated from trading data
            average_strategy_rating=db.query(func.avg(Strategy.average_rating)).scalar() or 0.0,
            top_performing_strategies=[StrategyResponse.from_orm(s) for s in top_strategies],
            categories={cat: count for cat, count in categories},
            risk_levels={risk: count for risk, count in risk_levels}
        )
    
    async def _get_user_name(self, user_id: str) -> str:
        """Get user name from user service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:8001/users/{user_id}")
                if response.status_code == 200:
                    user_data = response.json()
                    return user_data.get("username", f"User_{user_id[:8]}")
        except Exception:
            pass
        return f"User_{user_id[:8]}"
    
    async def _register_strategy_with_engine(self, strategy: Strategy):
        """Register strategy with the strategy engine"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.strategy_engine_url}/strategies/register",
                    json={
                        "strategy_id": str(strategy.id),
                        "name": strategy.name,
                        "parameters": strategy.parameters,
                        "creator_id": str(strategy.creator_id)
                    }
                )
        except Exception as e:
            print(f"Failed to register strategy with engine: {e}")
    
    async def _is_verified_subscriber(self, strategy_id: str, user_id: str, db: Session) -> bool:
        """Check if user is a verified subscriber"""
        subscription = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.strategy_id == strategy_id,
                StrategySubscription.subscriber_id == user_id,
                StrategySubscription.is_active == True
            )
        ).first()
        return subscription is not None
    
    async def _update_strategy_rating(self, strategy_id: str, db: Session):
        """Update strategy average rating"""
        avg_rating = db.query(func.avg(StrategyRating.rating)).filter(
            StrategyRating.strategy_id == strategy_id
        ).scalar()
        
        total_ratings = db.query(func.count(StrategyRating.id)).filter(
            StrategyRating.strategy_id == strategy_id
        ).scalar()
        
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if strategy:
            strategy.average_rating = avg_rating or 0.0
            strategy.total_ratings = total_ratings or 0
            db.commit()

class SubscriptionService:
    """Service for managing strategy subscriptions"""
    
    def __init__(self):
        self.trading_engine_url = "http://localhost:8002"
    
    async def create_subscription(
        self, 
        strategy_id: str, 
        subscriber_id: str, 
        subscription_data: SubscriptionCreate, 
        db: Session
    ) -> Dict[str, Any]:
        """Create a new strategy subscription"""
        
        # Check if already subscribed
        existing = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.strategy_id == strategy_id,
                StrategySubscription.subscriber_id == subscriber_id,
                StrategySubscription.is_active == True
            )
        ).first()
        
        if existing:
            raise ValueError("Already subscribed to this strategy")
        
        # Create subscription
        subscription = StrategySubscription(
            strategy_id=strategy_id,
            subscriber_id=subscriber_id,
            subscriber_name=await self._get_user_name(subscriber_id),
            allocation_percentage=subscription_data.allocation_percentage,
            max_position_size=subscription_data.max_position_size,
            risk_multiplier=subscription_data.risk_multiplier,
            auto_trade=subscription_data.auto_trade
        )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        # Update strategy subscriber count
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if strategy:
            strategy.total_subscribers += 1
            db.commit()
        
        # Register with trading engine for copy trading
        await self._register_copy_trading(subscription)
        
        return {"message": "Successfully subscribed to strategy", "subscription_id": str(subscription.id)}
    
    async def cancel_subscription(self, strategy_id: str, subscriber_id: str, db: Session) -> bool:
        """Cancel a strategy subscription"""
        
        subscription = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.strategy_id == strategy_id,
                StrategySubscription.subscriber_id == subscriber_id,
                StrategySubscription.is_active == True
            )
        ).first()
        
        if not subscription:
            return False
        
        subscription.is_active = False
        db.commit()
        
        # Update strategy subscriber count
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if strategy and strategy.total_subscribers > 0:
            strategy.total_subscribers -= 1
            db.commit()
        
        # Unregister from trading engine
        await self._unregister_copy_trading(subscription)
        
        return True
    
    async def get_user_subscriptions(self, subscriber_id: str, db: Session) -> List[Dict[str, Any]]:
        """Get user's active subscriptions"""
        
        subscriptions = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.subscriber_id == subscriber_id,
                StrategySubscription.is_active == True
            )
        ).all()
        
        result = []
        for sub in subscriptions:
            strategy = db.query(Strategy).filter(Strategy.id == sub.strategy_id).first()
            result.append({
                "subscription": sub,
                "strategy": strategy
            })
        
        return result
    
    async def process_copy_trade_signal(self, signal: CopyTradeSignal, db: Session):
        """Process copy trading signal for all subscribers"""
        
        # Get all active subscribers for this strategy
        subscriptions = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.strategy_id == signal.strategy_id,
                StrategySubscription.is_active == True,
                StrategySubscription.auto_trade == True
            )
        ).all()
        
        for subscription in subscriptions:
            try:
                # Calculate position size based on allocation and risk multiplier
                adjusted_quantity = (
                    signal.quantity * 
                    (subscription.allocation_percentage / 100) * 
                    subscription.risk_multiplier
                )
                
                # Apply max position size limit
                if subscription.max_position_size:
                    adjusted_quantity = min(adjusted_quantity, subscription.max_position_size)
                
                # Send trade signal to trading engine
                await self._execute_copy_trade(subscription, signal, adjusted_quantity)
                
            except Exception as e:
                print(f"Failed to copy trade for subscription {subscription.id}: {e}")
    
    async def _get_user_name(self, user_id: str) -> str:
        """Get user name from user service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:8001/users/{user_id}")
                if response.status_code == 200:
                    user_data = response.json()
                    return user_data.get("username", f"User_{user_id[:8]}")
        except Exception:
            pass
        return f"User_{user_id[:8]}"
    
    async def _register_copy_trading(self, subscription: StrategySubscription):
        """Register subscription for copy trading"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.trading_engine_url}/copy-trading/register",
                    json={
                        "subscription_id": str(subscription.id),
                        "strategy_id": str(subscription.strategy_id),
                        "subscriber_id": str(subscription.subscriber_id),
                        "allocation_percentage": subscription.allocation_percentage,
                        "risk_multiplier": subscription.risk_multiplier
                    }
                )
        except Exception as e:
            print(f"Failed to register copy trading: {e}")
    
    async def _unregister_copy_trading(self, subscription: StrategySubscription):
        """Unregister subscription from copy trading"""
        try:
            async with httpx.AsyncClient() as client:
                await client.delete(
                    f"{self.trading_engine_url}/copy-trading/{subscription.id}"
                )
        except Exception as e:
            print(f"Failed to unregister copy trading: {e}")
    
    async def _execute_copy_trade(self, subscription: StrategySubscription, signal: CopyTradeSignal, quantity: float):
        """Execute copy trade for a subscription"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.trading_engine_url}/copy-trading/execute",
                    json={
                        "subscription_id": str(subscription.id),
                        "subscriber_id": str(subscription.subscriber_id),
                        "signal_type": signal.signal_type,
                        "symbol": signal.symbol,
                        "side": signal.side,
                        "quantity": quantity,
                        "price": signal.price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit
                    }
                )
        except Exception as e:
            print(f"Failed to execute copy trade: {e}")

class PerformanceTracker:
    """Service for tracking strategy performance"""
    
    async def get_performance_metrics(
        self, 
        strategy_id: str, 
        period: str, 
        db: Session
    ) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a strategy"""
        
        # Calculate period dates
        end_date = datetime.utcnow()
        if period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "30d":
            start_date = end_date - timedelta(days=30)
        elif period == "90d":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Get latest performance record for the period
        performance = db.query(StrategyPerformance).filter(
            and_(
                StrategyPerformance.strategy_id == strategy_id,
                StrategyPerformance.period_start >= start_date,
                StrategyPerformance.period_end <= end_date
            )
        ).order_by(desc(StrategyPerformance.calculated_at)).first()
        
        if performance:
            return PerformanceMetrics.from_orm(performance)
        
        return None
    
    async def calculate_and_store_performance(self, strategy_id: str, db: Session):
        """Calculate and store performance metrics for a strategy"""
        
        # This would integrate with the trading engine to get actual trade data
        # For now, we'll create a placeholder implementation
        
        performance = StrategyPerformance(
            strategy_id=strategy_id,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            period_type="monthly",
            total_return=0.0,  # Would be calculated from actual trades
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0
        )
        
        db.add(performance)
        db.commit()

class MonetizationService:
    """Service for handling strategy monetization and revenue sharing"""
    
    async def calculate_strategy_earnings(
        self, 
        strategy_id: str, 
        creator_id: str, 
        db: Session
    ) -> Dict[str, Any]:
        """Calculate earnings for a strategy creator"""
        
        # Get current month period
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = now
        
        # Get active subscriptions
        subscriptions = db.query(StrategySubscription).filter(
            and_(
                StrategySubscription.strategy_id == strategy_id,
                StrategySubscription.is_active == True
            )
        ).all()
        
        # Calculate subscription revenue
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        subscription_revenue = len(subscriptions) * strategy.subscription_fee if strategy else 0
        
        # Calculate performance revenue (would need actual trading data)
        performance_revenue = 0.0  # Placeholder
        
        total_revenue = subscription_revenue + performance_revenue
        platform_fee_rate = 0.3  # 30% platform fee
        platform_fee_amount = total_revenue * platform_fee_rate
        net_earnings = total_revenue - platform_fee_amount
        
        # Store earnings record
        earnings = StrategyEarnings(
            strategy_id=strategy_id,
            creator_id=creator_id,
            period_start=period_start,
            period_end=period_end,
            subscription_revenue=subscription_revenue,
            performance_revenue=performance_revenue,
            total_revenue=total_revenue,
            platform_fee_rate=platform_fee_rate,
            platform_fee_amount=platform_fee_amount,
            net_earnings=net_earnings,
            active_subscribers=len(subscriptions)
        )
        
        db.add(earnings)
        db.commit()
        
        return {
            "total_revenue": total_revenue,
            "platform_fee": platform_fee_amount,
            "net_earnings": net_earnings,
            "active_subscribers": len(subscriptions)
        }