"""
Strategy Marketplace Service

This service handles strategy publication, subscription, performance tracking,
and monetization for the advanced trading platform.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
from datetime import datetime, timedelta
import uuid

from database import get_db, engine
from models import Base, Strategy, StrategySubscription, StrategyPerformance, StrategyRating
from schemas import (
    StrategyCreate, StrategyResponse, StrategyUpdate,
    SubscriptionCreate, SubscriptionResponse,
    PerformanceMetrics, RatingCreate, RatingResponse,
    MarketplaceStats
)
from services import (
    StrategyService, SubscriptionService, 
    PerformanceTracker, MonetizationService
)
from auth import verify_token, get_current_user

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Strategy Marketplace Service",
    description="Handles strategy sharing, subscription, and monetization",
    version="1.0.0"
)

security = HTTPBearer()

# Initialize services
strategy_service = StrategyService()
subscription_service = SubscriptionService()
performance_tracker = PerformanceTracker()
monetization_service = MonetizationService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "strategy-marketplace"}

@app.post("/strategies", response_model=StrategyResponse)
async def publish_strategy(
    strategy: StrategyCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Publish a new strategy to the marketplace"""
    try:
        published_strategy = await strategy_service.publish_strategy(
            strategy, current_user["user_id"], db
        )
        return published_strategy
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to publish strategy: {str(e)}"
        )

@app.get("/strategies", response_model=List[StrategyResponse])
async def get_strategies(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    sort_by: str = "performance",
    db: Session = Depends(get_db)
):
    """Get list of published strategies with filtering and sorting"""
    strategies = await strategy_service.get_strategies(
        db, skip=skip, limit=limit, category=category,
        min_rating=min_rating, sort_by=sort_by
    )
    return strategies

@app.get("/strategies/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific strategy"""
    strategy = await strategy_service.get_strategy_by_id(strategy_id, db)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    return strategy

@app.post("/strategies/{strategy_id}/subscribe", response_model=SubscriptionResponse)
async def subscribe_to_strategy(
    strategy_id: str,
    subscription: SubscriptionCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Subscribe to a strategy for copy trading"""
    try:
        subscription_result = await subscription_service.create_subscription(
            strategy_id, current_user["user_id"], subscription, db
        )
        return subscription_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to subscribe: {str(e)}"
        )

@app.delete("/strategies/{strategy_id}/subscribe")
async def unsubscribe_from_strategy(
    strategy_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Unsubscribe from a strategy"""
    success = await subscription_service.cancel_subscription(
        strategy_id, current_user["user_id"], db
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    return {"message": "Successfully unsubscribed"}

@app.get("/strategies/{strategy_id}/performance", response_model=PerformanceMetrics)
async def get_strategy_performance(
    strategy_id: str,
    period: str = "30d",
    db: Session = Depends(get_db)
):
    """Get performance metrics for a strategy"""
    performance = await performance_tracker.get_performance_metrics(
        strategy_id, period, db
    )
    if not performance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Performance data not found"
        )
    return performance

@app.post("/strategies/{strategy_id}/rate", response_model=RatingResponse)
async def rate_strategy(
    strategy_id: str,
    rating: RatingCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Rate and review a strategy"""
    try:
        rating_result = await strategy_service.rate_strategy(
            strategy_id, current_user["user_id"], rating, db
        )
        return rating_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to rate strategy: {str(e)}"
        )

@app.get("/my-strategies", response_model=List[StrategyResponse])
async def get_my_strategies(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get strategies published by the current user"""
    strategies = await strategy_service.get_user_strategies(
        current_user["user_id"], db
    )
    return strategies

@app.get("/my-subscriptions", response_model=List[SubscriptionResponse])
async def get_my_subscriptions(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get strategies subscribed by the current user"""
    subscriptions = await subscription_service.get_user_subscriptions(
        current_user["user_id"], db
    )
    return subscriptions

@app.get("/marketplace/stats", response_model=MarketplaceStats)
async def get_marketplace_stats(db: Session = Depends(get_db)):
    """Get overall marketplace statistics"""
    stats = await strategy_service.get_marketplace_stats(db)
    return stats

@app.post("/strategies/{strategy_id}/earnings")
async def calculate_earnings(
    strategy_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Calculate and distribute earnings for strategy creators"""
    try:
        earnings = await monetization_service.calculate_strategy_earnings(
            strategy_id, current_user["user_id"], db
        )
        return {"earnings": earnings, "period": "current_month"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to calculate earnings: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)