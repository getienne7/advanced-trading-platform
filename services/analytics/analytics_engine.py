"""
Real-time Analytics Engine - Core analytics processing and coordination
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import json

from performance_calculator import PerformanceCalculator
from pnl_attribution import PnLAttributionEngine
from risk_metrics_calculator import RiskMetricsCalculator
from data_aggregator import DataAggregator

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics engine that coordinates real-time P&L calculation,
    performance metrics, and risk analysis
    """
    
    def __init__(self):
        self.performance_calculator = PerformanceCalculator()
        self.pnl_attribution = PnLAttributionEngine()
        self.risk_calculator = RiskMetricsCalculator()
        self.data_aggregator = DataAggregator()
        
        self.running = False
        self.update_interval = 5  # seconds
        self.cache = {}
        self.cache_ttl = 60  # seconds
        
    async def start_real_time_processing(self):
        """Start real-time analytics processing"""
        self.running = True
        logger.info("Starting real-time analytics processing")
        
        # Start background tasks
        await asyncio.gather(
            self._real_time_pnl_loop(),
            self._risk_metrics_loop(),
            self._performance_update_loop(),
            self._cache_cleanup_loop()
        )
    
    async def stop(self):
        """Stop analytics processing"""
        self.running = False
        logger.info("Stopping analytics engine")
    
    async def _real_time_pnl_loop(self):
        """Continuously update P&L calculations"""
        while self.running:
            try:
                # Get all active users/portfolios
                active_users = await self.data_aggregator.get_active_users()
                
                for user_id in active_users:
                    # Update real-time P&L
                    await self._update_user_pnl(user_id)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time P&L loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _risk_metrics_loop(self):
        """Continuously update risk metrics"""
        while self.running:
            try:
                # Get all active portfolios
                active_portfolios = await self.data_aggregator.get_active_portfolios()
                
                for portfolio_id in active_portfolios:
                    # Update risk metrics
                    await self._update_portfolio_risk_metrics(portfolio_id)
                
                await asyncio.sleep(self.update_interval * 2)  # Less frequent updates
                
            except Exception as e:
                logger.error(f"Error in risk metrics loop: {e}")
                await asyncio.sleep(self.update_interval * 2)
    
    async def _performance_update_loop(self):
        """Continuously update performance metrics"""
        while self.running:
            try:
                # Update performance metrics every minute
                active_strategies = await self.data_aggregator.get_active_strategies()
                
                for strategy_id in active_strategies:
                    await self._update_strategy_performance(strategy_id)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance update loop: {e}")
                await asyncio.sleep(60)
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, (data, timestamp) in self.cache.items():
                    if (current_time - timestamp).total_seconds() > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
                
                await asyncio.sleep(30)  # Clean up every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(30)
    
    async def _update_user_pnl(self, user_id: str):
        """Update real-time P&L for a user"""
        try:
            # Get current positions and trades
            positions = await self.data_aggregator.get_user_positions(user_id)
            recent_trades = await self.data_aggregator.get_recent_trades(user_id)
            
            # Calculate P&L attribution
            pnl_data = await self.pnl_attribution.calculate_real_time_pnl(
                positions, recent_trades
            )
            
            # Cache the result
            cache_key = f"pnl_{user_id}"
            self.cache[cache_key] = (pnl_data, datetime.utcnow())
            
        except Exception as e:
            logger.error(f"Error updating P&L for user {user_id}: {e}")
    
    async def _update_portfolio_risk_metrics(self, portfolio_id: str):
        """Update risk metrics for a portfolio"""
        try:
            # Get portfolio data
            portfolio_data = await self.data_aggregator.get_portfolio_data(portfolio_id)
            
            # Calculate risk metrics
            risk_metrics = await self.risk_calculator.calculate_real_time_metrics(
                portfolio_data
            )
            
            # Cache the result
            cache_key = f"risk_{portfolio_id}"
            self.cache[cache_key] = (risk_metrics, datetime.utcnow())
            
        except Exception as e:
            logger.error(f"Error updating risk metrics for portfolio {portfolio_id}: {e}")
    
    async def _update_strategy_performance(self, strategy_id: str):
        """Update performance metrics for a strategy"""
        try:
            # Get strategy data
            strategy_data = await self.data_aggregator.get_strategy_data(strategy_id)
            
            # Calculate performance metrics
            performance = await self.performance_calculator.calculate_strategy_performance(
                strategy_data
            )
            
            # Cache the result
            cache_key = f"performance_{strategy_id}"
            self.cache[cache_key] = (performance, datetime.utcnow())
            
        except Exception as e:
            logger.error(f"Error updating performance for strategy {strategy_id}: {e}")
    
    async def get_pnl_attribution(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get P&L attribution analysis"""
        try:
            # Check cache first
            cache_key = f"pnl_attr_{user_id}_{strategy_id}_{start_time.isoformat()}_{end_time.isoformat()}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Get historical data
            trades = await self.data_aggregator.get_trades_in_period(
                user_id, start_time, end_time, strategy_id
            )
            positions = await self.data_aggregator.get_positions_in_period(
                user_id, start_time, end_time, strategy_id
            )
            
            # Calculate attribution
            attribution_data = await self.pnl_attribution.calculate_attribution(
                trades, positions, start_time, end_time
            )
            
            # Cache result
            self.cache[cache_key] = (attribution_data, datetime.utcnow())
            
            return attribution_data
            
        except Exception as e:
            logger.error(f"Error getting P&L attribution: {e}")
            raise
    
    async def get_performance_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            # Check cache first
            cache_key = f"perf_{user_id}_{strategy_id}_{start_time.isoformat()}_{end_time.isoformat()}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Get performance data
            performance_data = await self.performance_calculator.calculate_comprehensive_metrics(
                user_id, start_time, end_time, strategy_id
            )
            
            # Cache result
            self.cache[cache_key] = (performance_data, datetime.utcnow())
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            raise
    
    async def get_real_time_risk_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get real-time risk metrics"""
        try:
            # Check cache first
            cache_key = f"risk_{user_id}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < 30:  # 30 second cache for real-time
                    return data
            
            # Get current portfolio data
            portfolio_data = await self.data_aggregator.get_user_portfolio_data(user_id)
            
            # Calculate real-time risk metrics
            risk_metrics = await self.risk_calculator.calculate_real_time_metrics(
                portfolio_data
            )
            
            # Cache result
            self.cache[cache_key] = (risk_metrics, datetime.utcnow())
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time risk metrics: {e}")
            raise
    
    async def get_performance_attribution(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get detailed performance attribution analysis"""
        try:
            # Get attribution data
            attribution_data = await self.performance_calculator.calculate_performance_attribution(
                user_id, start_time, end_time
            )
            
            return attribution_data
            
        except Exception as e:
            logger.error(f"Error getting performance attribution: {e}")
            raise