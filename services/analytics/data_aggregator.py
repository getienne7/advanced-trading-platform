"""
Data Aggregator - Centralized data access for analytics
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import aiohttp
import json

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Centralized data aggregator for analytics engine
    Handles data retrieval from various microservices and databases
    """
    
    def __init__(self):
        self.service_urls = {
            'trading_engine': 'http://localhost:8001',
            'risk_management': 'http://localhost:8002',
            'strategy_engine': 'http://localhost:8003',
            'exchange_gateway': 'http://localhost:8004',
            'ai_ml': 'http://localhost:8005'
        }
        self.cache = {}
        self.cache_ttl = 30  # seconds
    
    async def get_active_users(self) -> List[str]:
        """Get list of active users with positions or recent trades"""
        try:
            # Mock data - in real implementation, would query user service
            return ['user1', 'user2', 'user3']
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    async def get_active_portfolios(self) -> List[str]:
        """Get list of active portfolios"""
        try:
            # Mock data - in real implementation, would query portfolio service
            return ['portfolio1', 'portfolio2', 'portfolio3']
            
        except Exception as e:
            logger.error(f"Error getting active portfolios: {e}")
            return []
    
    async def get_active_strategies(self) -> List[str]:
        """Get list of active strategies"""
        try:
            # Mock data - in real implementation, would query strategy service
            return ['strategy1', 'strategy2', 'strategy3']
            
        except Exception as e:
            logger.error(f"Error getting active strategies: {e}")
            return []
    
    async def get_user_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get current positions for a user"""
        try:
            cache_key = f"positions_{user_id}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Mock data - in real implementation, would query trading engine
            positions = [
                {
                    'id': 'pos1',
                    'user_id': user_id,
                    'symbol': 'BTC/USDT',
                    'exchange': 'BINANCE',
                    'side': 'LONG',
                    'quantity': 0.5,
                    'entry_price': 45000.0,
                    'current_price': 47000.0,
                    'unrealized_pnl': 1000.0,
                    'margin': 5000.0,
                    'leverage': 10,
                    'strategy_id': 'strategy1'
                },
                {
                    'id': 'pos2',
                    'user_id': user_id,
                    'symbol': 'ETH/USDT',
                    'exchange': 'BINANCE',
                    'side': 'LONG',
                    'quantity': 2.0,
                    'entry_price': 3000.0,
                    'current_price': 3200.0,
                    'unrealized_pnl': 400.0,
                    'margin': 1200.0,
                    'leverage': 5,
                    'strategy_id': 'strategy2'
                }
            ]
            
            # Cache the result
            self.cache[cache_key] = (positions, datetime.utcnow())
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting user positions: {e}")
            return []
    
    async def get_recent_trades(self, user_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent trades for a user"""
        try:
            cache_key = f"trades_{user_id}_{hours}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Mock data - in real implementation, would query trading engine
            trades = [
                {
                    'id': 'trade1',
                    'user_id': user_id,
                    'symbol': 'BTC/USDT',
                    'exchange': 'BINANCE',
                    'side': 'BUY',
                    'quantity': 0.1,
                    'price': 46000.0,
                    'pnl': 100.0,
                    'fees': 5.0,
                    'strategy_id': 'strategy1',
                    'executed_at': (datetime.utcnow() - timedelta(hours=2)).isoformat()
                },
                {
                    'id': 'trade2',
                    'user_id': user_id,
                    'symbol': 'ETH/USDT',
                    'exchange': 'BINANCE',
                    'side': 'SELL',
                    'quantity': 0.5,
                    'price': 3100.0,
                    'pnl': -50.0,
                    'fees': 3.0,
                    'strategy_id': 'strategy2',
                    'executed_at': (datetime.utcnow() - timedelta(hours=1)).isoformat()
                }
            ]
            
            # Cache the result
            self.cache[cache_key] = (trades, datetime.utcnow())
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    async def get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio data"""
        try:
            cache_key = f"portfolio_{portfolio_id}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Mock data - in real implementation, would query multiple services
            portfolio_data = {
                'portfolio_id': portfolio_id,
                'user_id': 'user1',
                'total_value': 50000.0,
                'available_balance': 10000.0,
                'allocated_balance': 40000.0,
                'unrealized_pnl': 1400.0,
                'realized_pnl': 500.0,
                'positions': await self.get_user_positions('user1'),
                'historical_data': await self._get_historical_portfolio_data(portfolio_id)
            }
            
            # Cache the result
            self.cache[cache_key] = (portfolio_data, datetime.utcnow())
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return {}
    
    async def get_user_portfolio_data(self, user_id: str) -> Dict[str, Any]:
        """Get portfolio data for a user"""
        try:
            # Get user's positions
            positions = await self.get_user_positions(user_id)
            
            # Get recent trades
            recent_trades = await self.get_recent_trades(user_id)
            
            # Calculate portfolio metrics
            total_value = sum(
                pos.get('quantity', 0) * pos.get('current_price', 0)
                for pos in positions
            )
            
            total_unrealized_pnl = sum(
                pos.get('unrealized_pnl', 0)
                for pos in positions
            )
            
            total_realized_pnl = sum(
                trade.get('pnl', 0)
                for trade in recent_trades
            )
            
            return {
                'user_id': user_id,
                'total_value': total_value,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'positions': positions,
                'recent_trades': recent_trades,
                'historical_data': await self._get_historical_user_data(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting user portfolio data: {e}")
            return {}
    
    async def get_strategy_data(self, strategy_id: str) -> Dict[str, Any]:
        """Get comprehensive strategy data"""
        try:
            cache_key = f"strategy_{strategy_id}"
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                    return data
            
            # Mock data - in real implementation, would query strategy service
            strategy_data = {
                'strategy_id': strategy_id,
                'name': f'Strategy {strategy_id}',
                'type': 'ARBITRAGE',
                'active': True,
                'trades': [
                    {
                        'id': f'trade_{strategy_id}_1',
                        'symbol': 'BTC/USDT',
                        'side': 'BUY',
                        'quantity': 0.1,
                        'price': 46000.0,
                        'pnl': 100.0,
                        'fees': 5.0,
                        'executed_at': (datetime.utcnow() - timedelta(hours=2)).isoformat()
                    },
                    {
                        'id': f'trade_{strategy_id}_2',
                        'symbol': 'ETH/USDT',
                        'side': 'SELL',
                        'quantity': 0.5,
                        'price': 3100.0,
                        'pnl': -50.0,
                        'fees': 3.0,
                        'executed_at': (datetime.utcnow() - timedelta(hours=1)).isoformat()
                    }
                ],
                'performance_metrics': {
                    'total_pnl': 50.0,
                    'win_rate': 0.5,
                    'trade_count': 2
                }
            }
            
            # Cache the result
            self.cache[cache_key] = (strategy_data, datetime.utcnow())
            
            return strategy_data
            
        except Exception as e:
            logger.error(f"Error getting strategy data: {e}")
            return {}
    
    async def get_trades_in_period(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trades for a user in a specific time period"""
        try:
            # Mock data - in real implementation, would query database
            all_trades = [
                {
                    'id': 'trade1',
                    'user_id': user_id,
                    'symbol': 'BTC/USDT',
                    'exchange': 'BINANCE',
                    'side': 'BUY',
                    'quantity': 0.1,
                    'price': 46000.0,
                    'pnl': 100.0,
                    'fees': 5.0,
                    'strategy_id': 'strategy1',
                    'executed_at': start_time + timedelta(hours=1)
                },
                {
                    'id': 'trade2',
                    'user_id': user_id,
                    'symbol': 'ETH/USDT',
                    'exchange': 'BINANCE',
                    'side': 'SELL',
                    'quantity': 0.5,
                    'price': 3100.0,
                    'pnl': -50.0,
                    'fees': 3.0,
                    'strategy_id': 'strategy2',
                    'executed_at': start_time + timedelta(hours=2)
                }
            ]
            
            # Filter by time period and strategy
            filtered_trades = []
            for trade in all_trades:
                trade_time = trade['executed_at']
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                
                if start_time <= trade_time <= end_time:
                    if strategy_id is None or trade.get('strategy_id') == strategy_id:
                        # Convert datetime back to string for JSON serialization
                        trade['executed_at'] = trade_time.isoformat()
                        filtered_trades.append(trade)
            
            return filtered_trades
            
        except Exception as e:
            logger.error(f"Error getting trades in period: {e}")
            return []
    
    async def get_positions_in_period(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get positions for a user in a specific time period"""
        try:
            # For simplicity, return current positions
            # In real implementation, would get historical position snapshots
            positions = await self.get_user_positions(user_id)
            
            if strategy_id:
                positions = [
                    pos for pos in positions
                    if pos.get('strategy_id') == strategy_id
                ]
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions in period: {e}")
            return []
    
    async def _get_historical_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get historical data for portfolio risk calculations"""
        try:
            # Mock historical data
            return {
                'daily_returns': [0.01, -0.02, 0.015, -0.01, 0.005] * 50,  # 250 days
                'volatility_data': {
                    'BTC/USDT': 0.04,
                    'ETH/USDT': 0.05
                },
                'correlation_data': {
                    'BTC/USDT': {
                        'ETH/USDT': 0.7
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting historical portfolio data: {e}")
            return {}
    
    async def _get_historical_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get historical data for user risk calculations"""
        try:
            # Mock historical data
            return {
                'daily_pnl': [100, -200, 150, -100, 50] * 50,  # 250 days
                'position_history': [],
                'trade_history': []
            }
            
        except Exception as e:
            logger.error(f"Error getting historical user data: {e}")
            return {}
    
    async def _make_service_request(
        self,
        service: str,
        endpoint: str,
        method: str = 'GET',
        data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request to a microservice"""
        try:
            if service not in self.service_urls:
                logger.error(f"Unknown service: {service}")
                return None
            
            url = f"{self.service_urls[service]}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                if method == 'GET':
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                elif method == 'POST':
                    async with session.post(url, json=data) as response:
                        if response.status == 200:
                            return await response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error making service request to {service}{endpoint}: {e}")
            return None