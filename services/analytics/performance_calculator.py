"""
Performance Calculator - Comprehensive performance metrics calculation
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Calculator for comprehensive performance metrics including Sharpe, Sortino, Calmar ratios
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days_per_year = 252
    
    async def calculate_comprehensive_metrics(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            # Get performance data (this would typically come from data aggregator)
            performance_data = await self._get_performance_data(
                user_id, start_time, end_time, strategy_id
            )
            
            if not performance_data:
                return self._empty_metrics()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(performance_data)
            
            # Calculate basic metrics
            basic_metrics = await self._calculate_basic_metrics(df)
            
            # Calculate risk-adjusted metrics
            risk_metrics = await self._calculate_risk_metrics(df)
            
            # Calculate drawdown metrics
            drawdown_metrics = await self._calculate_drawdown_metrics(df)
            
            # Calculate trade-based metrics
            trade_metrics = await self._calculate_trade_metrics(df)
            
            # Calculate time-based metrics
            time_metrics = await self._calculate_time_metrics(df, start_time, end_time)
            
            return {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_days': (end_time - start_time).days
                },
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'drawdown_metrics': drawdown_metrics,
                'trade_metrics': trade_metrics,
                'time_metrics': time_metrics,
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            raise
    
    async def calculate_strategy_performance(
        self,
        strategy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics for a specific strategy"""
        try:
            if not strategy_data or 'trades' not in strategy_data:
                return self._empty_strategy_metrics()
            
            trades_df = pd.DataFrame(strategy_data['trades'])
            
            if trades_df.empty:
                return self._empty_strategy_metrics()
            
            # Calculate strategy-specific metrics
            total_pnl = trades_df['pnl'].sum()
            total_fees = trades_df['fees'].sum()
            net_pnl = total_pnl - total_fees
            
            win_trades = trades_df[trades_df['pnl'] > 0]
            loss_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
            profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if loss_trades['pnl'].sum() != 0 else float('inf')
            
            # Calculate returns series for risk metrics
            if 'executed_at' in trades_df.columns:
                trades_df['executed_at'] = pd.to_datetime(trades_df['executed_at'])
                trades_df = trades_df.sort_values('executed_at')
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                # Calculate daily returns
                daily_returns = self._calculate_daily_returns(trades_df)
                
                # Risk metrics
                volatility = daily_returns.std() * np.sqrt(self.trading_days_per_year) if len(daily_returns) > 1 else 0
                sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
                sortino_ratio = self._calculate_sortino_ratio(daily_returns)
                
                # Drawdown metrics
                drawdown_metrics = self._calculate_drawdown_from_cumulative(trades_df['cumulative_pnl'])
            else:
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
                drawdown_metrics = {'max_drawdown': 0, 'max_drawdown_pct': 0, 'current_drawdown': 0}
            
            return {
                'strategy_id': strategy_data.get('strategy_id'),
                'total_pnl': float(total_pnl),
                'net_pnl': float(net_pnl),
                'total_fees': float(total_fees),
                'trade_count': len(trades_df),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor),
                'best_trade': float(trades_df['pnl'].max()),
                'worst_trade': float(trades_df['pnl'].min()),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(drawdown_metrics['max_drawdown']),
                'max_drawdown_pct': float(drawdown_metrics['max_drawdown_pct']),
                'current_drawdown': float(drawdown_metrics['current_drawdown'])
            }
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return self._empty_strategy_metrics()
    
    async def calculate_performance_attribution(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate detailed performance attribution analysis"""
        try:
            # Get attribution data
            attribution_data = await self._get_attribution_data(user_id, start_time, end_time)
            
            if not attribution_data:
                return {}
            
            # Calculate attribution by different factors
            factor_attribution = {
                'asset_allocation': await self._calculate_asset_allocation_attribution(attribution_data),
                'security_selection': await self._calculate_security_selection_attribution(attribution_data),
                'timing': await self._calculate_timing_attribution(attribution_data),
                'interaction': await self._calculate_interaction_attribution(attribution_data)
            }
            
            return {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'factor_attribution': factor_attribution,
                'total_attribution': sum(
                    attr.get('contribution', 0) 
                    for attr in factor_attribution.values()
                ),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            raise
    
    async def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        if df.empty:
            return {}
        
        total_return = df['pnl'].sum() if 'pnl' in df.columns else 0
        trade_count = len(df)
        
        if 'executed_at' in df.columns:
            df['executed_at'] = pd.to_datetime(df['executed_at'])
            period_days = (df['executed_at'].max() - df['executed_at'].min()).days
            annualized_return = (total_return / period_days * 365) if period_days > 0 else 0
        else:
            annualized_return = 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'trade_count': trade_count,
            'avg_trade_return': float(total_return / trade_count) if trade_count > 0 else 0
        }
    
    async def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        if df.empty or 'pnl' not in df.columns:
            return {}
        
        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(df)
        
        if len(daily_returns) < 2:
            return {}
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(daily_returns.cumsum())
        calmar_ratio = (daily_returns.mean() * self.trading_days_per_year) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio)
        }
    
    async def _calculate_drawdown_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown-related metrics"""
        if df.empty or 'pnl' not in df.columns:
            return {}
        
        # Calculate cumulative returns
        cumulative_returns = df['pnl'].cumsum()
        
        # Calculate drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max()) * 100 if running_max.max() != 0 else 0
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        # Calculate recovery time
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown_pct),
            'current_drawdown': float(current_drawdown),
            'avg_recovery_time_days': float(recovery_time)
        }
    
    async def _calculate_trade_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-based metrics"""
        if df.empty or 'pnl' not in df.columns:
            return {}
        
        win_trades = df[df['pnl'] > 0]
        loss_trades = df[df['pnl'] < 0]
        
        win_rate = len(win_trades) / len(df) if len(df) > 0 else 0
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        
        profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if loss_trades['pnl'].sum() != 0 else float('inf')
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins(df['pnl'])
        consecutive_losses = self._calculate_consecutive_losses(df['pnl'])
        
        return {
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'best_trade': float(df['pnl'].max()),
            'worst_trade': float(df['pnl'].min()),
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    
    async def _calculate_time_metrics(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Calculate time-based metrics"""
        if df.empty:
            return {}
        
        period_days = (end_time - start_time).days
        
        if 'executed_at' in df.columns:
            df['executed_at'] = pd.to_datetime(df['executed_at'])
            
            # Calculate average trade duration (if we have position data)
            avg_trade_duration = 0  # Would need position open/close times
            
            # Trading frequency
            trades_per_day = len(df) / period_days if period_days > 0 else 0
            
            # Time-based performance
            hourly_performance = self._calculate_hourly_performance(df)
            daily_performance = self._calculate_daily_performance(df)
        else:
            avg_trade_duration = 0
            trades_per_day = 0
            hourly_performance = {}
            daily_performance = {}
        
        return {
            'avg_trade_duration_hours': float(avg_trade_duration),
            'trades_per_day': float(trades_per_day),
            'hourly_performance': hourly_performance,
            'daily_performance': daily_performance
        }
    
    def _calculate_daily_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from trade data"""
        if 'executed_at' not in df.columns:
            return pd.Series()
        
        df['executed_at'] = pd.to_datetime(df['executed_at'])
        daily_pnl = df.groupby(df['executed_at'].dt.date)['pnl'].sum()
        
        return daily_pnl
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days_per_year) if excess_returns.std() != 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        return (excess_returns.mean() / downside_deviation) * np.sqrt(self.trading_days_per_year) if downside_deviation != 0 else 0
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        return drawdown.min()
    
    def _calculate_drawdown_from_cumulative(self, cumulative_pnl: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics from cumulative P&L"""
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max()) * 100 if running_max.max() != 0 else 0
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'current_drawdown': current_drawdown
        }
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        # Simplified recovery time calculation
        recovery_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_periods.append(i - drawdown_start)
        
        return np.mean(recovery_periods) if recovery_periods else 0
    
    def _calculate_consecutive_wins(self, pnl_series: pd.Series) -> int:
        """Calculate maximum consecutive wins"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_series:
            if pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, pnl_series: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnl_series:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_hourly_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance by hour of day"""
        if 'executed_at' not in df.columns:
            return {}
        
        df['hour'] = df['executed_at'].dt.hour
        hourly_pnl = df.groupby('hour')['pnl'].sum()
        
        return {str(hour): float(pnl) for hour, pnl in hourly_pnl.items()}
    
    def _calculate_daily_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance by day of week"""
        if 'executed_at' not in df.columns:
            return {}
        
        df['day_of_week'] = df['executed_at'].dt.day_name()
        daily_pnl = df.groupby('day_of_week')['pnl'].sum()
        
        return {day: float(pnl) for day, pnl in daily_pnl.items()}
    
    async def _get_performance_data(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get performance data from data source"""
        # Mock data for now - in real implementation, this would query the database
        return []
    
    async def _get_attribution_data(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get attribution data from data source"""
        # Mock data for now
        return {}
    
    async def _calculate_asset_allocation_attribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate asset allocation attribution"""
        return {'contribution': 0.0, 'description': 'Asset allocation effect'}
    
    async def _calculate_security_selection_attribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate security selection attribution"""
        return {'contribution': 0.0, 'description': 'Security selection effect'}
    
    async def _calculate_timing_attribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate timing attribution"""
        return {'contribution': 0.0, 'description': 'Timing effect'}
    
    async def _calculate_interaction_attribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate interaction attribution"""
        return {'contribution': 0.0, 'description': 'Interaction effect'}
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'basic_metrics': {},
            'risk_metrics': {},
            'drawdown_metrics': {},
            'trade_metrics': {},
            'time_metrics': {}
        }
    
    def _empty_strategy_metrics(self) -> Dict[str, Any]:
        """Return empty strategy metrics structure"""
        return {
            'total_pnl': 0,
            'net_pnl': 0,
            'trade_count': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }