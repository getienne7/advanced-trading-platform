"""
Risk Metrics Calculator - Real-time risk metrics calculation
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """
    Calculator for real-time risk metrics including VaR, correlation, and concentration risk
    """
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.lookback_days = 252  # 1 year of trading days
        self.correlation_threshold = 0.7
        self.concentration_threshold = 0.3  # 30% max concentration
    
    async def calculate_real_time_metrics(
        self,
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive real-time risk metrics"""
        try:
            if not portfolio_data or 'positions' not in portfolio_data:
                return self._empty_risk_metrics()
            
            positions = portfolio_data['positions']
            historical_data = portfolio_data.get('historical_data', {})
            
            # Calculate portfolio value and weights
            portfolio_metrics = await self._calculate_portfolio_metrics(positions)
            
            # Calculate VaR metrics
            var_metrics = await self._calculate_var_metrics(positions, historical_data)
            
            # Calculate correlation metrics
            correlation_metrics = await self._calculate_correlation_metrics(positions, historical_data)
            
            # Calculate concentration risk
            concentration_metrics = await self._calculate_concentration_metrics(positions)
            
            # Calculate leverage and margin metrics
            leverage_metrics = await self._calculate_leverage_metrics(positions)
            
            # Calculate stress test metrics
            stress_metrics = await self._calculate_stress_metrics(positions, historical_data)
            
            return {
                'portfolio_metrics': portfolio_metrics,
                'var_metrics': var_metrics,
                'correlation_metrics': correlation_metrics,
                'concentration_metrics': concentration_metrics,
                'leverage_metrics': leverage_metrics,
                'stress_metrics': stress_metrics,
                'risk_score': await self._calculate_overall_risk_score(
                    var_metrics, correlation_metrics, concentration_metrics, leverage_metrics
                ),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time risk metrics: {e}")
            raise
    
    async def _calculate_portfolio_metrics(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate basic portfolio metrics"""
        try:
            if not positions:
                return {}
            
            total_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            total_unrealized_pnl = sum(
                float(pos.get('unrealized_pnl', 0))
                for pos in positions
            )
            
            total_margin = sum(
                float(pos.get('margin', 0))
                for pos in positions
            )
            
            position_count = len(positions)
            
            # Calculate position weights
            weights = {}
            for pos in positions:
                symbol = pos.get('symbol', '')
                position_value = float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                weight = position_value / total_value if total_value > 0 else 0
                weights[symbol] = weight
            
            return {
                'total_value': total_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_margin': total_margin,
                'position_count': position_count,
                'position_weights': weights,
                'largest_position_weight': max(weights.values()) if weights else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def _calculate_var_metrics(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate Value at Risk metrics"""
        try:
            var_metrics = {}
            
            for confidence_level in self.confidence_levels:
                # Parametric VaR
                parametric_var = await self._calculate_parametric_var(
                    positions, historical_data, confidence_level
                )
                
                # Historical VaR
                historical_var = await self._calculate_historical_var(
                    positions, historical_data, confidence_level
                )
                
                # Monte Carlo VaR
                monte_carlo_var = await self._calculate_monte_carlo_var(
                    positions, historical_data, confidence_level
                )
                
                confidence_key = f"var_{int(confidence_level * 100)}"
                var_metrics[confidence_key] = {
                    'parametric': parametric_var,
                    'historical': historical_var,
                    'monte_carlo': monte_carlo_var,
                    'recommended': min(parametric_var, historical_var, monte_carlo_var)
                }
            
            # Expected Shortfall (Conditional VaR)
            expected_shortfall = await self._calculate_expected_shortfall(
                positions, historical_data, 0.95
            )
            
            var_metrics['expected_shortfall'] = expected_shortfall
            
            return var_metrics
            
        except Exception as e:
            logger.error(f"Error calculating VaR metrics: {e}")
            return {}
    
    async def _calculate_correlation_metrics(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate correlation and diversification metrics"""
        try:
            if len(positions) < 2:
                return {'correlation_matrix': {}, 'avg_correlation': 0, 'diversification_ratio': 1}
            
            symbols = [pos.get('symbol', '') for pos in positions]
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(symbols, historical_data)
            
            # Calculate average correlation
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        correlations.append(correlation_matrix[symbol1][symbol2])
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Calculate diversification ratio
            diversification_ratio = await self._calculate_diversification_ratio(
                positions, correlation_matrix
            )
            
            # Identify high correlation pairs
            high_correlation_pairs = []
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        corr = correlation_matrix[symbol1][symbol2]
                        if abs(corr) > self.correlation_threshold:
                            high_correlation_pairs.append({
                                'symbol1': symbol1,
                                'symbol2': symbol2,
                                'correlation': corr
                            })
            
            return {
                'correlation_matrix': correlation_matrix,
                'avg_correlation': avg_correlation,
                'diversification_ratio': diversification_ratio,
                'high_correlation_pairs': high_correlation_pairs,
                'correlation_risk_score': min(abs(avg_correlation) * 10, 10)  # 0-10 scale
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation metrics: {e}")
            return {}
    
    async def _calculate_concentration_metrics(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate concentration risk metrics"""
        try:
            if not positions:
                return {}
            
            # Calculate position weights
            total_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            weights = []
            position_weights = {}
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                position_value = float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                weight = position_value / total_value if total_value > 0 else 0
                weights.append(weight)
                position_weights[symbol] = weight
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            hhi = sum(w**2 for w in weights)
            
            # Calculate concentration ratio (top 3 positions)
            sorted_weights = sorted(weights, reverse=True)
            top3_concentration = sum(sorted_weights[:3])
            
            # Calculate effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0
            
            # Identify concentrated positions
            concentrated_positions = [
                {'symbol': symbol, 'weight': weight}
                for symbol, weight in position_weights.items()
                if weight > self.concentration_threshold
            ]
            
            # Calculate concentration risk score
            concentration_risk_score = min(hhi * 10, 10)  # 0-10 scale
            
            return {
                'herfindahl_index': hhi,
                'top3_concentration': top3_concentration,
                'effective_positions': effective_positions,
                'concentrated_positions': concentrated_positions,
                'concentration_risk_score': concentration_risk_score,
                'position_weights': position_weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {}
    
    async def _calculate_leverage_metrics(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate leverage and margin metrics"""
        try:
            total_notional = 0
            total_margin = 0
            leveraged_positions = []
            
            for pos in positions:
                quantity = float(pos.get('quantity', 0))
                current_price = float(pos.get('current_price', 0))
                margin = float(pos.get('margin', 0))
                leverage = int(pos.get('leverage', 1))
                
                notional_value = quantity * current_price
                total_notional += notional_value
                total_margin += margin
                
                if leverage > 1:
                    leveraged_positions.append({
                        'symbol': pos.get('symbol', ''),
                        'leverage': leverage,
                        'notional_value': notional_value,
                        'margin': margin
                    })
            
            # Calculate overall leverage ratio
            overall_leverage = total_notional / total_margin if total_margin > 0 else 1
            
            # Calculate margin utilization
            available_margin = total_margin * 0.8  # Assume 80% margin utilization limit
            margin_utilization = (total_margin - available_margin) / total_margin if total_margin > 0 else 0
            
            # Calculate leverage risk score
            leverage_risk_score = min(overall_leverage / 10 * 10, 10)  # 0-10 scale
            
            return {
                'overall_leverage': overall_leverage,
                'total_notional': total_notional,
                'total_margin': total_margin,
                'margin_utilization': margin_utilization,
                'leveraged_positions': leveraged_positions,
                'leverage_risk_score': leverage_risk_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating leverage metrics: {e}")
            return {}
    
    async def _calculate_stress_metrics(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate stress test metrics"""
        try:
            stress_scenarios = {
                'market_crash_10pct': -0.10,
                'market_crash_20pct': -0.20,
                'volatility_spike_2x': 2.0,
                'correlation_spike_90pct': 0.90
            }
            
            stress_results = {}
            
            for scenario, shock in stress_scenarios.items():
                if 'crash' in scenario:
                    # Price shock scenario
                    stressed_pnl = await self._calculate_price_shock_pnl(positions, shock)
                elif 'volatility' in scenario:
                    # Volatility shock scenario
                    stressed_var = await self._calculate_volatility_shock_var(positions, historical_data, shock)
                    stressed_pnl = -stressed_var  # Approximate P&L impact
                else:
                    # Correlation shock scenario
                    stressed_pnl = await self._calculate_correlation_shock_pnl(positions, historical_data, shock)
                
                stress_results[scenario] = {
                    'stressed_pnl': stressed_pnl,
                    'stress_ratio': abs(stressed_pnl) / sum(
                        float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                        for pos in positions
                    ) if positions else 0
                }
            
            # Calculate overall stress score
            worst_case_pnl = min(result['stressed_pnl'] for result in stress_results.values())
            stress_score = min(abs(worst_case_pnl) / 10000, 10)  # 0-10 scale
            
            return {
                'stress_scenarios': stress_results,
                'worst_case_pnl': worst_case_pnl,
                'stress_score': stress_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating stress metrics: {e}")
            return {}
    
    async def _calculate_parametric_var(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        confidence_level: float
    ) -> float:
        """Calculate parametric VaR using normal distribution assumption"""
        try:
            # Mock calculation - in real implementation, would use historical volatility
            portfolio_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            # Assume 2% daily volatility
            daily_volatility = 0.02
            z_score = norm.ppf(confidence_level)
            
            var = portfolio_value * daily_volatility * z_score
            return var
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return 0
    
    async def _calculate_historical_var(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        confidence_level: float
    ) -> float:
        """Calculate historical VaR using historical simulation"""
        try:
            # Mock calculation - in real implementation, would use historical returns
            portfolio_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            # Mock historical returns (normally distributed)
            np.random.seed(42)
            historical_returns = np.random.normal(0, 0.02, 252)  # 1 year of daily returns
            
            portfolio_returns = historical_returns * portfolio_value
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(portfolio_returns, var_percentile)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            return 0
    
    async def _calculate_monte_carlo_var(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        confidence_level: float
    ) -> float:
        """Calculate Monte Carlo VaR using simulation"""
        try:
            # Mock calculation - in real implementation, would run Monte Carlo simulation
            portfolio_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            # Mock Monte Carlo simulation
            np.random.seed(42)
            num_simulations = 10000
            simulated_returns = np.random.normal(0, 0.02, num_simulations)
            
            portfolio_returns = simulated_returns * portfolio_value
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(portfolio_returns, var_percentile)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0
    
    async def _calculate_expected_shortfall(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        confidence_level: float
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            # Mock calculation
            portfolio_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            np.random.seed(42)
            historical_returns = np.random.normal(0, 0.02, 252)
            portfolio_returns = historical_returns * portfolio_value
            
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(portfolio_returns, var_percentile)
            
            # Expected shortfall is the average of losses beyond VaR
            tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else 0
            
            return abs(expected_shortfall)
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0
    
    async def _calculate_correlation_matrix(
        self,
        symbols: List[str],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets"""
        try:
            # Mock correlation matrix
            correlation_matrix = {}
            
            for symbol1 in symbols:
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Mock correlation (in real implementation, would calculate from historical data)
                        np.random.seed(hash(symbol1 + symbol2) % 2**32)
                        correlation = np.random.uniform(-0.5, 0.8)
                        correlation_matrix[symbol1][symbol2] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    async def _calculate_diversification_ratio(
        self,
        positions: List[Dict[str, Any]],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate diversification ratio"""
        try:
            if len(positions) < 2:
                return 1.0
            
            # Mock calculation
            # Diversification ratio = weighted average volatility / portfolio volatility
            # Higher ratio indicates better diversification
            
            # For simplicity, return a mock value based on number of positions
            num_positions = len(positions)
            diversification_ratio = min(np.sqrt(num_positions), 3.0)  # Cap at 3.0
            
            return diversification_ratio
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0
    
    async def _calculate_price_shock_pnl(
        self,
        positions: List[Dict[str, Any]],
        shock_percentage: float
    ) -> float:
        """Calculate P&L under price shock scenario"""
        try:
            total_shocked_pnl = 0
            
            for pos in positions:
                quantity = float(pos.get('quantity', 0))
                current_price = float(pos.get('current_price', 0))
                side = pos.get('side', 'LONG')
                
                shocked_price = current_price * (1 + shock_percentage)
                
                if side == 'LONG':
                    shocked_pnl = quantity * (shocked_price - current_price)
                else:  # SHORT
                    shocked_pnl = quantity * (current_price - shocked_price)
                
                total_shocked_pnl += shocked_pnl
            
            return total_shocked_pnl
            
        except Exception as e:
            logger.error(f"Error calculating price shock P&L: {e}")
            return 0
    
    async def _calculate_volatility_shock_var(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        volatility_multiplier: float
    ) -> float:
        """Calculate VaR under volatility shock scenario"""
        try:
            # Mock calculation - multiply base VaR by volatility multiplier
            base_var = await self._calculate_parametric_var(positions, historical_data, 0.95)
            shocked_var = base_var * volatility_multiplier
            
            return shocked_var
            
        except Exception as e:
            logger.error(f"Error calculating volatility shock VaR: {e}")
            return 0
    
    async def _calculate_correlation_shock_pnl(
        self,
        positions: List[Dict[str, Any]],
        historical_data: Dict[str, Any],
        correlation_level: float
    ) -> float:
        """Calculate P&L under correlation shock scenario"""
        try:
            # Mock calculation - assume higher correlation increases risk
            portfolio_value = sum(
                float(pos.get('quantity', 0)) * float(pos.get('current_price', 0))
                for pos in positions
            )
            
            # Higher correlation means less diversification benefit
            correlation_impact = correlation_level * 0.05  # 5% impact at 100% correlation
            shocked_pnl = -portfolio_value * correlation_impact
            
            return shocked_pnl
            
        except Exception as e:
            logger.error(f"Error calculating correlation shock P&L: {e}")
            return 0
    
    async def _calculate_overall_risk_score(
        self,
        var_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        concentration_metrics: Dict[str, Any],
        leverage_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall risk score (0-10 scale)"""
        try:
            scores = []
            
            # VaR score (normalized)
            if 'var_95' in var_metrics and 'recommended' in var_metrics['var_95']:
                var_score = min(var_metrics['var_95']['recommended'] / 10000, 10)
                scores.append(var_score)
            
            # Correlation score
            if 'correlation_risk_score' in correlation_metrics:
                scores.append(correlation_metrics['correlation_risk_score'])
            
            # Concentration score
            if 'concentration_risk_score' in concentration_metrics:
                scores.append(concentration_metrics['concentration_risk_score'])
            
            # Leverage score
            if 'leverage_risk_score' in leverage_metrics:
                scores.append(leverage_metrics['leverage_risk_score'])
            
            # Calculate weighted average
            if scores:
                overall_score = np.mean(scores)
                return min(overall_score, 10)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 0
    
    def _empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics structure"""
        return {
            'portfolio_metrics': {},
            'var_metrics': {},
            'correlation_metrics': {},
            'concentration_metrics': {},
            'leverage_metrics': {},
            'stress_metrics': {},
            'risk_score': 0
        }