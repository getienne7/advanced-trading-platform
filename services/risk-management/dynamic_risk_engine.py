"""
Dynamic Risk Calculation Engine for Advanced Trading Platform.
Implements Monte Carlo simulation, historical simulation, and parametric VaR models.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import math
import statistics
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("dynamic-risk-engine")

class RiskModel(str, Enum):
    """Risk calculation models."""
    MONTE_CARLO = "monte_carlo"
    HISTORICAL_SIMULATION = "historical_simulation"
    PARAMETRIC = "parametric"
    HYBRID = "hybrid"

class AssetClass(str, Enum):
    """Asset classes for risk modeling."""
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    EQUITY = "equity"
    COMMODITY = "commodity"
    BOND = "bond"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall_95: float  # Expected Shortfall (CVaR) at 95%
    expected_shortfall_99: float  # Expected Shortfall (CVaR) at 99%
    volatility: float  # Annualized volatility
    skewness: float  # Distribution skewness
    kurtosis: float  # Distribution kurtosis
    max_drawdown: float  # Maximum drawdown
    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return
    calmar_ratio: float  # Return to max drawdown ratio
    beta: Optional[float] = None  # Market beta
    correlation_to_market: Optional[float] = None
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_var_95: float
    total_var_99: float
    diversification_ratio: float
    concentration_risk: float
    correlation_matrix: np.ndarray
    component_var: Dict[str, float]  # VaR contribution by asset
    marginal_var: Dict[str, float]   # Marginal VaR by asset
    incremental_var: Dict[str, float]  # Incremental VaR by asset
    risk_budget: Dict[str, float]    # Risk budget allocation
    active_risk: float  # Tracking error
    information_ratio: float
    portfolio_beta: float
    systematic_risk: float
    idiosyncratic_risk: float

class RiskConfiguration(BaseModel):
    """Configuration for risk calculations."""
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="VaR confidence levels")
    time_horizon_days: int = Field(default=1, description="Risk time horizon in days")
    lookback_days: int = Field(default=252, description="Historical lookback period")
    monte_carlo_simulations: int = Field(default=10000, description="Number of Monte Carlo simulations")
    rebalancing_frequency: str = Field(default="daily", description="Portfolio rebalancing frequency")
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate for Sharpe ratio")
    market_benchmark: str = Field(default="BTC", description="Market benchmark for beta calculation")
    correlation_threshold: float = Field(default=0.7, description="High correlation threshold")
    concentration_limit: float = Field(default=0.25, description="Maximum position concentration")

class DynamicRiskEngine:
    """Advanced risk calculation engine with multiple methodologies."""
    
    def __init__(self, config: RiskConfiguration = None):
        self.config = config or RiskConfiguration()
        self.market_data_cache = {}
        self.correlation_cache = {}
        self.volatility_models = {}
        
        # Risk model parameters
        self.garch_params = {}
        self.ewma_lambda = 0.94  # Exponentially weighted moving average decay
        self.var_scaling_factors = {
            1: 1.0,      # 1 day
            5: 2.236,    # 1 week (sqrt(5))
            22: 4.690,   # 1 month (sqrt(22))
            252: 15.875  # 1 year (sqrt(252))
        }
        
        logger.info("Dynamic Risk Engine initialized", config=self.config.dict())
    
    async def calculate_var(self, 
                          returns: np.ndarray, 
                          position_value: float,
                          asset_class: AssetClass = AssetClass.CRYPTOCURRENCY,
                          model: RiskModel = RiskModel.HYBRID) -> RiskMetrics:
        """Calculate Value at Risk using specified model."""
        try:
            if len(returns) < 30:
                raise ValueError("Insufficient data for VaR calculation (minimum 30 observations)")
            
            # Clean and validate returns
            returns = self._clean_returns(returns)
            
            # Calculate VaR using different models
            if model == RiskModel.MONTE_CARLO:
                risk_metrics = await self._monte_carlo_var(returns, position_value, asset_class)
            elif model == RiskModel.HISTORICAL_SIMULATION:
                risk_metrics = await self._historical_simulation_var(returns, position_value)
            elif model == RiskModel.PARAMETRIC:
                risk_metrics = await self._parametric_var(returns, position_value, asset_class)
            else:  # HYBRID
                risk_metrics = await self._hybrid_var(returns, position_value, asset_class)
            
            logger.info("VaR calculated successfully",
                       model=model.value,
                       var_95=risk_metrics.var_95,
                       var_99=risk_metrics.var_99,
                       position_value=position_value)
            
            return risk_metrics
            
        except Exception as e:
            logger.error("VaR calculation failed", error=str(e), model=model.value)
            raise
    
    async def _monte_carlo_var(self, 
                             returns: np.ndarray, 
                             position_value: float,
                             asset_class: AssetClass) -> RiskMetrics:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            # Estimate distribution parameters
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            # Choose appropriate distribution based on asset class and moments
            if abs(skew) > 0.5 or abs(kurt) > 3:
                # Use skewed t-distribution for fat tails
                distribution = self._fit_skewed_t_distribution(returns)
            else:
                # Use normal distribution
                distribution = stats.norm(mu, sigma)
            
            # Generate Monte Carlo simulations
            n_sims = self.config.monte_carlo_simulations
            
            if hasattr(distribution, 'rvs'):
                simulated_returns = distribution.rvs(size=n_sims)
            else:
                # Fallback to normal distribution
                simulated_returns = np.random.normal(mu, sigma, n_sims)
            
            # Scale for time horizon
            time_scaling = math.sqrt(self.config.time_horizon_days)
            simulated_returns *= time_scaling
            
            # Calculate P&L scenarios
            simulated_pnl = simulated_returns * position_value
            
            # Calculate VaR and Expected Shortfall
            var_95 = -np.percentile(simulated_pnl, 5)
            var_99 = -np.percentile(simulated_pnl, 1)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = -np.mean(simulated_pnl[simulated_pnl <= -var_95])
            es_99 = -np.mean(simulated_pnl[simulated_pnl <= -var_99])
            
            # Additional risk metrics
            volatility = sigma * math.sqrt(252)  # Annualized
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Confidence intervals
            ci_lower = np.percentile(simulated_pnl, 2.5)
            ci_upper = np.percentile(simulated_pnl, 97.5)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                volatility=volatility,
                skewness=skew,
                kurtosis=kurt,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                confidence_interval=(ci_lower, ci_upper)
            )
            
        except Exception as e:
            logger.error("Monte Carlo VaR calculation failed", error=str(e))
            raise
    
    async def _historical_simulation_var(self, 
                                       returns: np.ndarray, 
                                       position_value: float) -> RiskMetrics:
        """Calculate VaR using historical simulation."""
        try:
            # Use actual historical returns
            historical_pnl = returns * position_value
            
            # Scale for time horizon if needed
            if self.config.time_horizon_days > 1:
                # Bootstrap or scale returns
                time_scaling = math.sqrt(self.config.time_horizon_days)
                historical_pnl *= time_scaling
            
            # Sort P&L scenarios
            sorted_pnl = np.sort(historical_pnl)
            n_obs = len(sorted_pnl)
            
            # Calculate VaR percentiles
            var_95_idx = int(n_obs * 0.05)
            var_99_idx = int(n_obs * 0.01)
            
            var_95 = -sorted_pnl[var_95_idx] if var_95_idx < n_obs else -sorted_pnl[0]
            var_99 = -sorted_pnl[var_99_idx] if var_99_idx < n_obs else -sorted_pnl[0]
            
            # Expected Shortfall
            es_95 = -np.mean(sorted_pnl[:var_95_idx+1]) if var_95_idx >= 0 else 0
            es_99 = -np.mean(sorted_pnl[:var_99_idx+1]) if var_99_idx >= 0 else 0
            
            # Calculate additional metrics
            volatility = np.std(returns, ddof=1) * math.sqrt(252)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                confidence_interval=(np.percentile(historical_pnl, 2.5), np.percentile(historical_pnl, 97.5))
            )
            
        except Exception as e:
            logger.error("Historical simulation VaR calculation failed", error=str(e))
            raise
    
    async def _parametric_var(self, 
                            returns: np.ndarray, 
                            position_value: float,
                            asset_class: AssetClass) -> RiskMetrics:
        """Calculate VaR using parametric approach."""
        try:
            # Calculate basic statistics
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            
            # Adjust for asset class characteristics
            if asset_class == AssetClass.CRYPTOCURRENCY:
                # Crypto typically has higher volatility and fat tails
                fat_tail_adjustment = 1.2
                sigma *= fat_tail_adjustment
            
            # Scale for time horizon
            time_scaling = math.sqrt(self.config.time_horizon_days)
            scaled_sigma = sigma * time_scaling
            scaled_mu = mu * self.config.time_horizon_days
            
            # Calculate VaR using normal distribution
            z_95 = stats.norm.ppf(0.05)  # -1.645
            z_99 = stats.norm.ppf(0.01)  # -2.326
            
            var_95 = -(scaled_mu + z_95 * scaled_sigma) * position_value
            var_99 = -(scaled_mu + z_99 * scaled_sigma) * position_value
            
            # Expected Shortfall for normal distribution
            es_95 = scaled_sigma * stats.norm.pdf(z_95) / 0.05 * position_value
            es_99 = scaled_sigma * stats.norm.pdf(z_99) / 0.01 * position_value
            
            # Additional metrics
            volatility = sigma * math.sqrt(252)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                confidence_interval=(
                    (scaled_mu + stats.norm.ppf(0.025) * scaled_sigma) * position_value,
                    (scaled_mu + stats.norm.ppf(0.975) * scaled_sigma) * position_value
                )
            )
            
        except Exception as e:
            logger.error("Parametric VaR calculation failed", error=str(e))
            raise
    
    async def _hybrid_var(self, 
                        returns: np.ndarray, 
                        position_value: float,
                        asset_class: AssetClass) -> RiskMetrics:
        """Calculate VaR using hybrid approach combining multiple methods."""
        try:
            # Calculate VaR using all three methods
            mc_metrics = await self._monte_carlo_var(returns, position_value, asset_class)
            hist_metrics = await self._historical_simulation_var(returns, position_value)
            param_metrics = await self._parametric_var(returns, position_value, asset_class)
            
            # Weight the results based on data characteristics
            n_obs = len(returns)
            skew = abs(stats.skew(returns))
            kurt = abs(stats.kurtosis(returns))
            
            # Determine weights based on data quality and characteristics
            if n_obs < 100:
                # Limited data - favor parametric
                weights = {'parametric': 0.5, 'monte_carlo': 0.3, 'historical': 0.2}
            elif skew > 1.0 or kurt > 5.0:
                # Fat tails - favor Monte Carlo
                weights = {'monte_carlo': 0.5, 'historical': 0.3, 'parametric': 0.2}
            else:
                # Normal conditions - balanced approach
                weights = {'historical': 0.4, 'monte_carlo': 0.35, 'parametric': 0.25}
            
            # Combine results
            var_95 = (weights['monte_carlo'] * mc_metrics.var_95 + 
                     weights['historical'] * hist_metrics.var_95 + 
                     weights['parametric'] * param_metrics.var_95)
            
            var_99 = (weights['monte_carlo'] * mc_metrics.var_99 + 
                     weights['historical'] * hist_metrics.var_99 + 
                     weights['parametric'] * param_metrics.var_99)
            
            es_95 = (weights['monte_carlo'] * mc_metrics.expected_shortfall_95 + 
                    weights['historical'] * hist_metrics.expected_shortfall_95 + 
                    weights['parametric'] * param_metrics.expected_shortfall_95)
            
            es_99 = (weights['monte_carlo'] * mc_metrics.expected_shortfall_99 + 
                    weights['historical'] * hist_metrics.expected_shortfall_99 + 
                    weights['parametric'] * param_metrics.expected_shortfall_99)
            
            # Use Monte Carlo for other metrics (most comprehensive)
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                volatility=mc_metrics.volatility,
                skewness=mc_metrics.skewness,
                kurtosis=mc_metrics.kurtosis,
                max_drawdown=mc_metrics.max_drawdown,
                sharpe_ratio=mc_metrics.sharpe_ratio,
                sortino_ratio=mc_metrics.sortino_ratio,
                calmar_ratio=mc_metrics.calmar_ratio,
                confidence_interval=mc_metrics.confidence_interval
            )
            
        except Exception as e:
            logger.error("Hybrid VaR calculation failed", error=str(e))
            raise
    
    async def calculate_portfolio_risk(self, 
                                     positions: Dict[str, float],
                                     returns_matrix: np.ndarray,
                                     symbols: List[str]) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            if len(positions) != len(symbols) or returns_matrix.shape[1] != len(symbols):
                raise ValueError("Positions, symbols, and returns matrix dimensions must match")
            
            # Portfolio weights
            total_value = sum(abs(v) for v in positions.values())
            weights = np.array([positions[symbol] / total_value for symbol in symbols])
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
            
            # Portfolio variance and volatility
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = math.sqrt(portfolio_variance * 252)  # Annualized
            
            # Correlation matrix
            corr_matrix = np.corrcoef(returns_matrix.T)
            
            # Portfolio VaR
            portfolio_var_95 = -np.percentile(portfolio_returns, 5) * total_value
            portfolio_var_99 = -np.percentile(portfolio_returns, 1) * total_value
            
            # Component VaR (contribution to portfolio VaR)
            component_var = {}
            marginal_var = {}
            incremental_var = {}
            
            for i, symbol in enumerate(symbols):
                # Marginal VaR
                portfolio_cov = np.dot(cov_matrix[i], weights)
                marginal_var[symbol] = portfolio_cov / portfolio_variance * portfolio_var_95
                
                # Component VaR
                component_var[symbol] = weights[i] * marginal_var[symbol]
                
                # Incremental VaR (approximate)
                incremental_var[symbol] = marginal_var[symbol] * abs(positions[symbol])
            
            # Risk budget allocation
            total_component_var = sum(abs(cv) for cv in component_var.values())
            risk_budget = {symbol: abs(cv) / total_component_var 
                          for symbol, cv in component_var.items()}
            
            # Diversification ratio
            weighted_avg_vol = sum(weights[i] * math.sqrt(cov_matrix[i, i] * 252) 
                                 for i in range(len(weights)))
            diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            # Concentration risk (Herfindahl index)
            concentration_risk = sum(w**2 for w in weights)
            
            # Calculate systematic vs idiosyncratic risk (simplified)
            # This would typically require a market benchmark
            systematic_risk = portfolio_volatility * 0.7  # Simplified assumption
            idiosyncratic_risk = portfolio_volatility * 0.3
            
            return PortfolioRisk(
                total_var_95=portfolio_var_95,
                total_var_99=portfolio_var_99,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                correlation_matrix=corr_matrix,
                component_var=component_var,
                marginal_var=marginal_var,
                incremental_var=incremental_var,
                risk_budget=risk_budget,
                active_risk=portfolio_volatility,  # Simplified
                information_ratio=0.0,  # Would need benchmark
                portfolio_beta=1.0,  # Would need market data
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk
            )
            
        except Exception as e:
            logger.error("Portfolio risk calculation failed", error=str(e))
            raise
    
    # Helper methods
    
    def _clean_returns(self, returns: np.ndarray) -> np.ndarray:
        """Clean and validate returns data."""
        # Remove NaN and infinite values
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        # Remove extreme outliers (beyond 10 standard deviations)
        std_dev = np.std(returns)
        mean_return = np.mean(returns)
        outlier_threshold = 10 * std_dev
        
        returns = returns[np.abs(returns - mean_return) <= outlier_threshold]
        
        return returns
    
    def _fit_skewed_t_distribution(self, returns: np.ndarray):
        """Fit skewed t-distribution to returns."""
        try:
            # Use scipy's skewed t-distribution
            params = stats.skewnorm.fit(returns)
            return stats.skewnorm(*params)
        except:
            # Fallback to normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            return stats.norm(mu, sigma)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.config.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * math.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - self.config.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns, ddof=1)
        return np.mean(excess_returns) / downside_deviation * math.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown if max_drawdown > 0 else 0.0
    
    async def stress_test_portfolio(self, 
                                  positions: Dict[str, float],
                                  stress_scenarios: List[Dict[str, float]]) -> Dict[str, float]:
        """Run stress tests on portfolio."""
        try:
            stress_results = {}
            
            for i, scenario in enumerate(stress_scenarios):
                scenario_pnl = 0.0
                
                for symbol, position_value in positions.items():
                    if symbol in scenario:
                        shock = scenario[symbol]
                        scenario_pnl += position_value * shock
                
                stress_results[f"scenario_{i+1}"] = scenario_pnl
            
            return stress_results
            
        except Exception as e:
            logger.error("Stress testing failed", error=str(e))
            raise
    
    async def calculate_liquidity_risk(self, 
                                     positions: Dict[str, float],
                                     daily_volumes: Dict[str, float],
                                     liquidation_horizon_days: int = 5) -> Dict[str, float]:
        """Calculate liquidity risk metrics."""
        try:
            liquidity_metrics = {}
            
            for symbol, position_value in positions.items():
                daily_volume = daily_volumes.get(symbol, 0)
                
                if daily_volume > 0:
                    # Participation rate (what % of daily volume is our position)
                    participation_rate = abs(position_value) / daily_volume
                    
                    # Liquidity cost (simplified model)
                    liquidity_cost = participation_rate * 0.01  # 1% cost per 100% participation
                    
                    # Time to liquidate
                    time_to_liquidate = abs(position_value) / (daily_volume * 0.1)  # Assume 10% max participation
                    
                    liquidity_metrics[symbol] = {
                        'participation_rate': participation_rate,
                        'liquidity_cost': liquidity_cost,
                        'time_to_liquidate_days': time_to_liquidate,
                        'liquidity_risk_score': min(participation_rate * 10, 1.0)  # 0-1 scale
                    }
                else:
                    liquidity_metrics[symbol] = {
                        'participation_rate': 1.0,
                        'liquidity_cost': 0.1,  # High cost for illiquid assets
                        'time_to_liquidate_days': float('inf'),
                        'liquidity_risk_score': 1.0  # Maximum risk
                    }
            
            return liquidity_metrics
            
        except Exception as e:
            logger.error("Liquidity risk calculation failed", error=str(e))
            raise


# Factory function
def create_dynamic_risk_engine(config: RiskConfiguration = None) -> DynamicRiskEngine:
    """Create a dynamic risk engine instance."""
    return DynamicRiskEngine(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_risk_engine():
        """Test the dynamic risk engine."""
        print("Testing Dynamic Risk Engine...")
        
        # Create engine
        config = RiskConfiguration(
            monte_carlo_simulations=5000,
            lookback_days=100
        )
        engine = create_dynamic_risk_engine(config)
        
        # Generate sample returns data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        position_value = 100000  # $100k position
        
        # Test VaR calculation
        risk_metrics = await engine.calculate_var(
            returns=returns,
            position_value=position_value,
            asset_class=AssetClass.CRYPTOCURRENCY,
            model=RiskModel.HYBRID
        )
        
        print(f"VaR 95%: ${risk_metrics.var_95:,.2f}")
        print(f"VaR 99%: ${risk_metrics.var_99:,.2f}")
        print(f"Expected Shortfall 95%: ${risk_metrics.expected_shortfall_95:,.2f}")
        print(f"Volatility: {risk_metrics.volatility:.2%}")
        print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
        
        # Test portfolio risk
        positions = {'BTC': 50000, 'ETH': 30000, 'ADA': 20000}
        symbols = ['BTC', 'ETH', 'ADA']
        returns_matrix = np.random.multivariate_normal(
            [0.001, 0.0008, 0.0005], 
            [[0.0004, 0.0002, 0.0001],
             [0.0002, 0.0003, 0.00008],
             [0.0001, 0.00008, 0.0002]], 
            252
        )
        
        portfolio_risk = await engine.calculate_portfolio_risk(
            positions=positions,
            returns_matrix=returns_matrix,
            symbols=symbols
        )
        
        print(f"\nPortfolio VaR 95%: ${portfolio_risk.total_var_95:,.2f}")
        print(f"Diversification Ratio: {portfolio_risk.diversification_ratio:.2f}")
        print(f"Concentration Risk: {portfolio_risk.concentration_risk:.3f}")
        
        print("Risk Engine test completed successfully!")
    
    # Run test
    import asyncio
    asyncio.run(test_risk_engine())