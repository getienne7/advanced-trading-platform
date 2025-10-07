"""
Portfolio Optimization Engine for Advanced Trading Platform.
Implements Modern Portfolio Theory, Risk Parity, and Black-Litterman models.
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
import warnings
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import inv, pinv
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("portfolio-optimizer")

class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"

@dataclass
class MarketView:
    """Market view for Black-Litterman model."""
    asset: str
    expected_return: float
    confidence: float  # 0.0 to 1.0
    view_type: str = "absolute"  # absolute or relative
    reference_asset: Optional[str] = None  # for relative views

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weights: Dict[str, float] = field(default_factory=dict)
    max_weights: Dict[str, float] = field(default_factory=dict)
    max_concentration: float = 0.4  # Maximum single asset weight

@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    objective_value: float
    optimization_status: str
    iterations: int
    computation_time: float
    risk_contribution: Dict[str, float]
    diversification_ratio: float
    concentration_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PortfolioOptimizerConfig(BaseModel):
    """Configuration for portfolio optimizer."""
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate")
    transaction_cost_bps: float = Field(default=10.0, description="Transaction costs in basis points")
    max_iterations: int = Field(default=1000, description="Maximum optimization iterations")
    tolerance: float = Field(default=1e-8, description="Optimization tolerance")
    regularization: float = Field(default=1e-5, description="Covariance matrix regularization")
    shrinkage_intensity: float = Field(default=0.1, description="Ledoit-Wolf shrinkage intensity")
    black_litterman_tau: float = Field(default=0.025, description="Black-Litterman scaling factor")
    risk_aversion: float = Field(default=3.0, description="Risk aversion parameter")

class PortfolioOptimizer:
    """Advanced portfolio optimization engine."""
    
    def __init__(self, config: PortfolioOptimizerConfig = None):
        self.config = config or PortfolioOptimizerConfig()
        
        logger.info("Portfolio Optimizer initialized", config=self.config.dict())
    
    async def optimize_portfolio(self,
                               returns_data: Dict[str, List[float]],
                               current_weights: Optional[Dict[str, float]] = None,
                               objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                               constraints: OptimizationConstraints = None,
                               market_views: List[MarketView] = None) -> OptimizationResult:
        """Optimize portfolio using specified objective and constraints."""
        try:
            start_time = datetime.utcnow()
            
            # Prepare data
            symbols = list(returns_data.keys())
            returns_matrix = self._prepare_returns_matrix(returns_data)
            
            if returns_matrix.shape[0] < 30:
                raise ValueError("Insufficient historical data for optimization")
            
            # Calculate expected returns and covariance matrix
            expected_returns = self._calculate_expected_returns(returns_matrix)
            covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
            
            # Apply constraints
            constraints = constraints or OptimizationConstraints()
            
            # Optimize based on objective
            if objective == OptimizationObjective.MAX_SHARPE:
                result = await self._maximize_sharpe_ratio(
                    expected_returns, covariance_matrix, symbols, constraints
                )
            elif objective == OptimizationObjective.MIN_VARIANCE:
                result = await self._minimize_variance(
                    covariance_matrix, symbols, constraints
                )
            elif objective == OptimizationObjective.RISK_PARITY:
                result = await self._risk_parity_optimization(
                    covariance_matrix, symbols, constraints
                )
            elif objective == OptimizationObjective.BLACK_LITTERMAN:
                if not market_views:
                    raise ValueError("Market views required for Black-Litterman optimization")
                result = await self._black_litterman_optimization(
                    expected_returns, covariance_matrix, symbols, market_views, constraints
                )
            else:
                raise ValueError(f"Unsupported optimization objective: {objective}")
            
            # Calculate additional metrics
            result = self._enhance_optimization_result(
                result, expected_returns, covariance_matrix, symbols, current_weights
            )
            
            # Calculate computation time
            computation_time = (datetime.utcnow() - start_time).total_seconds()
            result.computation_time = computation_time
            
            logger.info("Portfolio optimization completed",
                       objective=objective.value,
                       expected_return=result.expected_return,
                       volatility=result.expected_volatility,
                       sharpe_ratio=result.sharpe_ratio)
            
            return result
            
        except Exception as e:
            logger.error("Portfolio optimization failed", error=str(e))
            raise
    
    async def _maximize_sharpe_ratio(self,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   symbols: List[str],
                                   constraints: OptimizationConstraints) -> OptimizationResult:
        """Maximize Sharpe ratio using quadratic programming."""
        try:
            n_assets = len(symbols)
            
            def negative_sharpe_ratio(weights):
                """Negative Sharpe ratio for minimization."""
                weights = np.array(weights)
                portfolio_return = np.dot(expected_returns, weights)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                excess_return = portfolio_return * 252 - self.config.risk_free_rate
                sharpe_ratio = excess_return / portfolio_volatility
                return -sharpe_ratio  # Negative for minimization
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = []
            
            # Weights sum to 1
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Bounds for individual weights
            bounds = []
            for i, symbol in enumerate(symbols):
                min_weight = constraints.min_weights.get(symbol, 0.0)
                max_weight = constraints.max_weights.get(symbol, constraints.max_concentration)
                bounds.append((min_weight, max_weight))
            
            # Optimize
            result = minimize(
                negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)  # Normalize
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(expected_returns, optimal_weights)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)
                sharpe_ratio = (portfolio_return * 252 - self.config.risk_free_rate) / portfolio_volatility
                
                return OptimizationResult(
                    optimal_weights={symbols[i]: float(optimal_weights[i]) for i in range(n_assets)},
                    expected_return=portfolio_return * 252,
                    expected_volatility=portfolio_volatility,
                    sharpe_ratio=sharpe_ratio,
                    objective_value=-result.fun,  # Convert back to positive
                    optimization_status="optimal" if result.success else "failed",
                    iterations=result.nit,
                    computation_time=0.0,
                    risk_contribution={},
                    diversification_ratio=0.0,
                    concentration_ratio=0.0
                )
            
            raise ValueError(f"Sharpe ratio optimization failed: {result.message}")
            
        except Exception as e:
            logger.error("Sharpe ratio optimization failed", error=str(e))
            raise
    
    async def _minimize_variance(self,
                               covariance_matrix: np.ndarray,
                               symbols: List[str],
                               constraints: OptimizationConstraints) -> OptimizationResult:
        """Minimize portfolio variance."""
        try:
            n_assets = len(symbols)
            
            def portfolio_variance(weights):
                """Portfolio variance objective function."""
                weights = np.array(weights)
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = []
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Bounds
            bounds = []
            for i, symbol in enumerate(symbols):
                min_weight = constraints.min_weights.get(symbol, 0.0)
                max_weight = constraints.max_weights.get(symbol, constraints.max_concentration)
                bounds.append((min_weight, max_weight))
            
            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)
                
                return OptimizationResult(
                    optimal_weights={symbols[i]: float(optimal_weights[i]) for i in range(n_assets)},
                    expected_return=0.0,  # Will be calculated later
                    expected_volatility=portfolio_volatility,
                    sharpe_ratio=0.0,
                    objective_value=result.fun,
                    optimization_status="optimal" if result.success else "failed",
                    iterations=result.nit,
                    computation_time=0.0,
                    risk_contribution={},
                    diversification_ratio=0.0,
                    concentration_ratio=0.0
                )
            
            raise ValueError(f"Variance minimization failed: {result.message}")
            
        except Exception as e:
            logger.error("Variance minimization failed", error=str(e))
            raise
    
    async def _risk_parity_optimization(self,
                                      covariance_matrix: np.ndarray,
                                      symbols: List[str],
                                      constraints: OptimizationConstraints) -> OptimizationResult:
        """Risk parity portfolio optimization."""
        try:
            n_assets = len(symbols)
            
            def risk_parity_objective(weights):
                """Risk parity objective function."""
                weights = np.array(weights)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                
                # Calculate risk contributions
                marginal_risk = np.dot(covariance_matrix, weights)
                risk_contributions = weights * marginal_risk / portfolio_variance
                
                # Target equal risk contributions
                target_risk = np.ones(n_assets) / n_assets
                
                # Minimize sum of squared deviations
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = []
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Bounds
            bounds = []
            for i, symbol in enumerate(symbols):
                min_weight = constraints.min_weights.get(symbol, 0.01)  # Minimum 1%
                max_weight = constraints.max_weights.get(symbol, constraints.max_concentration)
                bounds.append((min_weight, max_weight))
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)
                
                # Calculate portfolio metrics
                portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)
                
                # Calculate risk contributions
                marginal_risk = np.dot(covariance_matrix, optimal_weights)
                risk_contributions = optimal_weights * marginal_risk / portfolio_variance
                
                return OptimizationResult(
                    optimal_weights={symbols[i]: float(optimal_weights[i]) for i in range(n_assets)},
                    expected_return=0.0,
                    expected_volatility=portfolio_volatility,
                    sharpe_ratio=0.0,
                    objective_value=result.fun,
                    optimization_status="optimal" if result.success else "failed",
                    iterations=result.nit,
                    computation_time=0.0,
                    risk_contribution={symbols[i]: float(risk_contributions[i]) for i in range(n_assets)},
                    diversification_ratio=0.0,
                    concentration_ratio=0.0
                )
            
            raise ValueError(f"Risk parity optimization failed: {result.message}")
            
        except Exception as e:
            logger.error("Risk parity optimization failed", error=str(e))
            raise
    
    async def _black_litterman_optimization(self,
                                          expected_returns: np.ndarray,
                                          covariance_matrix: np.ndarray,
                                          symbols: List[str],
                                          market_views: List[MarketView],
                                          constraints: OptimizationConstraints) -> OptimizationResult:
        """Black-Litterman model optimization."""
        try:
            n_assets = len(symbols)
            
            # Market weights (simplified - equal weights)
            market_weights = np.ones(n_assets) / n_assets
            
            # Implied equilibrium returns
            risk_aversion = self.config.risk_aversion
            implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
            
            # Process market views
            P, Q, Omega = self._process_market_views(market_views, symbols)
            
            if P is None:
                # No valid views, use implied returns
                bl_returns = implied_returns
            else:
                # Black-Litterman formula
                tau = self.config.black_litterman_tau
                
                try:
                    M1 = inv(tau * covariance_matrix)
                    M2 = np.dot(P.T, np.dot(inv(Omega), P))
                    M3 = np.dot(inv(tau * covariance_matrix), implied_returns)
                    M4 = np.dot(P.T, np.dot(inv(Omega), Q))
                    
                    bl_returns = np.dot(inv(M1 + M2), M3 + M4)
                except np.linalg.LinAlgError:
                    # Fallback to implied returns if matrix inversion fails
                    bl_returns = implied_returns
            
            # Optimize using Black-Litterman returns
            def negative_utility(weights):
                """Negative utility for minimization."""
                weights = np.array(weights)
                portfolio_return = np.dot(bl_returns, weights)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                return -utility
            
            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets
            
            # Constraints
            constraints_list = []
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Bounds
            bounds = []
            for i, symbol in enumerate(symbols):
                min_weight = constraints.min_weights.get(symbol, 0.0)
                max_weight = constraints.max_weights.get(symbol, constraints.max_concentration)
                bounds.append((min_weight, max_weight))
            
            # Optimize
            result = minimize(
                negative_utility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)
                
                # Calculate metrics using original expected returns
                portfolio_return = np.dot(expected_returns, optimal_weights)
                portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)
                sharpe_ratio = (portfolio_return * 252 - self.config.risk_free_rate) / portfolio_volatility
                
                return OptimizationResult(
                    optimal_weights={symbols[i]: float(optimal_weights[i]) for i in range(n_assets)},
                    expected_return=portfolio_return * 252,
                    expected_volatility=portfolio_volatility,
                    sharpe_ratio=sharpe_ratio,
                    objective_value=-result.fun,
                    optimization_status="optimal" if result.success else "failed",
                    iterations=result.nit,
                    computation_time=0.0,
                    risk_contribution={},
                    diversification_ratio=0.0,
                    concentration_ratio=0.0
                )
            
            raise ValueError(f"Black-Litterman optimization failed: {result.message}")
            
        except Exception as e:
            logger.error("Black-Litterman optimization failed", error=str(e))
            raise
    
    def _process_market_views(self, market_views: List[MarketView], symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process market views into Black-Litterman matrices."""
        try:
            if not market_views:
                return None, None, None
            
            n_assets = len(symbols)
            n_views = len(market_views)
            
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            Omega = np.zeros((n_views, n_views))
            
            symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}
            
            for i, view in enumerate(market_views):
                if view.asset in symbol_to_index:
                    asset_index = symbol_to_index[view.asset]
                    P[i, asset_index] = 1.0
                    Q[i] = view.expected_return / 252  # Convert to daily
                    
                    # View uncertainty
                    view_variance = (1.0 - view.confidence) * 0.01
                    Omega[i, i] = view_variance
            
            # Remove empty rows
            valid_views = np.any(P != 0, axis=1)
            P = P[valid_views]
            Q = Q[valid_views]
            Omega = Omega[np.ix_(valid_views, valid_views)]
            
            return P, Q, Omega
            
        except Exception as e:
            logger.error("Failed to process market views", error=str(e))
            return None, None, None
    
    def _prepare_returns_matrix(self, returns_data: Dict[str, List[float]]) -> np.ndarray:
        """Prepare returns matrix from dictionary data."""
        symbols = list(returns_data.keys())
        min_length = min(len(returns) for returns in returns_data.values())
        
        returns_matrix = np.array([
            returns_data[symbol][-min_length:] for symbol in symbols
        ]).T
        
        return returns_matrix
    
    def _calculate_expected_returns(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate expected returns using historical mean."""
        expected_returns = np.mean(returns_matrix, axis=0)
        
        # Apply shrinkage
        grand_mean = np.mean(expected_returns)
        shrinkage = self.config.shrinkage_intensity
        expected_returns = (1 - shrinkage) * expected_returns + shrinkage * grand_mean
        
        return expected_returns
    
    def _calculate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix with regularization."""
        cov_matrix = np.cov(returns_matrix.T)
        
        # Regularization
        regularization = self.config.regularization
        identity = np.eye(cov_matrix.shape[0])
        cov_matrix = cov_matrix + regularization * identity
        
        return cov_matrix
    
    def _enhance_optimization_result(self,
                                   result: OptimizationResult,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   symbols: List[str],
                                   current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Enhance optimization result with additional metrics."""
        try:
            weights_array = np.array([result.optimal_weights[symbol] for symbol in symbols])
            
            # Calculate expected return if not set
            if result.expected_return == 0.0:
                result.expected_return = np.dot(expected_returns, weights_array) * 252
            
            # Calculate Sharpe ratio if not set
            if result.sharpe_ratio == 0.0:
                excess_return = result.expected_return - self.config.risk_free_rate
                result.sharpe_ratio = excess_return / result.expected_volatility if result.expected_volatility > 0 else 0.0
            
            # Calculate risk contributions if not set
            if not result.risk_contribution:
                portfolio_variance = np.dot(weights_array.T, np.dot(covariance_matrix, weights_array))
                if portfolio_variance > 0:
                    marginal_risk = np.dot(covariance_matrix, weights_array)
                    risk_contributions = weights_array * marginal_risk / portfolio_variance
                    result.risk_contribution = {symbols[i]: float(risk_contributions[i]) for i in range(len(symbols))}
            
            # Calculate diversification ratio
            individual_volatilities = np.sqrt(np.diag(covariance_matrix) * 252)
            weighted_avg_vol = np.dot(weights_array, individual_volatilities)
            result.diversification_ratio = weighted_avg_vol / result.expected_volatility if result.expected_volatility > 0 else 1.0
            
            # Calculate concentration ratio
            result.concentration_ratio = np.sum(weights_array ** 2)
            
            return result
            
        except Exception as e:
            logger.error("Failed to enhance optimization result", error=str(e))
            return result


# Factory function
def create_portfolio_optimizer(config: PortfolioOptimizerConfig = None) -> PortfolioOptimizer:
    """Create a portfolio optimizer instance."""
    return PortfolioOptimizer(config)


# Example usage
if __name__ == "__main__":
    async def test_portfolio_optimizer():
        """Test the portfolio optimizer."""
        print("Testing Portfolio Optimizer...")
        
        # Create optimizer
        optimizer = create_portfolio_optimizer()
        
        # Generate sample data
        np.random.seed(42)
        returns_data = {
            'BTC': np.random.normal(0.001, 0.025, 252).tolist(),
            'ETH': np.random.normal(0.0008, 0.03, 252).tolist(),
            'ADA': np.random.normal(0.0005, 0.035, 252).tolist()
        }
        
        # Test Maximum Sharpe Ratio
        result = await optimizer.optimize_portfolio(
            returns_data=returns_data,
            objective=OptimizationObjective.MAX_SHARPE
        )
        
        print(f"Max Sharpe - Return: {result.expected_return:.2%}, Vol: {result.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Weights: {result.optimal_weights}")
        
        # Test Risk Parity
        result_rp = await optimizer.optimize_portfolio(
            returns_data=returns_data,
            objective=OptimizationObjective.RISK_PARITY
        )
        
        print(f"\nRisk Parity - Vol: {result_rp.expected_volatility:.2%}")
        print(f"Weights: {result_rp.optimal_weights}")
        print(f"Risk Contributions: {result_rp.risk_contribution}")
        
        print("\nPortfolio Optimizer test completed!")
    
    # Run test
    asyncio.run(test_portfolio_optimizer())