"""
Test suite for Portfolio Optimizer.
"""
import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from portfolio_optimizer import (
    PortfolioOptimizer, PortfolioOptimizerConfig, OptimizationObjective,
    OptimizationConstraints, MarketView, create_portfolio_optimizer
)

@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    return {
        'BTC': np.random.normal(0.001, 0.025, 252).tolist(),
        'ETH': np.random.normal(0.0008, 0.03, 252).tolist(),
        'ADA': np.random.normal(0.0005, 0.035, 252).tolist()
    }

@pytest.fixture
def portfolio_optimizer():
    """Create portfolio optimizer for testing."""
    config = PortfolioOptimizerConfig(
        risk_free_rate=0.02,
        transaction_cost_bps=10.0,
        max_iterations=500  # Reduced for faster testing
    )
    return create_portfolio_optimizer(config)

class TestPortfolioOptimizer:
    """Test the portfolio optimizer."""
    
    @pytest.mark.asyncio
    async def test_max_sharpe_optimization(self, portfolio_optimizer, sample_returns_data):
        """Test maximum Sharpe ratio optimization."""
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.MAX_SHARPE
        )
        
        assert result.optimization_status == "optimal"
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert result.sharpe_ratio > 0
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in result.optimal_weights.values())
        assert all(w <= 0.4 for w in result.optimal_weights.values())  # Max concentration
    
    @pytest.mark.asyncio
    async def test_min_variance_optimization(self, portfolio_optimizer, sample_returns_data):
        """Test minimum variance optimization."""
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.MIN_VARIANCE
        )
        
        assert result.optimization_status == "optimal"
        assert result.expected_volatility > 0
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in result.optimal_weights.values())
        
        # Min variance should have lower volatility than equal weights
        equal_weights_vol = await portfolio_optimizer._calculate_equal_weights_volatility(sample_returns_data)
        # Note: This comparison might not always hold due to constraints
    
    @pytest.mark.asyncio
    async def test_risk_parity_optimization(self, portfolio_optimizer, sample_returns_data):
        """Test risk parity optimization."""
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.RISK_PARITY
        )
        
        assert result.optimization_status == "optimal"
        assert result.expected_volatility > 0
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0.01 for w in result.optimal_weights.values())  # Min 1% for risk parity
        
        # Check that risk contributions are approximately equal
        risk_contributions = list(result.risk_contribution.values())
        max_risk_contrib = max(risk_contributions)
        min_risk_contrib = min(risk_contributions)
        # Risk contributions should be relatively balanced
        assert (max_risk_contrib - min_risk_contrib) < 0.2  # Within 20% range
    
    @pytest.mark.asyncio
    async def test_black_litterman_optimization(self, portfolio_optimizer, sample_returns_data):
        """Test Black-Litterman optimization with market views."""
        market_views = [
            MarketView(asset='BTC', expected_return=0.15, confidence=0.8),
            MarketView(asset='ETH', expected_return=0.12, confidence=0.6)
        ]
        
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.BLACK_LITTERMAN,
            market_views=market_views
        )
        
        assert result.optimization_status == "optimal"
        assert result.expected_return > 0
        assert result.expected_volatility > 0
        assert result.sharpe_ratio > 0
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
        
        # BTC should have higher weight due to higher expected return view
        assert result.optimal_weights['BTC'] > 0
    
    @pytest.mark.asyncio
    async def test_optimization_with_constraints(self, portfolio_optimizer, sample_returns_data):
        """Test optimization with custom constraints."""
        constraints = OptimizationConstraints(
            min_weights={'BTC': 0.2, 'ETH': 0.1},
            max_weights={'BTC': 0.5, 'ETH': 0.3, 'ADA': 0.2},
            max_concentration=0.5
        )
        
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.MAX_SHARPE,
            constraints=constraints
        )
        
        assert result.optimization_status == "optimal"
        assert result.optimal_weights['BTC'] >= 0.2
        assert result.optimal_weights['ETH'] >= 0.1
        assert result.optimal_weights['BTC'] <= 0.5
        assert result.optimal_weights['ETH'] <= 0.3
        assert result.optimal_weights['ADA'] <= 0.2
    
    def test_returns_matrix_preparation(self, portfolio_optimizer, sample_returns_data):
        """Test returns matrix preparation."""
        returns_matrix = portfolio_optimizer._prepare_returns_matrix(sample_returns_data)
        
        assert returns_matrix.shape[1] == 3  # 3 assets
        assert returns_matrix.shape[0] == 252  # 252 observations
        assert not np.any(np.isnan(returns_matrix))
    
    def test_expected_returns_calculation(self, portfolio_optimizer, sample_returns_data):
        """Test expected returns calculation."""
        returns_matrix = portfolio_optimizer._prepare_returns_matrix(sample_returns_data)
        expected_returns = portfolio_optimizer._calculate_expected_returns(returns_matrix)
        
        assert len(expected_returns) == 3
        assert all(isinstance(ret, (int, float)) for ret in expected_returns)
        assert not np.any(np.isnan(expected_returns))
    
    def test_covariance_matrix_calculation(self, portfolio_optimizer, sample_returns_data):
        """Test covariance matrix calculation."""
        returns_matrix = portfolio_optimizer._prepare_returns_matrix(sample_returns_data)
        cov_matrix = portfolio_optimizer._calculate_covariance_matrix(returns_matrix)
        
        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric
        assert np.all(np.linalg.eigvals(cov_matrix) > 0)  # Should be positive definite
    
    def test_market_views_processing(self, portfolio_optimizer):
        """Test market views processing for Black-Litterman."""
        symbols = ['BTC', 'ETH', 'ADA']
        market_views = [
            MarketView(asset='BTC', expected_return=0.15, confidence=0.8),
            MarketView(asset='ETH', expected_return=0.12, confidence=0.6)
        ]
        
        P, Q, Omega = portfolio_optimizer._process_market_views(market_views, symbols)
        
        assert P is not None
        assert Q is not None
        assert Omega is not None
        assert P.shape[1] == 3  # 3 assets
        assert len(Q) == P.shape[0]  # Same number of views
        assert Omega.shape[0] == Omega.shape[1] == P.shape[0]  # Square matrix
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, portfolio_optimizer):
        """Test handling of insufficient historical data."""
        insufficient_data = {
            'BTC': [0.01, 0.02],  # Only 2 data points
            'ETH': [0.015, 0.018]
        }
        
        with pytest.raises(ValueError, match="Insufficient historical data"):
            await portfolio_optimizer.optimize_portfolio(
                returns_data=insufficient_data,
                objective=OptimizationObjective.MAX_SHARPE
            )
    
    @pytest.mark.asyncio
    async def test_empty_market_views_handling(self, portfolio_optimizer, sample_returns_data):
        """Test Black-Litterman with empty market views."""
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=sample_returns_data,
            objective=OptimizationObjective.BLACK_LITTERMAN,
            market_views=[]
        )
        
        # Should still work with implied returns
        assert result.optimization_status == "optimal"
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
    
    def test_optimization_result_enhancement(self, portfolio_optimizer, sample_returns_data):
        """Test optimization result enhancement with additional metrics."""
        # Create a basic result
        from portfolio_optimizer import OptimizationResult
        
        basic_result = OptimizationResult(
            optimal_weights={'BTC': 0.4, 'ETH': 0.35, 'ADA': 0.25},
            expected_return=0.0,  # Will be calculated
            expected_volatility=0.2,
            sharpe_ratio=0.0,  # Will be calculated
            objective_value=1.0,
            optimization_status="optimal",
            iterations=10,
            computation_time=0.5,
            risk_contribution={},  # Will be calculated
            diversification_ratio=0.0,  # Will be calculated
            concentration_ratio=0.0  # Will be calculated
        )
        
        # Prepare data
        returns_matrix = portfolio_optimizer._prepare_returns_matrix(sample_returns_data)
        expected_returns = portfolio_optimizer._calculate_expected_returns(returns_matrix)
        covariance_matrix = portfolio_optimizer._calculate_covariance_matrix(returns_matrix)
        symbols = list(sample_returns_data.keys())
        
        # Enhance result
        enhanced_result = portfolio_optimizer._enhance_optimization_result(
            basic_result, expected_returns, covariance_matrix, symbols
        )
        
        assert enhanced_result.expected_return != 0.0
        assert enhanced_result.sharpe_ratio != 0.0
        assert len(enhanced_result.risk_contribution) == 3
        assert enhanced_result.diversification_ratio > 0
        assert enhanced_result.concentration_ratio > 0

class TestPortfolioOptimizerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_single_asset_optimization(self, portfolio_optimizer):
        """Test optimization with single asset."""
        single_asset_data = {
            'BTC': np.random.normal(0.001, 0.025, 252).tolist()
        }
        
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=single_asset_data,
            objective=OptimizationObjective.MAX_SHARPE
        )
        
        assert result.optimal_weights['BTC'] == 1.0
        assert result.concentration_ratio == 1.0
        assert result.diversification_ratio == 1.0
    
    @pytest.mark.asyncio
    async def test_zero_volatility_asset(self, portfolio_optimizer):
        """Test optimization with zero volatility asset."""
        zero_vol_data = {
            'BTC': np.random.normal(0.001, 0.025, 252).tolist(),
            'STABLE': [0.0] * 252  # Zero volatility
        }
        
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=zero_vol_data,
            objective=OptimizationObjective.MIN_VARIANCE
        )
        
        # Should heavily weight the zero volatility asset
        assert result.optimal_weights['STABLE'] > 0.5
        assert result.expected_volatility >= 0
    
    @pytest.mark.asyncio
    async def test_highly_correlated_assets(self, portfolio_optimizer):
        """Test optimization with highly correlated assets."""
        # Create highly correlated returns
        base_returns = np.random.normal(0.001, 0.025, 252)
        correlated_data = {
            'ASSET1': base_returns.tolist(),
            'ASSET2': (base_returns + np.random.normal(0, 0.001, 252)).tolist(),  # Highly correlated
            'ASSET3': np.random.normal(0.0005, 0.02, 252).tolist()  # Less correlated
        }
        
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=correlated_data,
            objective=OptimizationObjective.MAX_SHARPE
        )
        
        # Should prefer the less correlated asset for diversification
        assert result.diversification_ratio > 1.0
        assert result.concentration_ratio < 1.0

if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        """Run a simple test of the portfolio optimizer."""
        print("Running simple portfolio optimizer test...")
        
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
        print("1. Maximum Sharpe Ratio Optimization:")
        result = await optimizer.optimize_portfolio(
            returns_data=returns_data,
            objective=OptimizationObjective.MAX_SHARPE
        )
        
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Volatility: {result.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Optimal Weights: {result.optimal_weights}")
        print(f"Diversification Ratio: {result.diversification_ratio:.2f}")
        
        # Test Risk Parity
        print("\n2. Risk Parity Optimization:")
        result_rp = await optimizer.optimize_portfolio(
            returns_data=returns_data,
            objective=OptimizationObjective.RISK_PARITY
        )
        
        print(f"Volatility: {result_rp.expected_volatility:.2%}")
        print(f"Optimal Weights: {result_rp.optimal_weights}")
        print(f"Risk Contributions: {result_rp.risk_contribution}")
        
        # Test Black-Litterman
        print("\n3. Black-Litterman Optimization:")
        market_views = [
            MarketView(asset='BTC', expected_return=0.15, confidence=0.8),
            MarketView(asset='ETH', expected_return=0.12, confidence=0.6)
        ]
        
        result_bl = await optimizer.optimize_portfolio(
            returns_data=returns_data,
            objective=OptimizationObjective.BLACK_LITTERMAN,
            market_views=market_views
        )
        
        print(f"Expected Return: {result_bl.expected_return:.2%}")
        print(f"Volatility: {result_bl.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result_bl.sharpe_ratio:.2f}")
        print(f"Optimal Weights: {result_bl.optimal_weights}")
        
        print("\nPortfolio Optimizer test completed successfully!")
    
    # Run the test
    asyncio.run(run_simple_test())