"""
Test suite for Risk Management Service.
"""
import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from dynamic_risk_engine import (
    DynamicRiskEngine, RiskConfiguration, RiskModel, AssetClass,
    create_dynamic_risk_engine
)
from main import app

# Test client
client = TestClient(app)

@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252).tolist()  # Daily returns for 1 year

@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing."""
    return {
        'positions': {'BTC': 50000, 'ETH': 30000, 'ADA': 20000},
        'returns_data': {
            'BTC': np.random.normal(0.001, 0.025, 252).tolist(),
            'ETH': np.random.normal(0.0008, 0.03, 252).tolist(),
            'ADA': np.random.normal(0.0005, 0.035, 252).tolist()
        }
    }

@pytest.fixture
def risk_engine():
    """Create risk engine for testing."""
    config = RiskConfiguration(
        monte_carlo_simulations=1000,  # Reduced for faster testing
        lookback_days=100
    )
    return create_dynamic_risk_engine(config)

class TestDynamicRiskEngine:
    """Test the dynamic risk engine."""
    
    @pytest.mark.asyncio
    async def test_monte_carlo_var(self, risk_engine):
        """Test Monte Carlo VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value,
            asset_class=AssetClass.CRYPTOCURRENCY,
            model=RiskModel.MONTE_CARLO
        )
        
        assert risk_metrics.var_95 > 0
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert risk_metrics.expected_shortfall_95 > risk_metrics.var_95
        assert risk_metrics.expected_shortfall_99 > risk_metrics.var_99
        assert risk_metrics.volatility > 0
        assert -3 < risk_metrics.skewness < 3  # Reasonable skewness range
        assert risk_metrics.kurtosis > -2  # Reasonable kurtosis
    
    @pytest.mark.asyncio
    async def test_historical_simulation_var(self, risk_engine):
        """Test Historical Simulation VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value,
            model=RiskModel.HISTORICAL_SIMULATION
        )
        
        assert risk_metrics.var_95 > 0
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert risk_metrics.volatility > 0
        assert risk_metrics.sharpe_ratio is not None
    
    @pytest.mark.asyncio
    async def test_parametric_var(self, risk_engine):
        """Test Parametric VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value,
            asset_class=AssetClass.CRYPTOCURRENCY,
            model=RiskModel.PARAMETRIC
        )
        
        assert risk_metrics.var_95 > 0
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert risk_metrics.volatility > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_var(self, risk_engine):
        """Test Hybrid VaR calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value,
            asset_class=AssetClass.CRYPTOCURRENCY,
            model=RiskModel.HYBRID
        )
        
        assert risk_metrics.var_95 > 0
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert risk_metrics.volatility > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation(self, risk_engine):
        """Test portfolio risk calculation."""
        positions = {'BTC': 50000, 'ETH': 30000, 'ADA': 20000}
        symbols = ['BTC', 'ETH', 'ADA']
        
        # Generate correlated returns
        np.random.seed(42)
        returns_matrix = np.random.multivariate_normal(
            [0.001, 0.0008, 0.0005], 
            [[0.0004, 0.0002, 0.0001],
             [0.0002, 0.0003, 0.00008],
             [0.0001, 0.00008, 0.0002]], 
            252
        )
        
        portfolio_risk = await risk_engine.calculate_portfolio_risk(
            positions=positions,
            returns_matrix=returns_matrix,
            symbols=symbols
        )
        
        assert portfolio_risk.total_var_95 > 0
        assert portfolio_risk.total_var_99 > portfolio_risk.total_var_95
        assert 0 < portfolio_risk.diversification_ratio <= 2
        assert 0 < portfolio_risk.concentration_risk <= 1
        assert portfolio_risk.correlation_matrix.shape == (3, 3)
        assert len(portfolio_risk.component_var) == 3
        assert len(portfolio_risk.marginal_var) == 3
        assert len(portfolio_risk.risk_budget) == 3
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, risk_engine):
        """Test stress testing functionality."""
        positions = {'BTC': 50000, 'ETH': 30000, 'ADA': 20000}
        scenarios = [
            {'BTC': -0.3, 'ETH': -0.35, 'ADA': -0.4},  # Market crash
            {'BTC': -0.1, 'ETH': -0.15, 'ADA': -0.2},  # Mild correction
            {'BTC': 0.2, 'ETH': 0.25, 'ADA': 0.3}      # Bull market
        ]
        
        stress_results = await risk_engine.stress_test_portfolio(
            positions=positions,
            stress_scenarios=scenarios
        )
        
        assert len(stress_results) == 3
        assert all(isinstance(pnl, (int, float)) for pnl in stress_results.values())
        
        # Check that crash scenario produces negative P&L
        crash_pnl = stress_results['scenario_1']
        assert crash_pnl < 0
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_calculation(self, risk_engine):
        """Test liquidity risk calculation."""
        positions = {'BTC': 50000, 'ETH': 30000, 'ADA': 20000}
        daily_volumes = {'BTC': 1000000, 'ETH': 500000, 'ADA': 100000}
        
        liquidity_metrics = await risk_engine.calculate_liquidity_risk(
            positions=positions,
            daily_volumes=daily_volumes,
            liquidation_horizon_days=5
        )
        
        assert len(liquidity_metrics) == 3
        for symbol, metrics in liquidity_metrics.items():
            assert 'participation_rate' in metrics
            assert 'liquidity_cost' in metrics
            assert 'time_to_liquidate_days' in metrics
            assert 'liquidity_risk_score' in metrics
            assert 0 <= metrics['liquidity_risk_score'] <= 1
    
    def test_clean_returns(self, risk_engine):
        """Test returns data cleaning."""
        # Create returns with NaN and outliers
        returns = np.array([0.01, 0.02, np.nan, 0.015, 10.0, -8.0, 0.005])
        
        cleaned_returns = risk_engine._clean_returns(returns)
        
        assert not np.any(np.isnan(cleaned_returns))
        assert not np.any(np.isinf(cleaned_returns))
        assert len(cleaned_returns) <= len(returns)
    
    def test_max_drawdown_calculation(self, risk_engine):
        """Test maximum drawdown calculation."""
        # Create returns that result in known drawdown
        returns = np.array([0.1, -0.2, -0.1, 0.05, -0.15, 0.2])
        
        max_drawdown = risk_engine._calculate_max_drawdown(returns)
        
        assert max_drawdown >= 0
        assert max_drawdown <= 1
    
    def test_sharpe_ratio_calculation(self, risk_engine):
        """Test Sharpe ratio calculation."""
        returns = np.random.normal(0.001, 0.02, 252)
        
        sharpe_ratio = risk_engine._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe_ratio, (int, float))
        assert not np.isnan(sharpe_ratio)

class TestRiskManagementAPI:
    """Test the Risk Management API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'risk-management'
    
    def test_var_calculation_endpoint(self, sample_returns):
        """Test VaR calculation endpoint."""
        request_data = {
            'returns': sample_returns,
            'position_value': 100000,
            'asset_class': 'cryptocurrency',
            'model': 'hybrid',
            'symbol': 'BTC'
        }
        
        response = client.post("/risk/var/calculate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'risk_metrics' in data
        assert 'var_95' in data['risk_metrics']
        assert 'var_99' in data['risk_metrics']
        assert data['risk_metrics']['var_95'] > 0
        assert data['risk_metrics']['var_99'] > data['risk_metrics']['var_95']
    
    def test_portfolio_risk_calculation_endpoint(self, sample_portfolio):
        """Test portfolio risk calculation endpoint."""
        response = client.post("/risk/portfolio/calculate", json=sample_portfolio)
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'portfolio_risk' in data
        assert 'total_var_95' in data['portfolio_risk']
        assert 'diversification_ratio' in data['portfolio_risk']
        assert data['portfolio_risk']['total_var_95'] > 0
    
    def test_stress_test_endpoint(self):
        """Test stress test endpoint."""
        request_data = {
            'positions': {'BTC': 50000, 'ETH': 30000},
            'scenarios': [
                {'BTC': -0.3, 'ETH': -0.35},
                {'BTC': 0.2, 'ETH': 0.25}
            ],
            'scenario_names': ['Market Crash', 'Bull Market']
        }
        
        response = client.post("/risk/stress-test", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'stress_test_results' in data
        assert 'Market Crash' in data['stress_test_results']
        assert 'Bull Market' in data['stress_test_results']
        assert 'summary' in data
    
    def test_liquidity_risk_endpoint(self):
        """Test liquidity risk assessment endpoint."""
        request_data = {
            'positions': {'BTC': 50000, 'ETH': 30000},
            'daily_volumes': {'BTC': 1000000, 'ETH': 500000},
            'liquidation_horizon_days': 5
        }
        
        response = client.post("/risk/liquidity/assess", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'liquidity_metrics' in data
        assert 'portfolio_summary' in data
        assert 'weighted_liquidity_score' in data['portfolio_summary']
    
    def test_predefined_scenarios_endpoint(self):
        """Test predefined scenarios endpoint."""
        response = client.get("/risk/scenarios/predefined")
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'scenarios' in data
        assert 'market_crash' in data['scenarios']
        assert 'crypto_winter' in data['scenarios']
    
    def test_risk_configuration_endpoints(self):
        """Test risk configuration endpoints."""
        # Test getting current config
        response = client.get("/risk/config")
        assert response.status_code == 200
        
        # Test updating config
        new_config = {
            'confidence_levels': [0.95, 0.99],
            'time_horizon_days': 1,
            'lookback_days': 252,
            'monte_carlo_simulations': 5000,
            'rebalancing_frequency': 'daily',
            'risk_free_rate': 0.02,
            'market_benchmark': 'BTC',
            'correlation_threshold': 0.7,
            'concentration_limit': 0.25
        }
        
        response = client.post("/risk/config/update", json=new_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
    
    def test_invalid_var_request(self):
        """Test VaR calculation with invalid data."""
        request_data = {
            'returns': [],  # Empty returns
            'position_value': 100000,
            'asset_class': 'cryptocurrency',
            'model': 'hybrid'
        }
        
        response = client.post("/risk/var/calculate", json=request_data)
        assert response.status_code == 500
    
    def test_invalid_portfolio_request(self):
        """Test portfolio risk calculation with mismatched data."""
        request_data = {
            'positions': {'BTC': 50000, 'ETH': 30000},
            'returns_data': {'BTC': [0.01, 0.02]}  # Missing ETH data
        }
        
        response = client.post("/risk/portfolio/calculate", json=request_data)
        assert response.status_code == 400

class TestRiskMetricsValidation:
    """Test risk metrics validation and edge cases."""
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self, risk_engine):
        """Test handling of insufficient data."""
        returns = np.array([0.01, 0.02])  # Only 2 data points
        position_value = 100000
        
        with pytest.raises(ValueError, match="Insufficient data"):
            await risk_engine.calculate_var(
                returns=returns,
                position_value=position_value
            )
    
    @pytest.mark.asyncio
    async def test_zero_volatility(self, risk_engine):
        """Test handling of zero volatility."""
        returns = np.zeros(100)  # All zeros
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value
        )
        
        # Should handle gracefully
        assert risk_metrics.var_95 >= 0
        assert risk_metrics.volatility >= 0
    
    @pytest.mark.asyncio
    async def test_extreme_returns(self, risk_engine):
        """Test handling of extreme returns."""
        returns = np.array([0.5, -0.6, 0.3, -0.4] * 25)  # Extreme daily returns
        position_value = 100000
        
        risk_metrics = await risk_engine.calculate_var(
            returns=returns,
            position_value=position_value
        )
        
        assert risk_metrics.var_95 > 0
        assert risk_metrics.var_99 > risk_metrics.var_95
        assert abs(risk_metrics.skewness) >= 0  # Should handle skewed data

if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        """Run a simple test of the risk engine."""
        print("Running simple risk management test...")
        
        # Create risk engine
        config = RiskConfiguration(monte_carlo_simulations=1000)
        engine = create_dynamic_risk_engine(config)
        
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        position_value = 100000
        
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
        
        print("Risk management test completed successfully!")
    
    # Run the test
    asyncio.run(run_simple_test())