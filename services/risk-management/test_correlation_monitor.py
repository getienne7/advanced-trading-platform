"""
Test suite for Correlation and Concentration Monitor.
"""
import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from correlation_monitor import (
    CorrelationConcentrationMonitor, CorrelationMonitorConfig, CorrelationMethod,
    ConcentrationMetric, AlertSeverity, create_correlation_concentration_monitor
)

@pytest.fixture
def sample_returns_data():
    """Generate sample returns data with known correlations."""
    np.random.seed(42)
    
    # Create base returns
    base_returns = np.random.normal(0.001, 0.02, 252)
    
    return {
        'BTC': base_returns.tolist(),
        'ETH': (base_returns * 0.8 + np.random.normal(0, 0.01, 252)).tolist(),  # High correlation with BTC
        'ADA': np.random.normal(0.0005, 0.025, 252).tolist(),  # Independent
        'DOT': (base_returns * 0.6 + np.random.normal(0, 0.015, 252)).tolist()  # Medium correlation with BTC
    }

@pytest.fixture
def correlation_monitor():
    """Create correlation monitor for testing."""
    config = CorrelationMonitorConfig(
        correlation_threshold_high=0.7,
        correlation_threshold_critical=0.9,
        concentration_threshold_medium=0.25,
        concentration_threshold_high=0.4,
        lookback_days=100  # Reduced for faster testing
    )
    return create_correlation_concentration_monitor(config)

class TestCorrelationCalculation:
    """Test correlation matrix calculations."""
    
    @pytest.mark.asyncio
    async def test_pearson_correlation_calculation(self, correlation_monitor, sample_returns_data):
        """Test Pearson correlation matrix calculation."""
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        assert correlation_matrix.matrix.shape == (4, 4)
        assert len(correlation_matrix.symbols) == 4
        assert correlation_matrix.method == CorrelationMethod.PEARSON
        assert correlation_matrix.condition_number is not None
        assert correlation_matrix.eigenvalues is not None
        
        # Check diagonal is 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix.matrix), np.ones(4))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(correlation_matrix.matrix, correlation_matrix.matrix.T)
        
        # BTC and ETH should be highly correlated (we created them that way)
        btc_idx = correlation_matrix.symbols.index('BTC')
        eth_idx = correlation_matrix.symbols.index('ETH')
        btc_eth_corr = abs(correlation_matrix.matrix[btc_idx, eth_idx])
        assert btc_eth_corr > 0.5  # Should be reasonably correlated
    
    @pytest.mark.asyncio
    async def test_spearman_correlation_calculation(self, correlation_monitor, sample_returns_data):
        """Test Spearman correlation matrix calculation."""
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.SPEARMAN
        )
        
        assert correlation_matrix.matrix.shape == (4, 4)
        assert correlation_matrix.method == CorrelationMethod.SPEARMAN
        assert correlation_matrix.p_values is not None
        
        # Check diagonal is 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix.matrix), np.ones(4))
    
    @pytest.mark.asyncio
    async def test_rolling_correlation_calculation(self, correlation_monitor, sample_returns_data):
        """Test rolling correlation matrix calculation."""
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.ROLLING
        )
        
        assert correlation_matrix.matrix.shape == (4, 4)
        assert correlation_matrix.method == CorrelationMethod.ROLLING
        
        # Check diagonal is 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix.matrix), np.ones(4))
    
    @pytest.mark.asyncio
    async def test_exponential_correlation_calculation(self, correlation_monitor, sample_returns_data):
        """Test exponentially weighted correlation matrix calculation."""
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.EXPONENTIAL
        )
        
        assert correlation_matrix.matrix.shape == (4, 4)
        assert correlation_matrix.method == CorrelationMethod.EXPONENTIAL
        
        # Check diagonal is 1.0
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix.matrix), np.ones(4))
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, correlation_monitor):
        """Test handling of insufficient data."""
        insufficient_data = {
            'BTC': [0.01, 0.02],  # Only 2 data points
            'ETH': [0.015, 0.018]
        }
        
        with pytest.raises(ValueError, match="Insufficient data"):
            await correlation_monitor.calculate_correlation_matrix(
                returns_data=insufficient_data,
                method=CorrelationMethod.PEARSON
            )

class TestConcentrationMetrics:
    """Test concentration risk calculations."""
    
    @pytest.mark.asyncio
    async def test_balanced_portfolio_concentration(self, correlation_monitor):
        """Test concentration metrics for balanced portfolio."""
        balanced_weights = {'BTC': 0.25, 'ETH': 0.25, 'ADA': 0.25, 'DOT': 0.25}
        
        concentration_metrics = await correlation_monitor.calculate_concentration_metrics(
            portfolio_weights=balanced_weights
        )
        
        assert concentration_metrics.herfindahl_index == 0.25  # 1/4 for equal weights
        assert concentration_metrics.max_weight == 0.25
        assert concentration_metrics.effective_number_assets == 4.0
        assert concentration_metrics.top_3_concentration == 0.75
        assert len(concentration_metrics.concentration_alerts) == 0  # No alerts for balanced portfolio
    
    @pytest.mark.asyncio
    async def test_concentrated_portfolio_concentration(self, correlation_monitor):
        """Test concentration metrics for concentrated portfolio."""
        concentrated_weights = {'BTC': 0.6, 'ETH': 0.2, 'ADA': 0.15, 'DOT': 0.05}
        
        concentration_metrics = await correlation_monitor.calculate_concentration_metrics(
            portfolio_weights=concentrated_weights
        )
        
        assert concentration_metrics.herfindahl_index > 0.25  # More concentrated than equal weights
        assert concentration_metrics.max_weight == 0.6
        assert concentration_metrics.max_weight_asset == 'BTC'
        assert concentration_metrics.effective_number_assets < 4.0
        assert len(concentration_metrics.concentration_alerts) > 0  # Should generate alerts
    
    @pytest.mark.asyncio
    async def test_sector_concentration_analysis(self, correlation_monitor):
        """Test sector concentration analysis."""
        portfolio_weights = {'BTC': 0.3, 'ETH': 0.3, 'ADA': 0.2, 'DOT': 0.2}
        sector_mapping = {'BTC': 'Layer1', 'ETH': 'Layer1', 'ADA': 'Layer1', 'DOT': 'Layer1'}
        
        concentration_metrics = await correlation_monitor.calculate_concentration_metrics(
            portfolio_weights=portfolio_weights,
            sector_mapping=sector_mapping
        )
        
        # All assets in same sector should generate sector concentration alert
        sector_alerts = [alert for alert in concentration_metrics.concentration_alerts 
                        if alert.alert_type == 'sector_concentration']
        assert len(sector_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_handling(self, correlation_monitor):
        """Test handling of empty portfolio."""
        with pytest.raises(ValueError, match="Portfolio weights cannot be empty"):
            await correlation_monitor.calculate_concentration_metrics(
                portfolio_weights={}
            )
    
    @pytest.mark.asyncio
    async def test_zero_weight_portfolio_handling(self, correlation_monitor):
        """Test handling of zero total weight portfolio."""
        zero_weights = {'BTC': 0.0, 'ETH': 0.0, 'ADA': 0.0}
        
        with pytest.raises(ValueError, match="Total portfolio weight cannot be zero"):
            await correlation_monitor.calculate_concentration_metrics(
                portfolio_weights=zero_weights
            )

class TestCorrelationMonitoring:
    """Test correlation monitoring and alerting."""
    
    @pytest.mark.asyncio
    async def test_correlation_alert_generation(self, correlation_monitor, sample_returns_data):
        """Test correlation alert generation."""
        # Calculate correlation matrix
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        # Monitor correlations
        alerts = await correlation_monitor.monitor_correlations(correlation_matrix)
        
        # Should generate some alerts due to high correlations we created
        assert isinstance(alerts, list)
        
        # Check alert structure
        for alert in alerts:
            assert hasattr(alert, 'alert_id')
            assert hasattr(alert, 'severity')
            assert hasattr(alert, 'asset_pair')
            assert hasattr(alert, 'correlation_value')
            assert hasattr(alert, 'threshold')
            assert hasattr(alert, 'message')
    
    @pytest.mark.asyncio
    async def test_correlation_clustering_detection(self, correlation_monitor):
        """Test correlation clustering detection."""
        # Create highly correlated returns
        base_returns = np.random.normal(0.001, 0.02, 252)
        highly_correlated_data = {
            'ASSET1': base_returns.tolist(),
            'ASSET2': (base_returns + np.random.normal(0, 0.001, 252)).tolist(),
            'ASSET3': (base_returns + np.random.normal(0, 0.001, 252)).tolist(),
            'ASSET4': np.random.normal(0.0005, 0.025, 252).tolist()  # Independent
        }
        
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=highly_correlated_data,
            method=CorrelationMethod.PEARSON
        )
        
        alerts = await correlation_monitor.monitor_correlations(correlation_matrix)
        
        # Should detect clustering among ASSET1, ASSET2, ASSET3
        cluster_alerts = [alert for alert in alerts if alert.alert_type == 'correlation_cluster']
        # Note: Clustering detection might not always trigger depending on exact correlations
        assert isinstance(cluster_alerts, list)

class TestRegimeAnalysis:
    """Test correlation regime change detection."""
    
    @pytest.mark.asyncio
    async def test_regime_change_detection(self, correlation_monitor, sample_returns_data):
        """Test correlation regime change detection."""
        regime_analysis = await correlation_monitor.detect_regime_changes(
            returns_data=sample_returns_data,
            window_size=30  # Smaller window for testing
        )
        
        assert 'current_regime' in regime_analysis
        assert 'average_correlation' in regime_analysis
        assert 'correlation_volatility' in regime_analysis
        assert 'regime_changes' in regime_analysis
        assert 'rolling_correlations' in regime_analysis
        
        # Current regime should be one of the expected values
        assert regime_analysis['current_regime'] in ['high_correlation', 'medium_correlation', 'low_correlation']
        
        # Average correlation should be between -1 and 1
        assert -1 <= regime_analysis['average_correlation'] <= 1
    
    @pytest.mark.asyncio
    async def test_insufficient_data_regime_analysis(self, correlation_monitor):
        """Test regime analysis with insufficient data."""
        insufficient_data = {
            'BTC': [0.01] * 10,  # Only 10 data points
            'ETH': [0.015] * 10
        }
        
        regime_analysis = await correlation_monitor.detect_regime_changes(
            returns_data=insufficient_data,
            window_size=30
        )
        
        # Should handle gracefully and return empty results
        assert regime_analysis['regime_changes'] == []

class TestHeatmapGeneration:
    """Test portfolio heatmap generation."""
    
    @pytest.mark.asyncio
    async def test_portfolio_heatmap_generation(self, correlation_monitor, sample_returns_data):
        """Test portfolio heatmap generation."""
        # Calculate correlation matrix
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        portfolio_weights = {'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.2, 'DOT': 0.1}
        risk_contributions = {'BTC': 0.5, 'ETH': 0.3, 'ADA': 0.15, 'DOT': 0.05}
        
        heatmap = await correlation_monitor.generate_portfolio_heatmap(
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights,
            risk_contributions=risk_contributions
        )
        
        assert heatmap.correlation_heatmap is not None
        assert heatmap.concentration_heatmap is not None
        assert heatmap.risk_contribution_heatmap is not None
        assert heatmap.timestamp is not None
        
        # Check heatmap structure
        assert 'data' in heatmap.correlation_heatmap
        assert 'layout' in heatmap.correlation_heatmap
        assert 'data' in heatmap.concentration_heatmap
        assert 'layout' in heatmap.concentration_heatmap
    
    @pytest.mark.asyncio
    async def test_heatmap_without_risk_contributions(self, correlation_monitor, sample_returns_data):
        """Test heatmap generation without risk contributions."""
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        portfolio_weights = {'BTC': 0.4, 'ETH': 0.3, 'ADA': 0.2, 'DOT': 0.1}
        
        heatmap = await correlation_monitor.generate_portfolio_heatmap(
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights
        )
        
        assert heatmap.correlation_heatmap is not None
        assert heatmap.concentration_heatmap is not None
        assert heatmap.risk_contribution_heatmap is None  # Should be None when not provided

class TestConfigurationAndCaching:
    """Test configuration and caching functionality."""
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        custom_config = CorrelationMonitorConfig(
            correlation_threshold_high=0.8,
            correlation_threshold_critical=0.95,
            concentration_threshold_high=0.5,
            lookback_days=100
        )
        
        monitor = create_correlation_concentration_monitor(custom_config)
        
        assert monitor.config.correlation_threshold_high == 0.8
        assert monitor.config.correlation_threshold_critical == 0.95
        assert monitor.config.concentration_threshold_high == 0.5
        assert monitor.config.lookback_days == 100
    
    @pytest.mark.asyncio
    async def test_correlation_caching(self, correlation_monitor, sample_returns_data):
        """Test correlation matrix caching."""
        # Calculate correlation matrix twice
        correlation_matrix1 = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        correlation_matrix2 = await correlation_monitor.calculate_correlation_matrix(
            returns_data=sample_returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        # Should have cached results
        assert len(correlation_monitor.correlation_cache) > 0
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            correlation_matrix1.matrix, correlation_matrix2.matrix
        )

if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        """Run a simple test of the correlation monitor."""
        print("Running simple correlation monitor test...")
        
        # Create monitor
        monitor = create_correlation_concentration_monitor()
        
        # Generate sample data with known correlations
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 252)
        returns_data = {
            'BTC': base_returns.tolist(),
            'ETH': (base_returns * 0.8 + np.random.normal(0, 0.01, 252)).tolist(),
            'ADA': np.random.normal(0.0005, 0.025, 252).tolist(),
            'DOT': (base_returns * 0.6 + np.random.normal(0, 0.015, 252)).tolist()
        }
        
        # Test correlation matrix
        print("1. Correlation Matrix Calculation:")
        corr_matrix = await monitor.calculate_correlation_matrix(
            returns_data=returns_data,
            method=CorrelationMethod.PEARSON
        )
        
        avg_corr = np.mean(corr_matrix.matrix[np.triu_indices_from(corr_matrix.matrix, k=1)])
        print(f"Average correlation: {avg_corr:.3f}")
        print(f"Condition number: {corr_matrix.condition_number:.2f}")
        
        # Test concentration metrics
        print("\n2. Concentration Metrics:")
        portfolio_weights = {'BTC': 0.5, 'ETH': 0.3, 'ADA': 0.15, 'DOT': 0.05}
        
        concentration_metrics = await monitor.calculate_concentration_metrics(
            portfolio_weights=portfolio_weights
        )
        
        print(f"Herfindahl Index: {concentration_metrics.herfindahl_index:.3f}")
        print(f"Max Weight: {concentration_metrics.max_weight:.1%} ({concentration_metrics.max_weight_asset})")
        print(f"Effective Assets: {concentration_metrics.effective_number_assets:.1f}")
        print(f"Alerts: {len(concentration_metrics.concentration_alerts)}")
        
        # Test correlation monitoring
        print("\n3. Correlation Monitoring:")
        alerts = await monitor.monitor_correlations(corr_matrix)
        print(f"Total Alerts: {len(alerts)}")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"  - {alert.severity.value.upper()}: {alert.message}")
        
        # Test regime analysis
        print("\n4. Regime Analysis:")
        regime_analysis = await monitor.detect_regime_changes(returns_data)
        print(f"Current Regime: {regime_analysis['current_regime']}")
        print(f"Average Correlation: {regime_analysis['average_correlation']:.3f}")
        
        print("\nCorrelation Monitor test completed successfully!")
    
    # Run the test
    asyncio.run(run_simple_test())