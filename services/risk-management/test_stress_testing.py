"""
Comprehensive tests for stress testing and scenario analysis functionality.
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from stress_testing_engine import (
    StressTestingEngine, StressTestType, ScenarioSeverity, StressScenario,
    StressTestResult, TailRiskMetrics, MonteCarloStressConfig,
    create_stress_testing_engine
)


class TestStressTestingEngine:
    """Test stress testing engine functionality."""
    
    @pytest.fixture
    def stress_engine(self):
        """Create stress testing engine for testing."""
        return create_stress_testing_engine()
    
    @pytest.fixture
    def sample_positions(self):
        """Sample portfolio positions."""
        return {
            'BTC': 50000,
            'ETH': 30000,
            'ADA': 20000,
            'DOT': 15000,
            'LINK': 10000
        }
    
    @pytest.fixture
    def sample_returns_data(self):
        """Sample returns data for testing."""
        np.random.seed(42)
        n_periods = 252  # 1 year of daily data
        
        returns_data = {}
        symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
        
        for symbol in symbols:
            # Generate correlated returns with different volatilities
            if symbol == 'BTC':
                returns = np.random.normal(0.001, 0.03, n_periods)
            elif symbol == 'ETH':
                returns = np.random.normal(0.0008, 0.035, n_periods)
            else:
                returns = np.random.normal(0.0005, 0.04, n_periods)
            
            returns_data[symbol] = returns.tolist()
        
        return returns_data
    
    async def test_historical_stress_testing(self, stress_engine, sample_positions):
        """Test historical stress test scenarios."""
        # Test with specific scenarios
        results = await stress_engine.run_historical_stress_test(
            positions=sample_positions,
            scenario_names=['covid_crash_2020', 'crypto_winter_2018']
        )
        
        assert len(results) == 2
        assert all(isinstance(result, StressTestResult) for result in results)
        
        # Check COVID crash scenario
        covid_result = next(r for r in results if 'covid' in r.scenario_name.lower())
        assert covid_result.total_pnl < 0  # Should be negative (loss)
        assert covid_result.scenario_type == StressTestType.HISTORICAL
        assert covid_result.var_breach is True  # Should breach VaR limits
        
        # Check crypto winter scenario
        winter_result = next(r for r in results if 'winter' in r.scenario_name.lower())
        assert winter_result.total_pnl < covid_result.total_pnl  # Should be worse
        assert winter_result.recovery_time_days > covid_result.recovery_time_days
    
    async def test_historical_stress_testing_all_scenarios(self, stress_engine, sample_positions):
        """Test historical stress testing with all scenarios."""
        # Test with all available scenarios
        results = await stress_engine.run_historical_stress_test(
            positions=sample_positions
        )
        
        assert len(results) > 0
        
        # Check that we have different types of scenarios
        scenario_types = set(result.scenario_type for result in results)
        assert StressTestType.HISTORICAL in scenario_types
        
        # Verify all results have required fields
        for result in results:
            assert result.scenario_name
            assert result.total_pnl is not None
            assert result.pnl_percentage is not None
            assert isinstance(result.asset_pnl, dict)
            assert isinstance(result.var_breach, bool)
    
    async def test_monte_carlo_stress_testing(self, stress_engine, sample_positions):
        """Test Monte Carlo stress testing framework."""
        config = MonteCarloStressConfig(
            n_simulations=1000,  # Smaller number for testing
            confidence_levels=[0.95, 0.99]
        )
        
        results = await stress_engine.run_monte_carlo_stress_test(
            positions=sample_positions,
            config=config
        )
        
        # Check results structure
        assert 'total_simulations' in results
        assert results['total_simulations'] == 1000
        assert 'worst_case_pnl' in results
        assert 'best_case_pnl' in results
        assert 'percentiles' in results
        assert 'tail_expectations' in results
        
        # Check percentiles
        percentiles = results['percentiles']
        assert 'VaR_95.0' in percentiles
        assert 'VaR_99.0' in percentiles
        assert percentiles['VaR_99.0'] >= percentiles['VaR_95.0']  # 99% VaR should be higher
        
        # Check tail expectations
        tail_exp = results['tail_expectations']
        assert 'tail_5pct_mean' in tail_exp
        assert 'tail_1pct_mean' in tail_exp
        assert tail_exp['tail_1pct_mean'] <= tail_exp['tail_5pct_mean']  # 1% tail should be worse
    
    async def test_tail_risk_analysis(self, stress_engine, sample_positions, sample_returns_data):
        """Test tail risk analysis functionality."""
        tail_metrics = await stress_engine.analyze_tail_risk(
            positions=sample_positions,
            returns_data=sample_returns_data
        )
        
        assert isinstance(tail_metrics, TailRiskMetrics)
        assert tail_metrics.tail_expectation < 0  # Should be negative (loss)
        assert tail_metrics.tail_variance > 0
        assert 0 <= tail_metrics.extreme_loss_probability <= 1
        assert tail_metrics.black_swan_threshold < tail_metrics.tail_expectation
        assert tail_metrics.maximum_credible_loss < 0
        
        # Check tail correlations
        assert isinstance(tail_metrics.tail_correlation, dict)
        assert len(tail_metrics.tail_correlation) == len(sample_positions)
        
        # Check that correlations are in valid range
        for symbol, corr in tail_metrics.tail_correlation.items():
            assert -1 <= corr <= 1
    
    async def test_black_swan_detection(self, stress_engine, sample_positions):
        """Test black swan event indicator detection."""
        # Create mock market data with various indicators
        market_data = {
            'volatility_data': [0.02, 0.025, 0.03, 0.08, 0.12],  # Volatility spike
            'correlation_data': [0.3, 0.35, 0.4, 0.8, 0.85],     # Correlation increase
            'volume_data': [1000, 950, 900, 400, 300],           # Volume drop
            'price_data': [100, 102, 98, 85, 95]                 # Price jumps
        }
        
        indicators = await stress_engine.detect_black_swan_indicators(
            market_data=market_data,
            positions=sample_positions
        )
        
        # Check indicator structure
        assert 'volatility_spike' in indicators
        assert 'correlation_breakdown' in indicators
        assert 'liquidity_stress' in indicators
        assert 'price_jumps' in indicators
        assert 'overall_assessment' in indicators
        
        # Check volatility spike detection
        vol_indicator = indicators['volatility_spike']
        assert vol_indicator['alert'] is True  # Should detect spike
        assert vol_indicator['severity'] in ['low', 'medium', 'high']
        
        # Check overall assessment
        overall = indicators['overall_assessment']
        assert 'risk_score' in overall
        assert 'alert_level' in overall
        assert 'recommendation' in overall
        assert 0 <= overall['risk_score'] <= 1
    
    async def test_custom_scenario_creation(self, stress_engine):
        """Test custom scenario creation."""
        scenario_id = await stress_engine.create_custom_scenario(
            name="Test Custom Scenario",
            description="A test scenario for unit testing",
            asset_shocks={'BTC': -0.3, 'ETH': -0.4, 'ADA': -0.5},
            severity=ScenarioSeverity.SEVERE
        )
        
        assert scenario_id == "custom_test_custom_scenario"
        assert scenario_id in stress_engine.custom_scenarios
        
        # Verify scenario properties
        scenario = stress_engine.custom_scenarios[scenario_id]
        assert scenario.name == "Test Custom Scenario"
        assert scenario.scenario_type == StressTestType.CUSTOM_SCENARIO
        assert scenario.severity == ScenarioSeverity.SEVERE
        assert scenario.asset_shocks['BTC'] == -0.3
    
    async def test_scenario_execution(self, stress_engine, sample_positions):
        """Test individual scenario execution."""
        # Create a test scenario
        test_scenario = StressScenario(
            name="Test Scenario",
            description="Test scenario for unit testing",
            scenario_type=StressTestType.CUSTOM_SCENARIO,
            severity=ScenarioSeverity.MODERATE,
            asset_shocks={'BTC': -0.2, 'ETH': -0.25, 'ADA': -0.3}
        )
        
        result = await stress_engine._execute_stress_scenario(sample_positions, test_scenario)
        
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "Test Scenario"
        assert result.total_pnl < 0  # Should be negative
        
        # Check individual asset P&L
        assert 'BTC' in result.asset_pnl
        assert 'ETH' in result.asset_pnl
        assert 'ADA' in result.asset_pnl
        
        # Verify calculations
        expected_btc_pnl = sample_positions['BTC'] * -0.2
        assert abs(result.asset_pnl['BTC'] - expected_btc_pnl) < 0.01
    
    def test_get_available_scenarios(self, stress_engine):
        """Test getting available scenarios."""
        scenarios = stress_engine.get_available_scenarios()
        
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0
        
        # Check that historical scenarios are included
        scenario_names = list(scenarios.keys())
        assert 'covid_crash_2020' in scenario_names
        assert 'crypto_winter_2018' in scenario_names
        
        # Check scenario structure
        for scenario_id, scenario_info in scenarios.items():
            assert 'name' in scenario_info
            assert 'description' in scenario_info
            assert 'type' in scenario_info
            assert 'severity' in scenario_info
    
    async def test_scenario_recommendations(self, stress_engine, sample_positions):
        """Test scenario recommendations based on portfolio."""
        recommendations = await stress_engine.get_scenario_recommendations(
            positions=sample_positions,
            risk_tolerance="moderate"
        )
        
        assert 'recommended_scenarios' in recommendations
        assert 'risk_assessment' in recommendations
        assert 'hedging_suggestions' in recommendations
        assert 'monitoring_priorities' in recommendations
        
        # Check risk assessment
        risk_assessment = recommendations['risk_assessment']
        assert 'crypto_concentration' in risk_assessment
        assert 'risk_level' in risk_assessment
        assert 'diversification_score' in risk_assessment
        
        # Should recommend crypto-specific scenarios for crypto-heavy portfolio
        recommended = recommendations['recommended_scenarios']
        assert 'covid_crash_2020' in recommended or 'crypto_winter_2018' in recommended
    
    async def test_stress_testing_with_empty_positions(self, stress_engine):
        """Test stress testing with empty positions."""
        empty_positions = {}
        
        results = await stress_engine.run_historical_stress_test(
            positions=empty_positions,
            scenario_names=['covid_crash_2020']
        )
        
        assert len(results) == 1
        assert results[0].total_pnl == 0.0
        assert results[0].pnl_percentage == 0.0
    
    async def test_stress_testing_with_single_asset(self, stress_engine):
        """Test stress testing with single asset portfolio."""
        single_asset_positions = {'BTC': 100000}
        
        results = await stress_engine.run_historical_stress_test(
            positions=single_asset_positions,
            scenario_names=['covid_crash_2020']
        )
        
        assert len(results) == 1
        result = results[0]
        
        # Should have loss only for BTC
        assert result.total_pnl < 0
        assert len(result.asset_pnl) == 1
        assert 'BTC' in result.asset_pnl
        assert result.asset_pnl['BTC'] == result.total_pnl
    
    async def test_monte_carlo_with_different_configs(self, stress_engine, sample_positions):
        """Test Monte Carlo with different configurations."""
        # Test with minimal simulations
        config_small = MonteCarloStressConfig(n_simulations=100)
        results_small = await stress_engine.run_monte_carlo_stress_test(
            positions=sample_positions,
            config=config_small
        )
        
        # Test with more simulations
        config_large = MonteCarloStressConfig(n_simulations=5000)
        results_large = await stress_engine.run_monte_carlo_stress_test(
            positions=sample_positions,
            config=config_large
        )
        
        # Both should complete successfully
        assert results_small['total_simulations'] == 100
        assert results_large['total_simulations'] == 5000
        
        # Larger simulation should have more stable results (lower variance in percentiles)
        # This is a statistical property, so we just check they're reasonable
        assert 'VaR_95.0' in results_small['percentiles']
        assert 'VaR_95.0' in results_large['percentiles']
    
    async def test_tail_risk_with_extreme_returns(self, stress_engine, sample_positions):
        """Test tail risk analysis with extreme returns data."""
        # Create returns data with extreme events
        extreme_returns_data = {}
        for symbol in sample_positions.keys():
            # Normal returns with some extreme events
            returns = np.random.normal(0.001, 0.02, 250).tolist()
            # Add extreme events
            returns.extend([-0.5, -0.3, -0.4])  # Extreme losses
            extreme_returns_data[symbol] = returns
        
        tail_metrics = await stress_engine.analyze_tail_risk(
            positions=sample_positions,
            returns_data=extreme_returns_data
        )
        
        # Should detect extreme characteristics
        assert tail_metrics.extreme_loss_probability > 0
        assert tail_metrics.fat_tail_indicator > 0  # Should indicate fat tails
        assert tail_metrics.maximum_credible_loss < tail_metrics.tail_expectation
    
    async def test_error_handling(self, stress_engine):
        """Test error handling in stress testing."""
        # Test with invalid scenario name
        results = await stress_engine.run_historical_stress_test(
            positions={'BTC': 1000},
            scenario_names=['nonexistent_scenario']
        )
        
        # Should return empty results for invalid scenarios
        assert len(results) == 0
        
        # Test with invalid positions format
        with pytest.raises(Exception):
            await stress_engine.analyze_tail_risk(
                positions={'BTC': 1000},
                returns_data={}  # Empty returns data
            )


class TestStressTestingIntegration:
    """Integration tests for stress testing with other components."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system for testing."""
        stress_engine = create_stress_testing_engine()
        
        return {
            'stress_engine': stress_engine
        }
    
    async def test_end_to_end_stress_testing_workflow(self, integrated_system):
        """Test complete stress testing workflow."""
        stress_engine = integrated_system['stress_engine']
        
        # Portfolio setup
        positions = {
            'BTC': 60000,
            'ETH': 40000,
            'ADA': 25000,
            'DOT': 20000
        }
        
        # 1. Get scenario recommendations
        recommendations = await stress_engine.get_scenario_recommendations(positions)
        assert len(recommendations['recommended_scenarios']) > 0
        
        # 2. Run recommended historical scenarios
        historical_results = await stress_engine.run_historical_stress_test(
            positions=positions,
            scenario_names=recommendations['recommended_scenarios'][:2]  # Test first 2
        )
        assert len(historical_results) > 0
        
        # 3. Run Monte Carlo stress test
        mc_config = MonteCarloStressConfig(n_simulations=1000)
        mc_results = await stress_engine.run_monte_carlo_stress_test(positions, mc_config)
        assert mc_results['total_simulations'] == 1000
        
        # 4. Create and test custom scenario
        scenario_id = await stress_engine.create_custom_scenario(
            name="Integration Test Scenario",
            description="Custom scenario for integration testing",
            asset_shocks={'BTC': -0.25, 'ETH': -0.3, 'ADA': -0.35, 'DOT': -0.4}
        )
        
        custom_results = await stress_engine.run_historical_stress_test(
            positions=positions,
            scenario_names=[scenario_id]
        )
        assert len(custom_results) == 1
        
        # 5. Verify all results are consistent
        for result in historical_results + custom_results:
            assert result.total_pnl < 0  # All should show losses
            assert abs(result.pnl_percentage) > 0  # Should have meaningful impact
    
    async def test_stress_testing_performance(self, integrated_system):
        """Test stress testing performance with larger datasets."""
        stress_engine = integrated_system['stress_engine']
        
        # Large portfolio
        large_positions = {f'ASSET_{i}': 10000 for i in range(20)}
        
        # Measure time for Monte Carlo stress test
        import time
        start_time = time.time()
        
        config = MonteCarloStressConfig(n_simulations=5000)
        results = await stress_engine.run_monte_carlo_stress_test(large_positions, config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (< 30 seconds)
        assert execution_time < 30
        assert results['total_simulations'] == 5000
        
        # Results should be reasonable
        assert results['worst_case_pnl'] < results['best_case_pnl']
        assert 'VaR_95.0' in results['percentiles']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])