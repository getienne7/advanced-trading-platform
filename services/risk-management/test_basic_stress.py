#!/usr/bin/env python3
"""
Basic test for stress testing engine functionality
"""
import asyncio
from stress_testing_engine import create_stress_testing_engine, MonteCarloStressConfig


async def test_stress_engine():
    """Test basic stress testing functionality."""
    print('Testing Stress Testing Engine...')
    
    # Create engine
    engine = create_stress_testing_engine()
    
    # Test portfolio
    positions = {'BTC': 50000, 'ETH': 30000, 'ADA': 20000}
    
    # Test historical scenarios
    print('Running historical stress tests...')
    historical_results = await engine.run_historical_stress_test(
        positions=positions,
        scenario_names=['covid_crash_2020']
    )
    
    if historical_results:
        result = historical_results[0]
        print(f'✓ COVID-19 scenario: {result.total_pnl:,.0f} USD ({result.pnl_percentage:.1%})')
    
    # Test Monte Carlo
    print('Running Monte Carlo stress test...')
    config = MonteCarloStressConfig(n_simulations=1000)
    mc_results = await engine.run_monte_carlo_stress_test(positions, config)
    
    var_95 = mc_results["percentiles"]["VaR_95.0"]
    worst_case = mc_results["worst_case_pnl"]
    
    print(f'✓ Monte Carlo VaR 95%: {var_95:,.0f} USD')
    print(f'✓ Worst case: {worst_case:,.0f} USD')
    
    # Test available scenarios
    scenarios = engine.get_available_scenarios()
    print(f'✓ Available scenarios: {len(scenarios)}')
    
    print('🎉 Stress Testing Engine test completed successfully!')


if __name__ == "__main__":
    asyncio.run(test_stress_engine())