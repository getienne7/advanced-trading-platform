"""
Demonstration of the strategy optimization framework.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from strategy_optimizer import (
    ParameterRange, GeneticAlgorithmOptimizer, StrategyOptimizationFramework
)
from backtesting_engine import (
    BacktestingConfig, SimpleMovingAverageStrategy, MarketData
)

async def demo_genetic_optimization():
    """Demonstrate genetic algorithm optimization."""
    print("=== Genetic Algorithm Optimization Demo ===")
    
    # Create sample market data with clear trend
    market_data = []
    base_time = datetime(2023, 1, 1)
    base_price = 50000.0
    
    for i in range(150):
        # Create trending data with some volatility
        trend = i * 30  # Strong uptrend
        cycle = 500 * (0.5 + 0.5 * (i % 20 - 10) / 10)  # Cyclical component
        noise = (i % 7 - 3) * 50  # Some noise
        price = base_price + trend + cycle + noise
        
        market_data.append(MarketData(
            timestamp=base_time + timedelta(hours=i),
            symbol="BTCUSDT",
            open=price,
            high=price + 100,
            low=price - 100,
            close=price,
            volume=1000.0
        ))
    
    # Define parameter ranges for optimization
    parameter_ranges = [
        ParameterRange(
            name="short_window",
            min_value=3,
            max_value=15,
            parameter_type="int"
        ),
        ParameterRange(
            name="long_window",
            min_value=10,
            max_value=40,
            parameter_type="int"
        ),
        ParameterRange(
            name="position_size",
            min_value=0.1,
            max_value=2.0,
            parameter_type="float"
        )
    ]
    
    # Create genetic algorithm optimizer
    optimizer = GeneticAlgorithmOptimizer(
        population_size=20,
        generations=10,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=3
    )
    
    # Backtesting configuration
    config = BacktestingConfig(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    print("Starting genetic algorithm optimization...")
    print(f"Population size: {optimizer.population_size}")
    print(f"Generations: {optimizer.generations}")
    print(f"Parameter ranges: {len(parameter_ranges)}")
    
    # Run optimization
    result = await optimizer.optimize(
        SimpleMovingAverageStrategy,
        parameter_ranges,
        market_data,
        config
    )
    
    # Display results
    print("\n=== Optimization Results ===")
    print(f"Best fitness: {result.best_individual.fitness:.4f}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Convergence generation: {result.convergence_generation}")
    
    print("\n=== Best Parameters ===")
    best_params = result.best_individual.parameters
    for param_name, value in best_params.items():
        print(f"{param_name}: {value}")
    
    print("\n=== Best Performance Metrics ===")
    metrics = result.best_individual.performance_metrics
    if metrics:
        print(f"Total Return: {metrics.total_return:.4f}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.4f}")
        print(f"Win Rate: {metrics.win_rate:.4f}")
        print(f"Total Trades: {metrics.total_trades}")
    
    return result

async def demo_comprehensive_optimization():
    """Demonstrate comprehensive optimization framework."""
    print("\n\n=== Comprehensive Optimization Demo ===")
    
    # Create sample market data
    market_data = []
    base_time = datetime(2023, 1, 1)
    base_price = 50000.0
    
    for i in range(300):  # Longer dataset for comprehensive testing
        # Create more complex price pattern
        trend = i * 20
        cycle1 = 800 * (0.5 + 0.5 * (i % 30 - 15) / 15)  # 30-period cycle
        cycle2 = 300 * (0.5 + 0.5 * (i % 10 - 5) / 5)    # 10-period cycle
        noise = (i % 13 - 6) * 30  # Noise
        price = base_price + trend + cycle1 + cycle2 + noise
        
        market_data.append(MarketData(
            timestamp=base_time + timedelta(hours=i),
            symbol="BTCUSDT",
            open=price,
            high=price + 80,
            low=price - 80,
            close=price,
            volume=1000.0
        ))
    
    # Define parameter ranges
    parameter_ranges = [
        ParameterRange(
            name="short_window",
            min_value=3,
            max_value=12,
            parameter_type="int"
        ),
        ParameterRange(
            name="long_window",
            min_value=15,
            max_value=35,
            parameter_type="int"
        ),
        ParameterRange(
            name="position_size",
            min_value=0.2,
            max_value=1.5,
            parameter_type="float"
        )
    ]
    
    # Create optimization framework
    framework = StrategyOptimizationFramework()
    
    # Configuration for comprehensive optimization
    optimization_config = {
        'initial_capital': 100000.0,
        'commission_rate': 0.001,
        'slippage_rate': 0.0005,
        'genetic_algorithm': {
            'population_size': 15,
            'generations': 8,
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'elite_size': 2
        },
        'walk_forward': {
            'training_period_days': 60,
            'testing_period_days': 20,
            'step_days': 10
        },
        'monte_carlo': {
            'num_simulations': 10,
            'bootstrap_method': 'block'
        }
    }
    
    print("Starting comprehensive optimization...")
    print("This includes: Genetic Algorithm + Walk-Forward Analysis + Monte Carlo Simulation")
    
    # Run comprehensive optimization
    result = await framework.optimize_strategy(
        SimpleMovingAverageStrategy,
        parameter_ranges,
        market_data,
        optimization_config
    )
    
    # Display results
    print("\n=== Comprehensive Optimization Results ===")
    
    # Genetic Algorithm Results
    ga_result = result['genetic_algorithm']
    print(f"\nGenetic Algorithm:")
    print(f"  Best fitness: {ga_result['best_fitness']:.4f}")
    print(f"  Optimization time: {ga_result['optimization_time']:.2f} seconds")
    print(f"  Total evaluations: {ga_result['total_evaluations']}")
    
    # Best Parameters
    print(f"\nBest Parameters:")
    for param_name, value in result['best_parameters'].items():
        print(f"  {param_name}: {value}")
    
    # Walk-Forward Analysis Results
    wf_result = result['walk_forward_analysis']
    print(f"\nWalk-Forward Analysis:")
    print(f"  Total periods: {wf_result.get('total_periods', 0)}")
    if 'aggregated_metrics' in wf_result:
        agg_metrics = wf_result['aggregated_metrics']
        print(f"  Mean return: {agg_metrics.get('mean_return', 0):.4f}")
        print(f"  Consistency score: {agg_metrics.get('consistency_score', 0):.4f}")
    
    # Monte Carlo Simulation Results
    mc_result = result['monte_carlo_simulation']
    print(f"\nMonte Carlo Simulation:")
    print(f"  Simulations: {mc_result.get('num_simulations', 0)}")
    if 'analysis' in mc_result:
        mc_analysis = mc_result['analysis']
        print(f"  Mean return: {mc_analysis.get('mean_return', 0):.4f}")
        print(f"  Return std: {mc_analysis.get('return_std', 0):.4f}")
        print(f"  Probability positive return: {mc_analysis.get('probability_positive_return', 0):.4f}")
        print(f"  Value at Risk (5%): {mc_analysis.get('value_at_risk_5', 0):.4f}")
    
    return result

async def main():
    """Run optimization demonstrations."""
    print("Strategy Optimization Framework Demonstration")
    print("=" * 50)
    
    # Run genetic algorithm demo
    ga_result = await demo_genetic_optimization()
    
    # Run comprehensive optimization demo
    comp_result = await demo_comprehensive_optimization()
    
    print("\n" + "=" * 50)
    print("Optimization demonstrations completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())