"""
Tests for the strategy optimization framework.
"""
import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List

from strategy_optimizer import (
    ParameterRange, Individual, GeneticAlgorithmOptimizer,
    MonteCarloSimulator, StrategyOptimizationFramework,
    SharpeRatioFitness, MultiObjectiveFitness
)
from backtesting_engine import (
    BacktestingConfig, SimpleMovingAverageStrategy, MarketData, PerformanceMetrics
)


class TestParameterRange:
    """Test suite for ParameterRange."""
    
    def test_float_parameter_generation(self):
        """Test float parameter generation."""
        param_range = ParameterRange(
            name="test_param",
            min_value=1.0,
            max_value=10.0,
            parameter_type="float"
        )
        
        for _ in range(100):
            value = param_range.generate_random_value()
            assert 1.0 <= value <= 10.0
            assert isinstance(value, float)
    
    def test_int_parameter_generation(self):
        """Test integer parameter generation."""
        param_range = ParameterRange(
            name="test_param",
            min_value=5,
            max_value=20,
            parameter_type="int"
        )
        
        for _ in range(100):
            value = param_range.generate_random_value()
            assert 5 <= value <= 20
            assert isinstance(value, int)
    
    def test_choice_parameter_generation(self):
        """Test choice parameter generation."""
        choices = ["option1", "option2", "option3"]
        param_range = ParameterRange(
            name="test_param",
            min_value=0,
            max_value=0,
            parameter_type="choice",
            choices=choices
        )
        
        for _ in range(100):
            value = param_range.generate_random_value()
            assert value in choices
    
    def test_parameter_mutation(self):
        """Test parameter mutation."""
        param_range = ParameterRange(
            name="test_param",
            min_value=1.0,
            max_value=10.0,
            parameter_type="float"
        )
        
        original_value = 5.0
        
        # Test with high mutation rate
        mutated_values = []
        for _ in range(100):
            mutated = param_range.mutate_value(original_value, mutation_rate=1.0)
            mutated_values.append(mutated)
        
        # Should have some mutations
        assert any(v != original_value for v in mutated_values)
        
        # All values should be within range
        assert all(1.0 <= v <= 10.0 for v in mutated_values)


class TestFitnessFunctions:
    """Test suite for fitness functions."""
    
    @pytest.fixture
    def sample_metrics(self) -> PerformanceMetrics:
        """Create sample performance metrics."""
        return PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            volatility=0.20,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            max_drawdown=0.08,
            max_drawdown_duration=30,
            win_rate=0.65,
            profit_factor=1.8,
            avg_win=0.02,
            avg_loss=0.015,
            total_trades=50,
            winning_trades=32,
            losing_trades=18,
            largest_win=0.08,
            largest_loss=-0.05,
            consecutive_wins=5,
            consecutive_losses=3,
            recovery_factor=1.9,
            ulcer_index=0.04,
            var_95=-0.03,
            expected_shortfall=-0.045,
            kelly_criterion=0.25,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            duration_days=365
        )
    
    def test_sharpe_ratio_fitness(self, sample_metrics):
        """Test Sharpe ratio fitness function."""
        fitness_func = SharpeRatioFitness(min_trades=10, drawdown_penalty=2.0)
        
        fitness = fitness_func.calculate_fitness(sample_metrics)
        
        # Should be positive for good metrics
        assert fitness > 0
        
        # Test with insufficient trades
        low_trade_metrics = sample_metrics
        low_trade_metrics.total_trades = 5
        
        low_fitness = fitness_func.calculate_fitness(low_trade_metrics)
        assert low_fitness < 0  # Should be heavily penalized
    
    def test_multi_objective_fitness(self, sample_metrics):
        """Test multi-objective fitness function."""
        fitness_func = MultiObjectiveFitness()
        
        fitness = fitness_func.calculate_fitness(sample_metrics)
        
        # Should be between 0 and 1
        assert 0 <= fitness <= 1
        
        # Test with poor metrics
        poor_metrics = sample_metrics
        poor_metrics.sharpe_ratio = -1.0
        poor_metrics.total_return = -0.3
        poor_metrics.max_drawdown = 0.5
        poor_metrics.win_rate = 0.2
        
        poor_fitness = fitness_func.calculate_fitness(poor_metrics)
        assert poor_fitness < fitness  # Should be lower


class TestGeneticAlgorithmOptimizer:
    """Test suite for GeneticAlgorithmOptimizer."""
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # Create trending data with volatility
            trend = i * 50
            noise = np.random.normal(0, 200)
            price = base_price + trend + noise
            
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 100,
                low=price - 100,
                close=price,
                volume=1000.0
            ))
        
        return data
    
    @pytest.fixture
    def parameter_ranges(self) -> List[ParameterRange]:
        """Create parameter ranges for testing."""
        return [
            ParameterRange(
                name="short_window",
                min_value=3,
                max_value=15,
                parameter_type="int"
            ),
            ParameterRange(
                name="long_window",
                min_value=10,
                max_value=50,
                parameter_type="int"
            ),
            ParameterRange(
                name="position_size",
                min_value=0.1,
                max_value=2.0,
                parameter_type="float"
            )
        ]
    
    @pytest.fixture
    def backtesting_config(self) -> BacktestingConfig:
        """Create backtesting configuration."""
        return BacktestingConfig(
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
    
    def test_population_initialization(self, parameter_ranges):
        """Test population initialization."""
        optimizer = GeneticAlgorithmOptimizer(population_size=20, generations=5)
        
        population = optimizer._initialize_population(parameter_ranges)
        
        assert len(population) == 20
        
        for individual in population:
            assert isinstance(individual, Individual)
            assert len(individual.parameters) == len(parameter_ranges)
            
            # Check parameter bounds
            assert 3 <= individual.parameters["short_window"] <= 15
            assert 10 <= individual.parameters["long_window"] <= 50
            assert 0.1 <= individual.parameters["position_size"] <= 2.0
    
    def test_tournament_selection(self, parameter_ranges):
        """Test tournament selection."""
        optimizer = GeneticAlgorithmOptimizer(population_size=10, generations=5)
        
        # Create population with known fitness values
        optimizer.population = []
        for i in range(10):
            individual = Individual(parameters={"test": i})
            individual.fitness = i  # Fitness = index
            optimizer.population.append(individual)
        
        # Tournament selection should favor higher fitness
        selected_fitness = []
        for _ in range(100):
            selected = optimizer._tournament_selection(tournament_size=3)
            selected_fitness.append(selected.fitness)
        
        # Average selected fitness should be higher than population average
        avg_selected = np.mean(selected_fitness)
        avg_population = np.mean([ind.fitness for ind in optimizer.population])
        
        assert avg_selected > avg_population
    
    def test_crossover(self, parameter_ranges):
        """Test crossover operation."""
        optimizer = GeneticAlgorithmOptimizer(population_size=10, generations=5)
        
        parent1 = Individual(parameters={"short_window": 5, "long_window": 20, "position_size": 1.0})
        parent2 = Individual(parameters={"short_window": 10, "long_window": 30, "position_size": 1.5})
        
        child1, child2 = optimizer._crossover(parent1, parent2, parameter_ranges)
        
        # Children should have parameters from both parents
        assert isinstance(child1, Individual)
        assert isinstance(child2, Individual)
        assert len(child1.parameters) == len(parent1.parameters)
        assert len(child2.parameters) == len(parent2.parameters)
    
    @pytest.mark.asyncio
    async def test_small_optimization(self, sample_market_data, parameter_ranges, backtesting_config):
        """Test small genetic algorithm optimization."""
        optimizer = GeneticAlgorithmOptimizer(
            population_size=5,  # Small for testing
            generations=3,      # Few generations for testing
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        result = await optimizer.optimize(
            SimpleMovingAverageStrategy,
            parameter_ranges,
            sample_market_data,
            backtesting_config
        )
        
        # Verify result structure
        assert result.best_individual is not None
        assert result.best_individual.fitness is not None
        assert len(result.population_history) <= 3  # Should have at most 3 generations
        assert result.total_evaluations > 0
        assert result.optimization_time > 0
        
        # Verify best individual has valid parameters
        best_params = result.best_individual.parameters
        assert 3 <= best_params["short_window"] <= 15
        assert 10 <= best_params["long_window"] <= 50
        assert 0.1 <= best_params["position_size"] <= 2.0


class TestMonteCarloSimulator:
    """Test suite for MonteCarloSimulator."""
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(50):  # Smaller dataset for testing
            price = base_price + i * 100 + np.random.normal(0, 100)
            
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 50,
                low=price - 50,
                close=price,
                volume=1000.0
            ))
        
        return data
    
    def test_block_bootstrap(self, sample_market_data):
        """Test block bootstrap method."""
        simulator = MonteCarloSimulator(num_simulations=10)
        
        bootstrapped = simulator._block_bootstrap(sample_market_data, block_size=5)
        
        assert len(bootstrapped) == len(sample_market_data)
        
        # Check that timestamps are sequential
        for i in range(1, len(bootstrapped)):
            time_diff = bootstrapped[i].timestamp - bootstrapped[i-1].timestamp
            assert time_diff == timedelta(hours=1)
    
    def test_simple_bootstrap(self, sample_market_data):
        """Test simple bootstrap method."""
        simulator = MonteCarloSimulator(num_simulations=10)
        
        bootstrapped = simulator._simple_bootstrap(sample_market_data)
        
        assert len(bootstrapped) == len(sample_market_data)
        
        # Check that timestamps are sequential
        for i in range(1, len(bootstrapped)):
            time_diff = bootstrapped[i].timestamp - bootstrapped[i-1].timestamp
            assert time_diff == timedelta(hours=1)
    
    @pytest.mark.asyncio
    async def test_monte_carlo_simulation(self, sample_market_data):
        """Test Monte Carlo simulation."""
        simulator = MonteCarloSimulator(num_simulations=5)  # Small number for testing
        
        strategy_parameters = {"short_window": 3, "long_window": 8, "position_size": 1.0}
        backtesting_config = BacktestingConfig()
        
        result = await simulator.run_simulation(
            SimpleMovingAverageStrategy,
            strategy_parameters,
            sample_market_data,
            backtesting_config,
            bootstrap_method="simple"
        )
        
        # Verify result structure
        assert 'simulations' in result
        assert 'analysis' in result
        assert 'num_simulations' in result
        
        # Should have some successful simulations
        assert result['num_simulations'] > 0
        
        # Verify analysis metrics
        analysis = result['analysis']
        assert 'mean_return' in analysis
        assert 'return_std' in analysis
        assert 'probability_positive_return' in analysis
        assert 0 <= analysis['probability_positive_return'] <= 1


class TestStrategyOptimizationFramework:
    """Test suite for StrategyOptimizationFramework."""
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(200):  # Larger dataset for comprehensive testing
            trend = i * 25
            noise = np.random.normal(0, 100)
            price = base_price + trend + noise
            
            data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 50,
                low=price - 50,
                close=price,
                volume=1000.0
            ))
        
        return data
    
    @pytest.fixture
    def parameter_ranges(self) -> List[ParameterRange]:
        """Create parameter ranges for testing."""
        return [
            ParameterRange(
                name="short_window",
                min_value=3,
                max_value=10,
                parameter_type="int"
            ),
            ParameterRange(
                name="long_window",
                min_value=10,
                max_value=20,
                parameter_type="int"
            ),
            ParameterRange(
                name="position_size",
                min_value=0.5,
                max_value=1.5,
                parameter_type="float"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, sample_market_data, parameter_ranges):
        """Test comprehensive strategy optimization."""
        framework = StrategyOptimizationFramework()
        
        # Use small parameters for testing
        optimization_config = {
            'initial_capital': 100000.0,
            'genetic_algorithm': {
                'population_size': 5,
                'generations': 3,
                'mutation_rate': 0.2
            },
            'walk_forward': {
                'training_period_days': 30,
                'testing_period_days': 10,
                'step_days': 5
            },
            'monte_carlo': {
                'num_simulations': 3,
                'bootstrap_method': 'simple'
            }
        }
        
        # Override Monte Carlo simulator for testing
        framework.monte_carlo_simulator = MonteCarloSimulator(num_simulations=3)
        
        result = await framework.optimize_strategy(
            SimpleMovingAverageStrategy,
            parameter_ranges,
            sample_market_data,
            optimization_config
        )
        
        # Verify comprehensive result structure
        assert 'best_parameters' in result
        assert 'genetic_algorithm' in result
        assert 'walk_forward_analysis' in result
        assert 'monte_carlo_simulation' in result
        assert 'parameter_ranges' in result
        
        # Verify best parameters are within ranges
        best_params = result['best_parameters']
        assert 3 <= best_params['short_window'] <= 10
        assert 10 <= best_params['long_window'] <= 20
        assert 0.5 <= best_params['position_size'] <= 1.5
        
        # Verify genetic algorithm results
        ga_result = result['genetic_algorithm']
        assert 'best_fitness' in ga_result
        assert 'optimization_time' in ga_result
        assert 'total_evaluations' in ga_result
        
        # Verify walk-forward analysis results
        wf_result = result['walk_forward_analysis']
        assert 'total_periods' in wf_result
        assert 'aggregated_metrics' in wf_result
        
        # Verify Monte Carlo simulation results
        mc_result = result['monte_carlo_simulation']
        assert 'analysis' in mc_result
        assert 'num_simulations' in mc_result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])