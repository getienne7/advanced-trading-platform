"""
Strategy Optimization Framework
Implements genetic algorithm for parameter optimization, walk-forward analysis, and Monte Carlo simulation.
"""
import asyncio
import numpy as np
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import structlog
from pydantic import BaseModel, Field

from backtesting_engine import (
    BacktestingEngine, BacktestingConfig, WalkForwardAnalyzer,
    TradingStrategy, MarketData, PerformanceMetrics
)

# Configure logging
logger = structlog.get_logger("strategy-optimizer")


@dataclass
class ParameterRange:
    """Parameter range definition for optimization."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    parameter_type: str = "float"  # "float", "int", "choice"
    choices: Optional[List[Any]] = None
    
    def generate_random_value(self) -> Any:
        """Generate a random value within the parameter range."""
        if self.parameter_type == "choice" and self.choices:
            return random.choice(self.choices)
        elif self.parameter_type == "int":
            return random.randint(int(self.min_value), int(self.max_value))
        else:  # float
            return random.uniform(self.min_value, self.max_value)
    
    def mutate_value(self, current_value: Any, mutation_rate: float = 0.1) -> Any:
        """Mutate a parameter value."""
        if random.random() > mutation_rate:
            return current_value
        
        if self.parameter_type == "choice" and self.choices:
            return random.choice(self.choices)
        elif self.parameter_type == "int":
            # Mutate by ±20% of range
            mutation_range = int((self.max_value - self.min_value) * 0.2)
            new_value = current_value + random.randint(-mutation_range, mutation_range)
            return max(int(self.min_value), min(int(self.max_value), new_value))
        else:  # float
            # Mutate by ±20% of range
            mutation_range = (self.max_value - self.min_value) * 0.2
            new_value = current_value + random.uniform(-mutation_range, mutation_range)
            return max(self.min_value, min(self.max_value, new_value))


@dataclass
class Individual:
    """Individual in genetic algorithm population."""
    parameters: Dict[str, Any]
    fitness: Optional[float] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    generation: int = 0
    
    def __post_init__(self):
        self.id = f"ind_{hash(str(self.parameters))}"


@dataclass
class OptimizationResult:
    """Result of strategy optimization."""
    best_individual: Individual
    population_history: List[List[Individual]]
    optimization_metrics: Dict[str, Any]
    parameter_ranges: List[ParameterRange]
    total_evaluations: int
    optimization_time: float
    convergence_generation: Optional[int] = None


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    @abstractmethod
    def calculate_fitness(self, performance_metrics: PerformanceMetrics) -> float:
        """Calculate fitness score from performance metrics."""
        pass


class SharpeRatioFitness(FitnessFunction):
    """Fitness function based on Sharpe ratio."""
    
    def __init__(self, min_trades: int = 10, drawdown_penalty: float = 2.0):
        self.min_trades = min_trades
        self.drawdown_penalty = drawdown_penalty
    
    def calculate_fitness(self, performance_metrics: PerformanceMetrics) -> float:
        """Calculate fitness based on Sharpe ratio with penalties."""
        if performance_metrics.total_trades < self.min_trades:
            return -1000.0  # Heavy penalty for insufficient trades
        
        base_fitness = performance_metrics.sharpe_ratio
        
        # Apply drawdown penalty
        drawdown_penalty = performance_metrics.max_drawdown * self.drawdown_penalty
        
        # Apply win rate bonus
        win_rate_bonus = performance_metrics.win_rate * 0.5
        
        return base_fitness - drawdown_penalty + win_rate_bonus


class MultiObjectiveFitness(FitnessFunction):
    """Multi-objective fitness function combining multiple metrics."""
    
    def __init__(self, 
                 sharpe_weight: float = 0.4,
                 return_weight: float = 0.3,
                 drawdown_weight: float = 0.2,
                 win_rate_weight: float = 0.1):
        self.sharpe_weight = sharpe_weight
        self.return_weight = return_weight
        self.drawdown_weight = drawdown_weight
        self.win_rate_weight = win_rate_weight
    
    def calculate_fitness(self, performance_metrics: PerformanceMetrics) -> float:
        """Calculate multi-objective fitness score."""
        # Normalize metrics to 0-1 scale
        sharpe_score = max(0, min(1, (performance_metrics.sharpe_ratio + 2) / 4))  # -2 to 2 -> 0 to 1
        return_score = max(0, min(1, performance_metrics.total_return + 0.5))  # -0.5 to 0.5 -> 0 to 1
        drawdown_score = 1 - performance_metrics.max_drawdown  # Lower drawdown is better
        win_rate_score = performance_metrics.win_rate
        
        fitness = (
            self.sharpe_weight * sharpe_score +
            self.return_weight * return_score +
            self.drawdown_weight * drawdown_score +
            self.win_rate_weight * win_rate_score
        )
        
        return fitness


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for strategy parameter optimization."""
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 fitness_function: Optional[FitnessFunction] = None,
                 parallel_evaluation: bool = True,
                 max_workers: Optional[int] = None):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.fitness_function = fitness_function or SharpeRatioFitness()
        self.parallel_evaluation = parallel_evaluation
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        self.population = []
        self.population_history = []
        self.best_individual = None
        
    async def optimize(self,
                      strategy_class: type,
                      parameter_ranges: List[ParameterRange],
                      market_data: List[MarketData],
                      backtesting_config: BacktestingConfig) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        start_time = datetime.now()
        
        logger.info("Starting genetic algorithm optimization",
                   population_size=self.population_size,
                   generations=self.generations,
                   parameters=len(parameter_ranges))
        
        # Initialize population
        self.population = self._initialize_population(parameter_ranges)
        
        # Evaluate initial population
        await self._evaluate_population(strategy_class, market_data, backtesting_config)
        
        convergence_generation = None
        best_fitness_history = []
        
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Store population history
            self.population_history.append(copy.deepcopy(self.population))
            
            # Track best fitness
            current_best = max(self.population, key=lambda x: x.fitness or -float('inf'))
            best_fitness_history.append(current_best.fitness)
            
            # Check for convergence
            if len(best_fitness_history) >= 10:
                recent_improvements = [
                    best_fitness_history[i] - best_fitness_history[i-1] 
                    for i in range(-9, 0)
                ]
                if all(improvement < 0.001 for improvement in recent_improvements):
                    convergence_generation = generation
                    logger.info(f"Convergence detected at generation {generation}")
                    break
            
            # Selection, crossover, and mutation
            new_population = self._evolve_population(parameter_ranges)
            
            # Evaluate new population
            self.population = new_population
            await self._evaluate_population(strategy_class, market_data, backtesting_config)
            
            # Log progress
            best_individual = max(self.population, key=lambda x: x.fitness or -float('inf'))
            logger.info(f"Generation {generation + 1} best fitness: {best_individual.fitness:.4f}")
        
        # Final results
        self.best_individual = max(self.population, key=lambda x: x.fitness or -float('inf'))
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        optimization_metrics = {
            'final_population_size': len(self.population),
            'best_fitness': self.best_individual.fitness,
            'fitness_history': best_fitness_history,
            'convergence_generation': convergence_generation,
            'total_evaluations': len(self.population_history) * self.population_size
        }
        
        logger.info("Genetic algorithm optimization completed",
                   best_fitness=self.best_individual.fitness,
                   optimization_time=optimization_time,
                   total_evaluations=optimization_metrics['total_evaluations'])
        
        return OptimizationResult(
            best_individual=self.best_individual,
            population_history=self.population_history,
            optimization_metrics=optimization_metrics,
            parameter_ranges=parameter_ranges,
            total_evaluations=optimization_metrics['total_evaluations'],
            optimization_time=optimization_time,
            convergence_generation=convergence_generation
        )
    
    def _initialize_population(self, parameter_ranges: List[ParameterRange]) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for i in range(self.population_size):
            parameters = {}
            for param_range in parameter_ranges:
                parameters[param_range.name] = param_range.generate_random_value()
            
            individual = Individual(parameters=parameters, generation=0)
            population.append(individual)
        
        return population
    
    async def _evaluate_population(self,
                                 strategy_class: type,
                                 market_data: List[MarketData],
                                 backtesting_config: BacktestingConfig):
        """Evaluate fitness for entire population."""
        if self.parallel_evaluation:
            await self._evaluate_population_parallel(strategy_class, market_data, backtesting_config)
        else:
            await self._evaluate_population_sequential(strategy_class, market_data, backtesting_config)
    
    async def _evaluate_population_sequential(self,
                                            strategy_class: type,
                                            market_data: List[MarketData],
                                            backtesting_config: BacktestingConfig):
        """Evaluate population sequentially."""
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness, individual.performance_metrics = await self._evaluate_individual(
                    individual, strategy_class, market_data, backtesting_config
                )
    
    async def _evaluate_population_parallel(self,
                                          strategy_class: type,
                                          market_data: List[MarketData],
                                          backtesting_config: BacktestingConfig):
        """Evaluate population in parallel."""
        # For now, use sequential evaluation as async parallel evaluation is complex
        # In production, this could use ProcessPoolExecutor with proper serialization
        await self._evaluate_population_sequential(strategy_class, market_data, backtesting_config)
    
    async def _evaluate_individual(self,
                                 individual: Individual,
                                 strategy_class: type,
                                 market_data: List[MarketData],
                                 backtesting_config: BacktestingConfig) -> Tuple[float, PerformanceMetrics]:
        """Evaluate a single individual."""
        try:
            # Create strategy with individual's parameters
            strategy = strategy_class(f"opt_strategy_{individual.id}", individual.parameters)
            
            # Run backtest
            engine = BacktestingEngine(backtesting_config)
            result = await engine.run_backtest(strategy, market_data)
            
            # Calculate fitness
            performance_metrics = result['performance_metrics']
            fitness = self.fitness_function.calculate_fitness(performance_metrics)
            
            return fitness, performance_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating individual {individual.id}: {str(e)}")
            return -1000.0, None  # Heavy penalty for failed evaluation
    
    def _evolve_population(self, parameter_ranges: List[ParameterRange]) -> List[Individual]:
        """Evolve population through selection, crossover, and mutation."""
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness or -float('inf'), reverse=True)
        
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(self.elite_size):
            elite = copy.deepcopy(self.population[i])
            elite.generation += 1
            new_population.append(elite)
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, parameter_ranges)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            self._mutate(child1, parameter_ranges)
            self._mutate(child2, parameter_ranges)
            
            # Add to new population
            child1.generation = parent1.generation + 1
            child2.generation = parent2.generation + 1
            child1.fitness = None  # Reset fitness for re-evaluation
            child2.fitness = None
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness or -float('inf'))
    
    def _crossover(self, 
                  parent1: Individual, 
                  parent2: Individual,
                  parameter_ranges: List[ParameterRange]) -> Tuple[Individual, Individual]:
        """Uniform crossover between two parents."""
        child1_params = {}
        child2_params = {}
        
        for param_range in parameter_ranges:
            param_name = param_range.name
            
            if random.random() < 0.5:
                child1_params[param_name] = parent1.parameters[param_name]
                child2_params[param_name] = parent2.parameters[param_name]
            else:
                child1_params[param_name] = parent2.parameters[param_name]
                child2_params[param_name] = parent1.parameters[param_name]
        
        child1 = Individual(parameters=child1_params)
        child2 = Individual(parameters=child2_params)
        
        return child1, child2
    
    def _mutate(self, individual: Individual, parameter_ranges: List[ParameterRange]):
        """Mutate individual parameters."""
        for param_range in parameter_ranges:
            param_name = param_range.name
            current_value = individual.parameters[param_name]
            individual.parameters[param_name] = param_range.mutate_value(current_value, self.mutation_rate)


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy validation."""
    
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
    
    async def run_simulation(self,
                           strategy_class: type,
                           strategy_parameters: Dict[str, Any],
                           market_data: List[MarketData],
                           backtesting_config: BacktestingConfig,
                           bootstrap_method: str = "block") -> Dict[str, Any]:
        """Run Monte Carlo simulation on strategy."""
        logger.info("Starting Monte Carlo simulation",
                   num_simulations=self.num_simulations,
                   bootstrap_method=bootstrap_method)
        
        results = []
        
        for i in range(self.num_simulations):
            if i % 100 == 0:
                logger.info(f"Monte Carlo simulation progress: {i}/{self.num_simulations}")
            
            # Bootstrap market data
            if bootstrap_method == "block":
                bootstrapped_data = self._block_bootstrap(market_data)
            else:
                bootstrapped_data = self._simple_bootstrap(market_data)
            
            # Run backtest on bootstrapped data
            try:
                strategy = strategy_class(f"mc_strategy_{i}", strategy_parameters)
                engine = BacktestingEngine(backtesting_config)
                result = await engine.run_backtest(strategy, bootstrapped_data)
                
                results.append({
                    'simulation': i,
                    'total_return': result['performance_metrics'].total_return,
                    'sharpe_ratio': result['performance_metrics'].sharpe_ratio,
                    'max_drawdown': result['performance_metrics'].max_drawdown,
                    'win_rate': result['performance_metrics'].win_rate,
                    'total_trades': result['total_trades'],
                    'final_capital': result['final_capital']
                })
                
            except Exception as e:
                logger.error(f"Monte Carlo simulation {i} failed: {str(e)}")
                continue
        
        # Analyze results
        analysis = self._analyze_monte_carlo_results(results)
        
        logger.info("Monte Carlo simulation completed",
                   successful_simulations=len(results),
                   mean_return=analysis['mean_return'],
                   return_std=analysis['return_std'])
        
        return {
            'simulations': results,
            'analysis': analysis,
            'num_simulations': len(results),
            'bootstrap_method': bootstrap_method
        }
    
    def _block_bootstrap(self, market_data: List[MarketData], block_size: int = 24) -> List[MarketData]:
        """Block bootstrap to preserve time series structure."""
        n = len(market_data)
        num_blocks = n // block_size
        
        bootstrapped_data = []
        
        for _ in range(num_blocks):
            start_idx = random.randint(0, n - block_size)
            block = market_data[start_idx:start_idx + block_size]
            bootstrapped_data.extend(block)
        
        # Adjust timestamps to be sequential
        base_time = market_data[0].timestamp
        for i, data_point in enumerate(bootstrapped_data):
            data_point.timestamp = base_time + timedelta(hours=i)
        
        return bootstrapped_data[:n]  # Trim to original length
    
    def _simple_bootstrap(self, market_data: List[MarketData]) -> List[MarketData]:
        """Simple bootstrap sampling."""
        n = len(market_data)
        bootstrapped_indices = [random.randint(0, n-1) for _ in range(n)]
        
        bootstrapped_data = []
        base_time = market_data[0].timestamp
        
        for i, idx in enumerate(bootstrapped_indices):
            data_point = copy.deepcopy(market_data[idx])
            data_point.timestamp = base_time + timedelta(hours=i)
            bootstrapped_data.append(data_point)
        
        return bootstrapped_data
    
    def _analyze_monte_carlo_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        if not results:
            return {}
        
        returns = [r['total_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        analysis = {
            'mean_return': np.mean(returns),
            'return_std': np.std(returns),
            'return_percentiles': {
                '5th': np.percentile(returns, 5),
                '25th': np.percentile(returns, 25),
                '50th': np.percentile(returns, 50),
                '75th': np.percentile(returns, 75),
                '95th': np.percentile(returns, 95)
            },
            'mean_sharpe': np.mean(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': max(max_drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'probability_positive_return': len([r for r in returns if r > 0]) / len(returns),
            'probability_sharpe_gt_1': len([s for s in sharpe_ratios if s > 1.0]) / len(sharpe_ratios),
            'value_at_risk_5': np.percentile(returns, 5),
            'expected_shortfall_5': np.mean([r for r in returns if r <= np.percentile(returns, 5)])
        }
        
        return analysis


class StrategyOptimizationFramework:
    """Comprehensive strategy optimization framework."""
    
    def __init__(self):
        self.genetic_optimizer = None
        self.monte_carlo_simulator = MonteCarloSimulator()
        self.walk_forward_analyzer = WalkForwardAnalyzer()
    
    async def optimize_strategy(self,
                              strategy_class: type,
                              parameter_ranges: List[ParameterRange],
                              market_data: List[MarketData],
                              optimization_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive strategy optimization."""
        config = optimization_config or {}
        
        # Backtesting configuration
        backtesting_config = BacktestingConfig(
            initial_capital=config.get('initial_capital', 100000.0),
            commission_rate=config.get('commission_rate', 0.001),
            slippage_rate=config.get('slippage_rate', 0.0005)
        )
        
        # Genetic algorithm optimization
        ga_config = config.get('genetic_algorithm', {})
        self.genetic_optimizer = GeneticAlgorithmOptimizer(
            population_size=ga_config.get('population_size', 50),
            generations=ga_config.get('generations', 100),
            mutation_rate=ga_config.get('mutation_rate', 0.1),
            crossover_rate=ga_config.get('crossover_rate', 0.8),
            elite_size=ga_config.get('elite_size', 5)
        )
        
        logger.info("Starting comprehensive strategy optimization")
        
        # Step 1: Genetic Algorithm Optimization
        ga_result = await self.genetic_optimizer.optimize(
            strategy_class, parameter_ranges, market_data, backtesting_config
        )
        
        best_parameters = ga_result.best_individual.parameters
        
        # Step 2: Walk-Forward Analysis on best parameters
        wf_config = config.get('walk_forward', {})
        wf_analyzer = WalkForwardAnalyzer(
            training_period_days=wf_config.get('training_period_days', 252),
            testing_period_days=wf_config.get('testing_period_days', 63),
            step_days=wf_config.get('step_days', 21)
        )
        
        wf_result = await wf_analyzer.run_walk_forward_analysis(
            strategy_class, best_parameters, market_data, backtesting_config
        )
        
        # Step 3: Monte Carlo Simulation on best parameters
        mc_config = config.get('monte_carlo', {})
        mc_result = await self.monte_carlo_simulator.run_simulation(
            strategy_class,
            best_parameters,
            market_data,
            backtesting_config,
            bootstrap_method=mc_config.get('bootstrap_method', 'block')
        )
        
        # Compile comprehensive results
        optimization_result = {
            'best_parameters': best_parameters,
            'genetic_algorithm': {
                'best_fitness': ga_result.best_individual.fitness,
                'optimization_time': ga_result.optimization_time,
                'total_evaluations': ga_result.total_evaluations,
                'convergence_generation': ga_result.convergence_generation,
                'optimization_metrics': ga_result.optimization_metrics
            },
            'walk_forward_analysis': wf_result,
            'monte_carlo_simulation': mc_result,
            'parameter_ranges': [
                {
                    'name': pr.name,
                    'min_value': pr.min_value,
                    'max_value': pr.max_value,
                    'parameter_type': pr.parameter_type
                }
                for pr in parameter_ranges
            ]
        }
        
        logger.info("Comprehensive strategy optimization completed",
                   best_fitness=ga_result.best_individual.fitness,
                   mc_mean_return=mc_result['analysis'].get('mean_return', 0),
                   wf_periods=wf_result.get('total_periods', 0))
        
        return optimization_result