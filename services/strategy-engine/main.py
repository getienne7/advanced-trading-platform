"""
Strategy Engine Service - Main Application
Provides backtesting, strategy optimization, and multi-timeframe analysis capabilities.
"""
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.base_service import BaseService
from shared.models import APIResponse
from backtesting_engine import (
    BacktestingEngine, BacktestingConfig, WalkForwardAnalyzer, 
    HistoricalDataReplay, SimpleMovingAverageStrategy, TradingStrategy,
    MarketData, PerformanceMetrics
)

# Configure logging
logger = structlog.get_logger("strategy-engine")

# Pydantic models for API
class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    strategy_name: str
    strategy_parameters: Dict[str, Any]
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "1h"
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    execution_model: str = "realistic"

class WalkForwardRequest(BaseModel):
    """Request model for walk-forward analysis."""
    strategy_name: str
    strategy_parameters: Dict[str, Any]
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "1h"
    training_period_days: int = 252
    testing_period_days: int = 63
    step_days: int = 21
    config: Optional[Dict[str, Any]] = None

class StrategyPerformanceResponse(BaseModel):
    """Response model for strategy performance."""
    strategy_name: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    total_trades: int
    final_capital: float
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]

class StrategyEngineService(BaseService):
    """Strategy Engine microservice."""
    
    def __init__(self):
        super().__init__("strategy-engine", "8007")
        self.backtesting_engine = None
        self.walk_forward_analyzer = WalkForwardAnalyzer()
        self.data_replay = HistoricalDataReplay()
        self.strategy_registry = {
            'simple_ma': SimpleMovingAverageStrategy
        }
        
    async def initialize(self):
        """Initialize the service."""
        await super().initialize()
        logger.info("Strategy Engine Service initialized")
    
    def get_strategy_class(self, strategy_name: str) -> type:
        """Get strategy class by name."""
        if strategy_name not in self.strategy_registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return self.strategy_registry[strategy_name]

# Create service instance
service = StrategyEngineService()
app = FastAPI(
    title="Strategy Engine Service",
    description="Advanced backtesting and strategy optimization service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await service.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "strategy-engine"}

@app.post("/backtest", response_model=APIResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest for a trading strategy."""
    try:
        logger.info("Starting backtest", 
                   strategy=request.strategy_name,
                   symbol=request.symbol,
                   start_date=request.start_date,
                   end_date=request.end_date)
        
        # Create backtesting configuration
        config = BacktestingConfig(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            execution_model=request.execution_model
        )
        
        # Get strategy class
        strategy_class = service.get_strategy_class(request.strategy_name)
        strategy = strategy_class(request.strategy_name, request.strategy_parameters)
        
        # Load historical data
        market_data = await service.data_replay.load_historical_data(
            request.symbol,
            request.start_date,
            request.end_date,
            request.timeframe
        )
        
        if not market_data:
            raise HTTPException(status_code=400, detail="No market data available for the specified period")
        
        # Run backtest
        backtesting_engine = BacktestingEngine(config)
        result = await backtesting_engine.run_backtest(strategy, market_data)
        
        logger.info("Backtest completed",
                   strategy=request.strategy_name,
                   total_return=result['performance_metrics'].total_return,
                   sharpe_ratio=result['performance_metrics'].sharpe_ratio,
                   total_trades=result['total_trades'])
        
        # Convert performance metrics to dict for JSON serialization
        performance_dict = {
            'total_return': result['performance_metrics'].total_return,
            'annualized_return': result['performance_metrics'].annualized_return,
            'volatility': result['performance_metrics'].volatility,
            'sharpe_ratio': result['performance_metrics'].sharpe_ratio,
            'sortino_ratio': result['performance_metrics'].sortino_ratio,
            'calmar_ratio': result['performance_metrics'].calmar_ratio,
            'max_drawdown': result['performance_metrics'].max_drawdown,
            'max_drawdown_duration': result['performance_metrics'].max_drawdown_duration,
            'win_rate': result['performance_metrics'].win_rate,
            'profit_factor': result['performance_metrics'].profit_factor,
            'total_trades': result['performance_metrics'].total_trades,
            'winning_trades': result['performance_metrics'].winning_trades,
            'losing_trades': result['performance_metrics'].losing_trades,
            'var_95': result['performance_metrics'].var_95,
            'expected_shortfall': result['performance_metrics'].expected_shortfall,
            'kelly_criterion': result['performance_metrics'].kelly_criterion
        }
        
        response_data = {
            'strategy_name': result['strategy_name'],
            'parameters': result['parameters'],
            'performance_metrics': performance_dict,
            'total_trades': result['total_trades'],
            'final_capital': result['final_capital'],
            'equity_curve': result['equity_curve'][:100],  # Limit for response size
            'trades': result['trades'][:50]  # Limit for response size
        }
        
        return APIResponse(
            success=True,
            message="Backtest completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.post("/walk-forward-analysis", response_model=APIResponse)
async def run_walk_forward_analysis(request: WalkForwardRequest):
    """Run walk-forward analysis for robust strategy testing."""
    try:
        logger.info("Starting walk-forward analysis",
                   strategy=request.strategy_name,
                   symbol=request.symbol,
                   training_days=request.training_period_days,
                   testing_days=request.testing_period_days)
        
        # Configure walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            training_period_days=request.training_period_days,
            testing_period_days=request.testing_period_days,
            step_days=request.step_days
        )
        
        # Get strategy class
        strategy_class = service.get_strategy_class(request.strategy_name)
        
        # Load historical data
        market_data = await service.data_replay.load_historical_data(
            request.symbol,
            request.start_date,
            request.end_date,
            request.timeframe
        )
        
        if not market_data:
            raise HTTPException(status_code=400, detail="No market data available for the specified period")
        
        # Create backtesting config
        config = BacktestingConfig(**(request.config or {}))
        
        # Run walk-forward analysis
        result = await analyzer.run_walk_forward_analysis(
            strategy_class,
            request.strategy_parameters,
            market_data,
            config
        )
        
        logger.info("Walk-forward analysis completed",
                   strategy=request.strategy_name,
                   total_periods=result['total_periods'],
                   mean_return=result['aggregated_metrics'].get('mean_return', 0))
        
        return APIResponse(
            success=True,
            message="Walk-forward analysis completed successfully",
            data=result
        )
        
    except Exception as e:
        logger.error("Walk-forward analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Walk-forward analysis failed: {str(e)}")

@app.get("/strategies")
async def list_strategies():
    """List available trading strategies."""
    strategies = []
    for name, strategy_class in service.strategy_registry.items():
        strategies.append({
            'name': name,
            'class_name': strategy_class.__name__,
            'description': strategy_class.__doc__ or "No description available"
        })
    
    return APIResponse(
        success=True,
        message="Available strategies retrieved",
        data=strategies
    )

@app.post("/validate-strategy")
async def validate_strategy(strategy_name: str, parameters: Dict[str, Any]):
    """Validate strategy parameters."""
    try:
        strategy_class = service.get_strategy_class(strategy_name)
        strategy = strategy_class(strategy_name, parameters)
        
        return APIResponse(
            success=True,
            message="Strategy parameters are valid",
            data={
                'strategy_name': strategy_name,
                'parameters': parameters,
                'validation': 'passed'
            }
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            message=f"Strategy validation failed: {str(e)}",
            data={
                'strategy_name': strategy_name,
                'parameters': parameters,
                'validation': 'failed',
                'error': str(e)
            }
        )

@app.get("/performance-metrics/{strategy_name}")
async def get_performance_metrics_info(strategy_name: str):
    """Get information about available performance metrics."""
    metrics_info = {
        'basic_metrics': [
            'total_return', 'annualized_return', 'volatility'
        ],
        'risk_metrics': [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
            'max_drawdown', 'var_95', 'expected_shortfall'
        ],
        'trade_metrics': [
            'win_rate', 'profit_factor', 'total_trades',
            'winning_trades', 'losing_trades', 'kelly_criterion'
        ],
        'advanced_metrics': [
            'ulcer_index', 'recovery_factor', 'consecutive_wins', 'consecutive_losses'
        ]
    }
    
    return APIResponse(
        success=True,
        message="Performance metrics information retrieved",
        data=metrics_info
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info"
    )