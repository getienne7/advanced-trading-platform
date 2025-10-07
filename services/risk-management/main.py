"""
Risk Management Service for Advanced Trading Platform.
Provides dynamic VaR calculation, portfolio risk monitoring, and risk controls.
"""
import asyncio
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from database import get_postgres_session, get_redis
from utils import setup_logging
from models import OrderSide, OrderType

from dynamic_risk_engine import (
    DynamicRiskEngine, RiskConfiguration, RiskModel, AssetClass,
    RiskMetrics, PortfolioRisk, create_dynamic_risk_engine
)

# Configure logging
logger = setup_logging("risk-management")

# Metrics
RISK_CALCULATIONS = Counter('risk_calculations_total', 'Total risk calculations', ['model', 'asset_class'])
VAR_CALCULATION_TIME = Histogram('var_calculation_seconds', 'VaR calculation time')
PORTFOLIO_VAR = Gauge('portfolio_var_95', 'Portfolio VaR at 95% confidence', ['portfolio_id'])
RISK_LIMIT_BREACHES = Counter('risk_limit_breaches_total', 'Risk limit breaches', ['limit_type'])

# FastAPI app
app = FastAPI(
    title="Risk Management Service",
    description="Advanced risk calculation and monitoring service",
    version="1.0.0"
)

# Global risk engine
risk_engine: Optional[DynamicRiskEngine] = None

# Request/Response Models

class VaRRequest(BaseModel):
    """Request for VaR calculation."""
    returns: List[float] = Field(..., description="Historical returns data")
    position_value: float = Field(..., description="Position value in USD")
    asset_class: AssetClass = Field(AssetClass.CRYPTOCURRENCY, description="Asset class")
    model: RiskModel = Field(RiskModel.HYBRID, description="Risk calculation model")
    symbol: Optional[str] = Field(None, description="Asset symbol")

class PortfolioRiskRequest(BaseModel):
    """Request for portfolio risk calculation."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    returns_data: Dict[str, List[float]] = Field(..., description="Returns data by symbol")
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")

class StressTestRequest(BaseModel):
    """Request for stress testing."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    scenarios: List[Dict[str, float]] = Field(..., description="Stress test scenarios")
    scenario_names: Optional[List[str]] = Field(None, description="Scenario names")

class LiquidityRiskRequest(BaseModel):
    """Request for liquidity risk assessment."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    daily_volumes: Dict[str, float] = Field(..., description="Daily trading volumes")
    liquidation_horizon_days: int = Field(5, description="Liquidation time horizon")

class RiskLimitsRequest(BaseModel):
    """Request for risk limits configuration."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    var_limit_95: float = Field(..., description="VaR limit at 95% confidence")
    var_limit_99: float = Field(..., description="VaR limit at 99% confidence")
    concentration_limit: float = Field(0.25, description="Maximum position concentration")
    correlation_limit: float = Field(0.7, description="Maximum correlation threshold")
    max_drawdown_limit: float = Field(0.2, description="Maximum drawdown limit")

# Risk Management Endpoints

@app.post("/risk/var/calculate")
async def calculate_var(request: VaRRequest):
    """Calculate Value at Risk for a position."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    try:
        with VAR_CALCULATION_TIME.time():
            returns_array = np.array(request.returns)
            
            risk_metrics = await risk_engine.calculate_var(
                returns=returns_array,
                position_value=request.position_value,
                asset_class=request.asset_class,
                model=request.model
            )
        
        # Record metrics
        RISK_CALCULATIONS.labels(
            model=request.model.value,
            asset_class=request.asset_class.value
        ).inc()
        
        logger.info("VaR calculated successfully",
                   symbol=request.symbol,
                   model=request.model.value,
                   var_95=risk_metrics.var_95,
                   var_99=risk_metrics.var_99)
        
        return {
            'success': True,
            'symbol': request.symbol,
            'model': request.model.value,
            'asset_class': request.asset_class.value,
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'expected_shortfall_95': risk_metrics.expected_shortfall_95,
                'expected_shortfall_99': risk_metrics.expected_shortfall_99,
                'volatility': risk_metrics.volatility,
                'skewness': risk_metrics.skewness,
                'kurtosis': risk_metrics.kurtosis,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'calmar_ratio': risk_metrics.calmar_ratio,
                'confidence_interval': risk_metrics.confidence_interval,
                'calculation_timestamp': risk_metrics.calculation_timestamp.isoformat()
            }
        }
    
    except Exception as e:
        logger.error("VaR calculation failed", error=str(e), symbol=request.symbol)
        raise HTTPException(status_code=500, detail=f"VaR calculation failed: {str(e)}")

@app.post("/risk/portfolio/calculate")
async def calculate_portfolio_risk(request: PortfolioRiskRequest):
    """Calculate comprehensive portfolio risk metrics."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    try:
        # Prepare returns matrix
        symbols = list(request.positions.keys())
        returns_data = []
        
        for symbol in symbols:
            if symbol not in request.returns_data:
                raise HTTPException(status_code=400, detail=f"Missing returns data for {symbol}")
            returns_data.append(request.returns_data[symbol])
        
        # Ensure all return series have the same length
        min_length = min(len(returns) for returns in returns_data)
        returns_matrix = np.array([returns[-min_length:] for returns in returns_data]).T
        
        portfolio_risk = await risk_engine.calculate_portfolio_risk(
            positions=request.positions,
            returns_matrix=returns_matrix,
            symbols=symbols
        )
        
        # Update metrics
        if request.portfolio_id:
            PORTFOLIO_VAR.labels(portfolio_id=request.portfolio_id).set(portfolio_risk.total_var_95)
        
        logger.info("Portfolio risk calculated successfully",
                   portfolio_id=request.portfolio_id,
                   total_var_95=portfolio_risk.total_var_95,
                   diversification_ratio=portfolio_risk.diversification_ratio)
        
        return {
            'success': True,
            'portfolio_id': request.portfolio_id,
            'portfolio_risk': {
                'total_var_95': portfolio_risk.total_var_95,
                'total_var_99': portfolio_risk.total_var_99,
                'diversification_ratio': portfolio_risk.diversification_ratio,
                'concentration_risk': portfolio_risk.concentration_risk,
                'correlation_matrix': portfolio_risk.correlation_matrix.tolist(),
                'component_var': portfolio_risk.component_var,
                'marginal_var': portfolio_risk.marginal_var,
                'incremental_var': portfolio_risk.incremental_var,
                'risk_budget': portfolio_risk.risk_budget,
                'active_risk': portfolio_risk.active_risk,
                'information_ratio': portfolio_risk.information_ratio,
                'portfolio_beta': portfolio_risk.portfolio_beta,
                'systematic_risk': portfolio_risk.systematic_risk,
                'idiosyncratic_risk': portfolio_risk.idiosyncratic_risk
            }
        }
    
    except Exception as e:
        logger.error("Portfolio risk calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Portfolio risk calculation failed: {str(e)}")

@app.post("/risk/stress-test")
async def run_stress_test(request: StressTestRequest):
    """Run stress tests on portfolio."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    try:
        stress_results = await risk_engine.stress_test_portfolio(
            positions=request.positions,
            stress_scenarios=request.scenarios
        )
        
        # Format results with scenario names if provided
        formatted_results = {}
        scenario_names = request.scenario_names or [f"Scenario {i+1}" for i in range(len(request.scenarios))]
        
        for i, (scenario_key, pnl) in enumerate(stress_results.items()):
            scenario_name = scenario_names[i] if i < len(scenario_names) else scenario_key
            formatted_results[scenario_name] = {
                'pnl': pnl,
                'pnl_percentage': pnl / sum(abs(v) for v in request.positions.values()) * 100
            }
        
        logger.info("Stress test completed successfully",
                   scenarios_count=len(request.scenarios),
                   worst_case_pnl=min(stress_results.values()))
        
        return {
            'success': True,
            'stress_test_results': formatted_results,
            'summary': {
                'worst_case_pnl': min(stress_results.values()),
                'best_case_pnl': max(stress_results.values()),
                'average_pnl': sum(stress_results.values()) / len(stress_results),
                'scenarios_tested': len(request.scenarios)
            }
        }
    
    except Exception as e:
        logger.error("Stress test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

@app.post("/risk/liquidity/assess")
async def assess_liquidity_risk(request: LiquidityRiskRequest):
    """Assess liquidity risk for portfolio positions."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    try:
        liquidity_metrics = await risk_engine.calculate_liquidity_risk(
            positions=request.positions,
            daily_volumes=request.daily_volumes,
            liquidation_horizon_days=request.liquidation_horizon_days
        )
        
        # Calculate portfolio-level liquidity metrics
        total_position_value = sum(abs(v) for v in request.positions.values())
        weighted_liquidity_score = sum(
            abs(request.positions[symbol]) / total_position_value * metrics['liquidity_risk_score']
            for symbol, metrics in liquidity_metrics.items()
        )
        
        max_liquidation_time = max(
            metrics['time_to_liquidate_days'] 
            for metrics in liquidity_metrics.values()
            if metrics['time_to_liquidate_days'] != float('inf')
        ) if any(m['time_to_liquidate_days'] != float('inf') for m in liquidity_metrics.values()) else float('inf')
        
        logger.info("Liquidity risk assessment completed",
                   weighted_liquidity_score=weighted_liquidity_score,
                   max_liquidation_time=max_liquidation_time)
        
        return {
            'success': True,
            'liquidity_metrics': liquidity_metrics,
            'portfolio_summary': {
                'weighted_liquidity_score': weighted_liquidity_score,
                'max_liquidation_time_days': max_liquidation_time,
                'total_position_value': total_position_value,
                'liquidity_risk_level': 'HIGH' if weighted_liquidity_score > 0.7 else 'MEDIUM' if weighted_liquidity_score > 0.3 else 'LOW'
            }
        }
    
    except Exception as e:
        logger.error("Liquidity risk assessment failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Liquidity risk assessment failed: {str(e)}")

@app.post("/risk/limits/check")
async def check_risk_limits(request: RiskLimitsRequest, portfolio_risk_data: Dict[str, Any]):
    """Check if portfolio breaches risk limits."""
    try:
        breaches = []
        
        # Check VaR limits
        if 'total_var_95' in portfolio_risk_data:
            if portfolio_risk_data['total_var_95'] > request.var_limit_95:
                breaches.append({
                    'limit_type': 'var_95',
                    'current_value': portfolio_risk_data['total_var_95'],
                    'limit': request.var_limit_95,
                    'breach_amount': portfolio_risk_data['total_var_95'] - request.var_limit_95
                })
                RISK_LIMIT_BREACHES.labels(limit_type='var_95').inc()
        
        if 'total_var_99' in portfolio_risk_data:
            if portfolio_risk_data['total_var_99'] > request.var_limit_99:
                breaches.append({
                    'limit_type': 'var_99',
                    'current_value': portfolio_risk_data['total_var_99'],
                    'limit': request.var_limit_99,
                    'breach_amount': portfolio_risk_data['total_var_99'] - request.var_limit_99
                })
                RISK_LIMIT_BREACHES.labels(limit_type='var_99').inc()
        
        # Check concentration limits
        if 'concentration_risk' in portfolio_risk_data:
            if portfolio_risk_data['concentration_risk'] > request.concentration_limit:
                breaches.append({
                    'limit_type': 'concentration',
                    'current_value': portfolio_risk_data['concentration_risk'],
                    'limit': request.concentration_limit,
                    'breach_amount': portfolio_risk_data['concentration_risk'] - request.concentration_limit
                })
                RISK_LIMIT_BREACHES.labels(limit_type='concentration').inc()
        
        # Check correlation limits (simplified - check max correlation)
        if 'correlation_matrix' in portfolio_risk_data:
            corr_matrix = np.array(portfolio_risk_data['correlation_matrix'])
            # Get upper triangle excluding diagonal
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            max_correlation = np.max(np.abs(upper_triangle)) if len(upper_triangle) > 0 else 0
            
            if max_correlation > request.correlation_limit:
                breaches.append({
                    'limit_type': 'correlation',
                    'current_value': max_correlation,
                    'limit': request.correlation_limit,
                    'breach_amount': max_correlation - request.correlation_limit
                })
                RISK_LIMIT_BREACHES.labels(limit_type='correlation').inc()
        
        has_breaches = len(breaches) > 0
        
        if has_breaches:
            logger.warning("Risk limit breaches detected",
                          portfolio_id=request.portfolio_id,
                          breaches_count=len(breaches))
        
        return {
            'success': True,
            'portfolio_id': request.portfolio_id,
            'has_breaches': has_breaches,
            'breaches': breaches,
            'risk_status': 'BREACH' if has_breaches else 'WITHIN_LIMITS',
            'check_timestamp': datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Risk limits check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Risk limits check failed: {str(e)}")

# Configuration and Management Endpoints

@app.post("/risk/config/update")
async def update_risk_configuration(config: RiskConfiguration):
    """Update risk engine configuration."""
    global risk_engine
    
    try:
        risk_engine = create_dynamic_risk_engine(config)
        
        logger.info("Risk engine configuration updated", config=config.dict())
        
        return {
            'success': True,
            'message': 'Risk engine configuration updated successfully',
            'config': config.dict()
        }
    
    except Exception as e:
        logger.error("Risk configuration update failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.get("/risk/config")
async def get_risk_configuration():
    """Get current risk engine configuration."""
    if not risk_engine:
        raise HTTPException(status_code=503, detail="Risk engine not initialized")
    
    return {
        'success': True,
        'config': risk_engine.config.dict()
    }

# Predefined stress test scenarios
@app.get("/risk/scenarios/predefined")
async def get_predefined_scenarios():
    """Get predefined stress test scenarios."""
    scenarios = {
        'market_crash': {
            'name': 'Market Crash (-30%)',
            'description': 'Severe market downturn affecting all assets',
            'scenario': {'BTC': -0.3, 'ETH': -0.35, 'ADA': -0.4, 'DOT': -0.38}
        },
        'crypto_winter': {
            'name': 'Crypto Winter (-50%)',
            'description': 'Extended bear market in cryptocurrencies',
            'scenario': {'BTC': -0.5, 'ETH': -0.55, 'ADA': -0.6, 'DOT': -0.58}
        },
        'flash_crash': {
            'name': 'Flash Crash (-20%)',
            'description': 'Sudden market crash with quick recovery',
            'scenario': {'BTC': -0.2, 'ETH': -0.22, 'ADA': -0.25, 'DOT': -0.23}
        },
        'altcoin_collapse': {
            'name': 'Altcoin Collapse',
            'description': 'Major altcoins lose value while BTC remains stable',
            'scenario': {'BTC': -0.05, 'ETH': -0.4, 'ADA': -0.6, 'DOT': -0.5}
        },
        'regulatory_shock': {
            'name': 'Regulatory Shock',
            'description': 'Negative regulatory news impacts all crypto',
            'scenario': {'BTC': -0.25, 'ETH': -0.3, 'ADA': -0.35, 'DOT': -0.32}
        }
    }
    
    return {
        'success': True,
        'scenarios': scenarios
    }

# Health and monitoring endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'service': 'risk-management',
        'timestamp': datetime.utcnow().isoformat(),
        'risk_engine_initialized': risk_engine is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global risk_engine, portfolio_optimizer, stress_engine
    
    try:
        # Initialize risk engine with default configuration
        config = RiskConfiguration()
        risk_engine = create_dynamic_risk_engine(config)
        
        # Initialize portfolio optimizer
        optimizer_config = PortfolioOptimizerConfig()
        portfolio_optimizer = create_portfolio_optimizer(optimizer_config)
        
        # Initialize stress testing engine
        stress_engine = create_stress_testing_engine()
        
        logger.info("Risk Management Service started successfully")
        
    except Exception as e:
        logger.error("Failed to start Risk Management Service", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Risk Management Service shutting down")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )

# Import portfolio optimizer
from portfolio_optimizer import (
    PortfolioOptimizer, PortfolioOptimizerConfig, OptimizationObjective,
    OptimizationConstraints, MarketView, OptimizationResult,
    create_portfolio_optimizer
)

# Import stress testing engine
from stress_testing_engine import (
    StressTestingEngine, StressTestType, ScenarioSeverity, StressScenario,
    StressTestResult, TailRiskMetrics, MonteCarloStressConfig,
    create_stress_testing_engine
)

# Global portfolio optimizer
portfolio_optimizer: Optional[PortfolioOptimizer] = None

# Global stress testing engine
stress_engine: Optional[StressTestingEngine] = None

# Portfolio Optimization Endpoints

@app.post("/portfolio/optimize")
async def optimize_portfolio(
    returns_data: Dict[str, List[float]],
    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
    current_weights: Optional[Dict[str, float]] = None,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4,
    market_views: Optional[List[Dict[str, Any]]] = None
):
    """Optimize portfolio allocation using specified objective."""
    if not portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")
    
    try:
        # Create constraints
        constraints = OptimizationConstraints(
            min_weights=min_weights or {},
            max_weights=max_weights or {},
            max_concentration=max_concentration
        )
        
        # Process market views if provided
        processed_views = []
        if market_views:
            for view_data in market_views:
                view = MarketView(
                    asset=view_data['asset'],
                    expected_return=view_data['expected_return'],
                    confidence=view_data['confidence'],
                    view_type=view_data.get('view_type', 'absolute'),
                    reference_asset=view_data.get('reference_asset')
                )
                processed_views.append(view)
        
        # Optimize portfolio
        result = await portfolio_optimizer.optimize_portfolio(
            returns_data=returns_data,
            current_weights=current_weights,
            objective=objective,
            constraints=constraints,
            market_views=processed_views if processed_views else None
        )
        
        logger.info("Portfolio optimization completed",
                   objective=objective.value,
                   expected_return=result.expected_return,
                   volatility=result.expected_volatility,
                   sharpe_ratio=result.sharpe_ratio)
        
        return {
            'success': True,
            'objective': objective.value,
            'optimization_result': {
                'optimal_weights': result.optimal_weights,
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'objective_value': result.objective_value,
                'optimization_status': result.optimization_status,
                'iterations': result.iterations,
                'computation_time': result.computation_time,
                'risk_contribution': result.risk_contribution,
                'diversification_ratio': result.diversification_ratio,
                'concentration_ratio': result.concentration_ratio,
                'timestamp': result.timestamp.isoformat()
            }
        }
    
    except Exception as e:
        logger.error("Portfolio optimization failed", error=str(e), objective=objective.value)
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

@app.post("/portfolio/optimize/max-sharpe")
async def optimize_max_sharpe(
    returns_data: Dict[str, List[float]],
    current_weights: Optional[Dict[str, float]] = None,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4
):
    """Optimize portfolio for maximum Sharpe ratio."""
    return await optimize_portfolio(
        returns_data=returns_data,
        objective=OptimizationObjective.MAX_SHARPE,
        current_weights=current_weights,
        min_weights=min_weights,
        max_weights=max_weights,
        max_concentration=max_concentration
    )

@app.post("/portfolio/optimize/min-variance")
async def optimize_min_variance(
    returns_data: Dict[str, List[float]],
    current_weights: Optional[Dict[str, float]] = None,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4
):
    """Optimize portfolio for minimum variance."""
    return await optimize_portfolio(
        returns_data=returns_data,
        objective=OptimizationObjective.MIN_VARIANCE,
        current_weights=current_weights,
        min_weights=min_weights,
        max_weights=max_weights,
        max_concentration=max_concentration
    )

@app.post("/portfolio/optimize/risk-parity")
async def optimize_risk_parity(
    returns_data: Dict[str, List[float]],
    current_weights: Optional[Dict[str, float]] = None,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4
):
    """Optimize portfolio using risk parity approach."""
    return await optimize_portfolio(
        returns_data=returns_data,
        objective=OptimizationObjective.RISK_PARITY,
        current_weights=current_weights,
        min_weights=min_weights,
        max_weights=max_weights,
        max_concentration=max_concentration
    )

@app.post("/portfolio/optimize/black-litterman")
async def optimize_black_litterman(
    returns_data: Dict[str, List[float]],
    market_views: List[Dict[str, Any]],
    current_weights: Optional[Dict[str, float]] = None,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4
):
    """Optimize portfolio using Black-Litterman model with market views."""
    return await optimize_portfolio(
        returns_data=returns_data,
        objective=OptimizationObjective.BLACK_LITTERMAN,
        current_weights=current_weights,
        min_weights=min_weights,
        max_weights=max_weights,
        max_concentration=max_concentration,
        market_views=market_views
    )

@app.post("/portfolio/rebalance")
async def calculate_rebalancing(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    rebalancing_threshold: float = 0.05,
    portfolio_value: float = 100000
):
    """Calculate portfolio rebalancing requirements."""
    if not portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")
    
    try:
        # Calculate rebalancing actions
        rebalancing_actions = {}
        total_deviation = 0.0
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            deviation = target_weight - current_weight
            
            if abs(deviation) > rebalancing_threshold:
                current_value = current_weight * portfolio_value
                target_value = target_weight * portfolio_value
                trade_amount = target_value - current_value
                
                rebalancing_actions[symbol] = {
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'deviation': deviation,
                    'current_value': current_value,
                    'target_value': target_value,
                    'trade_amount': trade_amount,
                    'action': 'buy' if trade_amount > 0 else 'sell',
                    'trade_size': abs(trade_amount)
                }
            
            total_deviation += abs(deviation)
        
        # Calculate transaction costs
        transaction_cost_bps = portfolio_optimizer.config.transaction_cost_bps
        transaction_costs = total_deviation * portfolio_value * transaction_cost_bps / 10000
        
        return {
            'success': True,
            'rebalancing_required': len(rebalancing_actions) > 0,
            'total_deviation': total_deviation,
            'transaction_costs': transaction_costs,
            'transaction_cost_percentage': transaction_costs / portfolio_value * 100,
            'rebalancing_actions': rebalancing_actions,
            'symbols_to_rebalance': len(rebalancing_actions),
            'estimated_trades': len(rebalancing_actions),
            'portfolio_value': portfolio_value,
            'rebalancing_threshold': rebalancing_threshold
        }
    
    except Exception as e:
        logger.error("Portfolio rebalancing calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Rebalancing calculation failed: {str(e)}")

# Stress Testing Endpoints

class HistoricalStressTestRequest(BaseModel):
    """Request for historical stress testing."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    scenario_names: Optional[List[str]] = Field(None, description="Specific scenarios to test")

class MonteCarloStressTestRequest(BaseModel):
    """Request for Monte Carlo stress testing."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    n_simulations: int = Field(10000, description="Number of simulations")
    confidence_levels: List[float] = Field([0.95, 0.99, 0.999], description="Confidence levels")

class TailRiskAnalysisRequest(BaseModel):
    """Request for tail risk analysis."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    returns_data: Dict[str, List[float]] = Field(..., description="Historical returns data")

class BlackSwanAnalysisRequest(BaseModel):
    """Request for black swan analysis."""
    positions: Dict[str, float] = Field(..., description="Positions by symbol")
    market_data: Dict[str, Any] = Field(..., description="Market data for analysis")

class CustomScenarioRequest(BaseModel):
    """Request to create custom scenario."""
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    asset_shocks: Dict[str, float] = Field(..., description="Asset shock percentages")
    severity: ScenarioSeverity = Field(ScenarioSeverity.MODERATE, description="Scenario severity")

@app.post("/stress-test/historical")
async def run_historical_stress_test(request: HistoricalStressTestRequest):
    """Run historical stress test scenarios."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        results = await stress_engine.run_historical_stress_test(
            positions=request.positions,
            scenario_names=request.scenario_names
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'scenario_name': result.scenario_name,
                'scenario_type': result.scenario_type.value,
                'total_pnl': result.total_pnl,
                'pnl_percentage': result.pnl_percentage,
                'asset_pnl': result.asset_pnl,
                'var_breach': result.var_breach,
                'liquidity_impact': result.liquidity_impact,
                'recovery_time_days': result.recovery_time_days,
                'risk_metrics': result.risk_metrics,
                'timestamp': result.timestamp.isoformat()
            })
        
        # Summary statistics
        total_pnls = [r.total_pnl for r in results]
        summary = {
            'worst_case_pnl': min(total_pnls) if total_pnls else 0,
            'best_case_pnl': max(total_pnls) if total_pnls else 0,
            'average_pnl': sum(total_pnls) / len(total_pnls) if total_pnls else 0,
            'scenarios_with_var_breach': sum(1 for r in results if r.var_breach),
            'total_scenarios_tested': len(results)
        }
        
        logger.info("Historical stress test completed",
                   scenarios_tested=len(results),
                   worst_case=summary['worst_case_pnl'])
        
        return {
            'success': True,
            'results': formatted_results,
            'summary': summary
        }
    
    except Exception as e:
        logger.error("Historical stress test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Historical stress test failed: {str(e)}")

@app.post("/stress-test/monte-carlo")
async def run_monte_carlo_stress_test(request: MonteCarloStressTestRequest):
    """Run Monte Carlo stress testing framework."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        config = MonteCarloStressConfig(
            n_simulations=request.n_simulations,
            confidence_levels=request.confidence_levels
        )
        
        results = await stress_engine.run_monte_carlo_stress_test(
            positions=request.positions,
            config=config
        )
        
        logger.info("Monte Carlo stress test completed",
                   simulations=request.n_simulations,
                   worst_case=results['worst_case_pnl'])
        
        return {
            'success': True,
            'monte_carlo_results': results
        }
    
    except Exception as e:
        logger.error("Monte Carlo stress test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Monte Carlo stress test failed: {str(e)}")

@app.post("/stress-test/tail-risk")
async def analyze_tail_risk(request: TailRiskAnalysisRequest):
    """Analyze tail risk and extreme event characteristics."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        tail_metrics = await stress_engine.analyze_tail_risk(
            positions=request.positions,
            returns_data=request.returns_data
        )
        
        logger.info("Tail risk analysis completed",
                   tail_expectation=tail_metrics.tail_expectation,
                   black_swan_threshold=tail_metrics.black_swan_threshold)
        
        return {
            'success': True,
            'tail_risk_metrics': {
                'tail_expectation': tail_metrics.tail_expectation,
                'tail_variance': tail_metrics.tail_variance,
                'extreme_loss_probability': tail_metrics.extreme_loss_probability,
                'black_swan_threshold': tail_metrics.black_swan_threshold,
                'maximum_credible_loss': tail_metrics.maximum_credible_loss,
                'tail_correlation': tail_metrics.tail_correlation,
                'fat_tail_indicator': tail_metrics.fat_tail_indicator,
                'tail_dependency': tail_metrics.tail_dependency
            }
        }
    
    except Exception as e:
        logger.error("Tail risk analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Tail risk analysis failed: {str(e)}")

@app.post("/stress-test/black-swan")
async def detect_black_swan_indicators(request: BlackSwanAnalysisRequest):
    """Detect potential black swan event indicators."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        indicators = await stress_engine.detect_black_swan_indicators(
            market_data=request.market_data,
            positions=request.positions
        )
        
        logger.info("Black swan analysis completed",
                   risk_score=indicators.get('overall_assessment', {}).get('risk_score', 0))
        
        return {
            'success': True,
            'black_swan_indicators': indicators
        }
    
    except Exception as e:
        logger.error("Black swan analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Black swan analysis failed: {str(e)}")

@app.post("/stress-test/scenarios/custom")
async def create_custom_scenario(request: CustomScenarioRequest):
    """Create a custom stress test scenario."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        scenario_id = await stress_engine.create_custom_scenario(
            name=request.name,
            description=request.description,
            asset_shocks=request.asset_shocks,
            severity=request.severity
        )
        
        logger.info("Custom scenario created", scenario_id=scenario_id)
        
        return {
            'success': True,
            'scenario_id': scenario_id,
            'message': 'Custom scenario created successfully'
        }
    
    except Exception as e:
        logger.error("Custom scenario creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Custom scenario creation failed: {str(e)}")

@app.get("/stress-test/scenarios")
async def get_available_scenarios():
    """Get all available stress test scenarios."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        scenarios = stress_engine.get_available_scenarios()
        
        return {
            'success': True,
            'scenarios': scenarios,
            'total_scenarios': len(scenarios)
        }
    
    except Exception as e:
        logger.error("Failed to get scenarios", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get scenarios: {str(e)}")

@app.post("/stress-test/recommendations")
async def get_scenario_recommendations(
    positions: Dict[str, float],
    risk_tolerance: str = "moderate"
):
    """Get recommendations for stress testing based on portfolio."""
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")
    
    try:
        recommendations = await stress_engine.get_scenario_recommendations(
            positions=positions,
            risk_tolerance=risk_tolerance
        )
        
        return {
            'success': True,
            'recommendations': recommendations
        }
    
    except Exception as e:
        logger.error("Scenario recommendations failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Scenario recommendations failed: {str(e)}")

@app.get("/portfolio/efficient-frontier")
async def calculate_efficient_frontier(
    returns_data: Dict[str, List[float]],
    n_portfolios: int = 50,
    min_weights: Optional[Dict[str, float]] = None,
    max_weights: Optional[Dict[str, float]] = None,
    max_concentration: float = 0.4
):
    """Calculate efficient frontier portfolios."""
    if not portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")
    
    try:
        # Create constraints
        constraints = OptimizationConstraints(
            min_weights=min_weights or {},
            max_weights=max_weights or {},
            max_concentration=max_concentration
        )
        
        # Calculate efficient frontier
        efficient_portfolios = await portfolio_optimizer.calculate_efficient_frontier(
            returns_data=returns_data,
            n_portfolios=n_portfolios,
            constraints=constraints
        )
        
        # Format results
        frontier_data = []
        for portfolio in efficient_portfolios:
            frontier_data.append({
                'expected_return': portfolio.expected_return,
                'expected_volatility': portfolio.expected_volatility,
                'sharpe_ratio': portfolio.sharpe_ratio,
                'weights': portfolio.optimal_weights,
                'diversification_ratio': portfolio.diversification_ratio,
                'concentration_ratio': portfolio.concentration_ratio
            })
        
        logger.info("Efficient frontier calculated",
                   n_portfolios=len(frontier_data),
                   target_portfolios=n_portfolios)
        
        return {
            'success': True,
            'efficient_frontier': frontier_data,
            'n_portfolios': len(frontier_data),
            'risk_free_rate': portfolio_optimizer.config.risk_free_rate
        }
    
    except Exception as e:
        logger.error("Efficient frontier calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Efficient frontier calculation failed: {str(e)}")

@app.post("/portfolio/config/update")
async def update_portfolio_optimizer_config(config: PortfolioOptimizerConfig):
    """Update portfolio optimizer configuration."""
    global portfolio_optimizer
    
    try:
        portfolio_optimizer = create_portfolio_optimizer(config)
        
        logger.info("Portfolio optimizer configuration updated", config=config.dict())
        
        return {
            'success': True,
            'message': 'Portfolio optimizer configuration updated successfully',
            'config': config.dict()
        }
    
    except Exception as e:
        logger.error("Portfolio optimizer configuration update failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.get("/portfolio/config")
async def get_portfolio_optimizer_config():
    """Get current portfolio optimizer configuration."""
    if not portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")
    
    return {
        'success': True,
        'config': portfolio_optimizer.config.dict()
    }

# Update startup event to initialize portfolio optimizer
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global risk_engine, portfolio_optimizer
    
    try:
        # Initialize risk engine
        risk_config = RiskConfiguration()
        risk_engine = create_dynamic_risk_engine(risk_config)
        
        # Initialize portfolio optimizer
        optimizer_config = PortfolioOptimizerConfig()
        portfolio_optimizer = create_portfolio_optimizer(optimizer_config)
        
        logger.info("Risk Management Service started successfully with portfolio optimization")
        
    except Exception as e:
        logger.error("Failed to start Risk Management Service", error=str(e))
        raise

# Update health check to include portfolio optimizer
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including portfolio optimizer."""
    return {
        'status': 'healthy',
        'service': 'risk-management',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'risk_engine': 'healthy' if risk_engine else 'not_initialized',
            'portfolio_optimizer': 'healthy' if portfolio_optimizer else 'not_initialized'
        }
    }# Im
port correlation and concentration monitor
from correlation_monitor import (
    CorrelationConcentrationMonitor, CorrelationMonitorConfig, CorrelationMethod,
    ConcentrationMetric, AlertSeverity, CorrelationMatrix, ConcentrationMetrics,
    PortfolioHeatMap, create_correlation_concentration_monitor
)

# Global correlation monitor
correlation_monitor: Optional[CorrelationConcentrationMonitor] = None

# Correlation and Concentration Monitoring Endpoints

@app.post("/correlation/calculate")
async def calculate_correlation_matrix(
    returns_data: Dict[str, List[float]],
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    lookback_days: Optional[int] = None
):
    """Calculate correlation matrix using specified method."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=returns_data,
            method=method,
            lookback_days=lookback_days
        )
        
        logger.info("Correlation matrix calculated",
                   method=method.value,
                   n_assets=len(correlation_matrix.symbols),
                   avg_correlation=float(np.mean(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)])))
        
        return {
            'success': True,
            'correlation_matrix': {
                'matrix': correlation_matrix.matrix.tolist(),
                'symbols': correlation_matrix.symbols,
                'method': correlation_matrix.method.value,
                'calculation_timestamp': correlation_matrix.calculation_timestamp.isoformat(),
                'lookback_days': correlation_matrix.lookback_days,
                'eigenvalues': correlation_matrix.eigenvalues.tolist() if correlation_matrix.eigenvalues is not None else None,
                'condition_number': correlation_matrix.condition_number,
                'average_correlation': float(np.mean(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)])),
                'max_correlation': float(np.max(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)])),
                'min_correlation': float(np.min(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)]))
            }
        }
    
    except Exception as e:
        logger.error("Correlation matrix calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Correlation calculation failed: {str(e)}")

@app.post("/concentration/calculate")
async def calculate_concentration_metrics(
    portfolio_weights: Dict[str, float],
    sector_mapping: Optional[Dict[str, str]] = None
):
    """Calculate comprehensive concentration risk metrics."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        concentration_metrics = await correlation_monitor.calculate_concentration_metrics(
            portfolio_weights=portfolio_weights,
            sector_mapping=sector_mapping
        )
        
        logger.info("Concentration metrics calculated",
                   herfindahl_index=concentration_metrics.herfindahl_index,
                   max_weight=concentration_metrics.max_weight,
                   max_weight_asset=concentration_metrics.max_weight_asset,
                   alerts_count=len(concentration_metrics.concentration_alerts))
        
        return {
            'success': True,
            'concentration_metrics': {
                'herfindahl_index': concentration_metrics.herfindahl_index,
                'entropy_measure': concentration_metrics.entropy_measure,
                'max_weight': concentration_metrics.max_weight,
                'max_weight_asset': concentration_metrics.max_weight_asset,
                'top_3_concentration': concentration_metrics.top_3_concentration,
                'top_5_concentration': concentration_metrics.top_5_concentration,
                'effective_number_assets': concentration_metrics.effective_number_assets,
                'diversification_ratio': concentration_metrics.diversification_ratio,
                'calculation_timestamp': concentration_metrics.calculation_timestamp.isoformat(),
                'concentration_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type,
                        'severity': alert.severity.value,
                        'asset': alert.asset,
                        'concentration_value': alert.concentration_value,
                        'threshold': alert.threshold,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in concentration_metrics.concentration_alerts
                ]
            }
        }
    
    except Exception as e:
        logger.error("Concentration metrics calculation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Concentration calculation failed: {str(e)}")

@app.post("/correlation/monitor")
async def monitor_correlations(
    returns_data: Dict[str, List[float]],
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    lookback_days: Optional[int] = None
):
    """Monitor correlations and generate alerts for threshold breaches."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        # First calculate correlation matrix
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=returns_data,
            method=method,
            lookback_days=lookback_days
        )
        
        # Monitor for alerts
        correlation_alerts = await correlation_monitor.monitor_correlations(correlation_matrix)
        
        logger.info("Correlation monitoring completed",
                   total_alerts=len(correlation_alerts),
                   critical_alerts=len([a for a in correlation_alerts if a.severity == AlertSeverity.CRITICAL]),
                   high_alerts=len([a for a in correlation_alerts if a.severity == AlertSeverity.HIGH]))
        
        return {
            'success': True,
            'monitoring_results': {
                'correlation_matrix': {
                    'average_correlation': float(np.mean(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)])),
                    'max_correlation': float(np.max(correlation_matrix.matrix[np.triu_indices_from(correlation_matrix.matrix, k=1)])),
                    'condition_number': correlation_matrix.condition_number
                },
                'alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type,
                        'severity': alert.severity.value,
                        'asset_pair': alert.asset_pair,
                        'correlation_value': alert.correlation_value,
                        'threshold': alert.threshold,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'metadata': alert.metadata
                    }
                    for alert in correlation_alerts
                ],
                'alert_summary': {
                    'total_alerts': len(correlation_alerts),
                    'critical_alerts': len([a for a in correlation_alerts if a.severity == AlertSeverity.CRITICAL]),
                    'high_alerts': len([a for a in correlation_alerts if a.severity == AlertSeverity.HIGH]),
                    'medium_alerts': len([a for a in correlation_alerts if a.severity == AlertSeverity.MEDIUM])
                }
            }
        }
    
    except Exception as e:
        logger.error("Correlation monitoring failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Correlation monitoring failed: {str(e)}")

@app.post("/portfolio/heatmap")
async def generate_portfolio_heatmap(
    returns_data: Dict[str, List[float]],
    portfolio_weights: Dict[str, float],
    risk_contributions: Optional[Dict[str, float]] = None,
    method: CorrelationMethod = CorrelationMethod.PEARSON
):
    """Generate comprehensive portfolio heat map visualizations."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        # Calculate correlation matrix
        correlation_matrix = await correlation_monitor.calculate_correlation_matrix(
            returns_data=returns_data,
            method=method
        )
        
        # Generate heatmap
        heatmap = await correlation_monitor.generate_portfolio_heatmap(
            correlation_matrix=correlation_matrix,
            portfolio_weights=portfolio_weights,
            risk_contributions=risk_contributions
        )
        
        logger.info("Portfolio heatmap generated",
                   n_assets=len(portfolio_weights),
                   has_risk_contributions=risk_contributions is not None)
        
        return {
            'success': True,
            'heatmap': {
                'correlation_heatmap': heatmap.correlation_heatmap,
                'concentration_heatmap': heatmap.concentration_heatmap,
                'risk_contribution_heatmap': heatmap.risk_contribution_heatmap,
                'timestamp': heatmap.timestamp.isoformat()
            }
        }
    
    except Exception as e:
        logger.error("Portfolio heatmap generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")

@app.post("/correlation/regime-analysis")
async def analyze_correlation_regimes(
    returns_data: Dict[str, List[float]],
    window_size: int = 60
):
    """Detect correlation regime changes using rolling correlations."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        regime_analysis = await correlation_monitor.detect_regime_changes(
            returns_data=returns_data,
            window_size=window_size
        )
        
        logger.info("Correlation regime analysis completed",
                   current_regime=regime_analysis['current_regime'],
                   average_correlation=regime_analysis['average_correlation'],
                   regime_changes=len(regime_analysis['regime_changes']))
        
        return {
            'success': True,
            'regime_analysis': {
                'current_regime': regime_analysis['current_regime'],
                'average_correlation': regime_analysis['average_correlation'],
                'correlation_volatility': regime_analysis['correlation_volatility'],
                'regime_changes': regime_analysis['regime_changes'],
                'rolling_correlations': regime_analysis['rolling_correlations'][-50:],  # Last 50 observations
                'analysis_window_size': window_size
            }
        }
    
    except Exception as e:
        logger.error("Correlation regime analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Regime analysis failed: {str(e)}")

@app.get("/correlation/alerts/active")
async def get_active_correlation_alerts():
    """Get all active correlation and concentration alerts."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    try:
        active_alerts = correlation_monitor.active_alerts
        
        # Format alerts
        formatted_alerts = []
        for alert_id, alert in active_alerts.items():
            if hasattr(alert, 'asset_pair'):  # Correlation alert
                formatted_alerts.append({
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'asset_pair': alert.asset_pair,
                    'correlation_value': alert.correlation_value,
                    'threshold': alert.threshold,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'category': 'correlation'
                })
            else:  # Concentration alert
                formatted_alerts.append({
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity.value,
                    'asset': alert.asset,
                    'concentration_value': alert.concentration_value,
                    'threshold': alert.threshold,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'category': 'concentration'
                })
        
        # Sort by severity and timestamp
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        formatted_alerts.sort(key=lambda x: (severity_order.get(x['severity'], 4), x['timestamp']), reverse=True)
        
        return {
            'success': True,
            'active_alerts': formatted_alerts,
            'alert_summary': {
                'total_alerts': len(formatted_alerts),
                'critical_alerts': len([a for a in formatted_alerts if a['severity'] == 'critical']),
                'high_alerts': len([a for a in formatted_alerts if a['severity'] == 'high']),
                'medium_alerts': len([a for a in formatted_alerts if a['severity'] == 'medium']),
                'correlation_alerts': len([a for a in formatted_alerts if a['category'] == 'correlation']),
                'concentration_alerts': len([a for a in formatted_alerts if a['category'] == 'concentration'])
            }
        }
    
    except Exception as e:
        logger.error("Failed to get active alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.post("/correlation/config/update")
async def update_correlation_monitor_config(config: CorrelationMonitorConfig):
    """Update correlation monitor configuration."""
    global correlation_monitor
    
    try:
        correlation_monitor = create_correlation_concentration_monitor(config)
        
        logger.info("Correlation monitor configuration updated", config=config.dict())
        
        return {
            'success': True,
            'message': 'Correlation monitor configuration updated successfully',
            'config': config.dict()
        }
    
    except Exception as e:
        logger.error("Correlation monitor configuration update failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.get("/correlation/config")
async def get_correlation_monitor_config():
    """Get current correlation monitor configuration."""
    if not correlation_monitor:
        raise HTTPException(status_code=503, detail="Correlation monitor not initialized")
    
    return {
        'success': True,
        'config': correlation_monitor.config.dict()
    }

# Enhanced startup event to initialize correlation monitor
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global risk_engine, portfolio_optimizer, correlation_monitor
    
    try:
        # Initialize risk engine
        risk_config = RiskConfiguration()
        risk_engine = create_dynamic_risk_engine(risk_config)
        
        # Initialize portfolio optimizer
        optimizer_config = PortfolioOptimizerConfig()
        portfolio_optimizer = create_portfolio_optimizer(optimizer_config)
        
        # Initialize correlation monitor
        correlation_config = CorrelationMonitorConfig()
        correlation_monitor = create_correlation_concentration_monitor(correlation_config)
        
        logger.info("Risk Management Service started successfully with all components")
        
    except Exception as e:
        logger.error("Failed to start Risk Management Service", error=str(e))
        raise

# Enhanced health check to include correlation monitor
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including all components."""
    return {
        'status': 'healthy',
        'service': 'risk-management',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'risk_engine': 'healthy' if risk_engine else 'not_initialized',
            'portfolio_optimizer': 'healthy' if portfolio_optimizer else 'not_initialized',
            'correlation_monitor': 'healthy' if correlation_monitor else 'not_initialized'
        },
        'active_alerts': len(correlation_monitor.active_alerts) if correlation_monitor else 0
    }