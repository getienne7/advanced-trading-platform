"""
Advanced Stress Testing and Scenario Analysis Engine
Implements historical stress testing scenarios, Monte Carlo stress testing framework,
and tail risk and black swan protection.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import structlog
from pydantic import BaseModel, Field

# Configure logging
logger = structlog.get_logger("stress-testing-engine")


class StressTestType(str, Enum):
    """Types of stress tests."""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    TAIL_RISK = "tail_risk"
    BLACK_SWAN = "black_swan"
    CUSTOM_SCENARIO = "custom_scenario"
    REGULATORY = "regulatory"


class ScenarioSeverity(str, Enum):
    """Severity levels for stress scenarios."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    description: str
    scenario_type: StressTestType
    severity: ScenarioSeverity
    asset_shocks: Dict[str, float]  # Asset -> shock percentage
    correlation_changes: Optional[Dict[Tuple[str, str], float]] = None
    volatility_multipliers: Optional[Dict[str, float]] = None
    duration_days: int = 1
    probability: Optional[float] = None
    historical_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Results from stress testing."""
    scenario_name: str
    scenario_type: StressTestType
    total_pnl: float
    pnl_percentage: float
    asset_pnl: Dict[str, float]
    var_breach: bool
    liquidity_impact: float
    recovery_time_days: Optional[int] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TailRiskMetrics:
    """Tail risk and extreme event metrics."""
    tail_expectation: float  # Expected loss in tail
    tail_variance: float
    extreme_loss_probability: float
    black_swan_threshold: float
    maximum_credible_loss: float
    tail_correlation: Dict[str, float]
    fat_tail_indicator: float
    tail_dependency: float


@dataclass
class MonteCarloStressConfig:
    """Configuration for Monte Carlo stress testing."""
    n_simulations: int = 10000
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    correlation_shock_range: Tuple[float, float] = (-0.5, 0.5)
    volatility_shock_range: Tuple[float, float] = (0.5, 3.0)
    return_shock_range: Tuple[float, float] = (-0.5, 0.5)
    fat_tail_probability: float = 0.05
    extreme_event_multiplier: float = 5.0


class StressTestingEngine:
    """Advanced stress testing and scenario analysis engine."""
    
    def __init__(self):
        self.historical_scenarios = {}
        self.custom_scenarios = {}
        self.tail_risk_models = {}
        self.black_swan_indicators = {}
        
        # Load predefined scenarios
        self._initialize_historical_scenarios()
        self._initialize_regulatory_scenarios()
        
        logger.info("Stress Testing Engine initialized")
    
    def _initialize_historical_scenarios(self):
        """Initialize historical stress test scenarios."""
        self.historical_scenarios = {
            "covid_crash_2020": StressScenario(
                name="COVID-19 Market Crash (March 2020)",
                description="Global market crash due to COVID-19 pandemic",
                scenario_type=StressTestType.HISTORICAL,
                severity=ScenarioSeverity.EXTREME,
                asset_shocks={
                    "BTC": -0.50,  # Bitcoin dropped ~50%
                    "ETH": -0.60,  # Ethereum dropped ~60%
                    "ADA": -0.65,  # Altcoins dropped more
                    "DOT": -0.70,
                    "LINK": -0.68,
                    "UNI": -0.75,
                    "STOCKS": -0.35,  # S&P 500 dropped ~35%
                    "BONDS": 0.05,   # Flight to safety
                    "GOLD": -0.12,   # Even gold dropped initially
                    "USD": 0.08      # USD strengthened
                },
                correlation_changes={
                    ("BTC", "ETH"): 0.15,  # Correlations increased during crisis
                    ("BTC", "STOCKS"): 0.25,
                    ("ETH", "STOCKS"): 0.30
                },
                volatility_multipliers={
                    "BTC": 3.0,
                    "ETH": 3.5,
                    "ADA": 4.0,
                    "STOCKS": 2.5
                },
                duration_days=30,
                probability=0.01,  # 1% annual probability
                historical_date=datetime(2020, 3, 12),
                metadata={
                    "trigger": "pandemic",
                    "recovery_months": 6,
                    "central_bank_response": "aggressive_easing"
                }
            ),
            
            "crypto_winter_2018": StressScenario(
                name="Crypto Winter (2018)",
                description="Extended bear market in cryptocurrencies",
                scenario_type=StressTestType.HISTORICAL,
                severity=ScenarioSeverity.SEVERE,
                asset_shocks={
                    "BTC": -0.84,  # Bitcoin dropped 84% from peak
                    "ETH": -0.94,  # Ethereum dropped 94%
                    "ADA": -0.97,  # Altcoins dropped even more
                    "DOT": -0.95,
                    "LINK": -0.96,
                    "STOCKS": -0.06,  # Traditional markets less affected
                    "BONDS": 0.02,
                    "GOLD": -0.02
                },
                duration_days=365,  # Extended period
                probability=0.05,   # 5% annual probability for crypto
                historical_date=datetime(2018, 1, 1),
                metadata={
                    "trigger": "regulatory_uncertainty",
                    "recovery_months": 24,
                    "sector_specific": True
                }
            ),
            
            "flash_crash_2010": StressScenario(
                name="Flash Crash (May 2010)",
                description="Rapid market crash and recovery within minutes",
                scenario_type=StressTestType.HISTORICAL,
                severity=ScenarioSeverity.MODERATE,
                asset_shocks={
                    "STOCKS": -0.09,  # 9% drop in minutes
                    "BTC": -0.15,     # Crypto would likely be more volatile
                    "ETH": -0.18,
                    "FUTURES": -0.12
                },
                duration_days=1,  # Intraday event
                probability=0.02,
                historical_date=datetime(2010, 5, 6),
                metadata={
                    "trigger": "algorithmic_trading",
                    "recovery_hours": 2,
                    "liquidity_crisis": True
                }
            ),
            
            "lehman_crisis_2008": StressScenario(
                name="Lehman Brothers Collapse (2008)",
                description="Global financial crisis triggered by Lehman collapse",
                scenario_type=StressTestType.HISTORICAL,
                severity=ScenarioSeverity.EXTREME,
                asset_shocks={
                    "STOCKS": -0.48,  # S&P 500 peak-to-trough
                    "BONDS": 0.20,    # Flight to quality
                    "GOLD": 0.25,     # Safe haven demand
                    "USD": 0.15,      # Dollar strength
                    "BTC": -0.60,     # Crypto didn't exist, but would likely crash
                    "ETH": -0.70,
                    "REAL_ESTATE": -0.33
                },
                duration_days=547,  # 18 months
                probability=0.008,  # Once in 125 years
                historical_date=datetime(2008, 9, 15),
                metadata={
                    "trigger": "financial_system_collapse",
                    "recovery_months": 60,
                    "systemic_risk": True
                }
            ),
            
            "china_devaluation_2015": StressScenario(
                name="China Yuan Devaluation (2015)",
                description="Surprise yuan devaluation causing global market turmoil",
                scenario_type=StressTestType.HISTORICAL,
                severity=ScenarioSeverity.MODERATE,
                asset_shocks={
                    "STOCKS": -0.11,
                    "EMERGING_MARKETS": -0.18,
                    "COMMODITIES": -0.15,
                    "USD": 0.08,
                    "BTC": -0.20,  # Would likely follow risk-off sentiment
                    "ETH": -0.25
                },
                duration_days=14,
                probability=0.03,
                historical_date=datetime(2015, 8, 11),
                metadata={
                    "trigger": "currency_devaluation",
                    "recovery_weeks": 8,
                    "contagion_effect": True
                }
            )
        }
    
    def _initialize_regulatory_scenarios(self):
        """Initialize regulatory stress scenarios."""
        regulatory_scenarios = {
            "crypto_ban": StressScenario(
                name="Major Jurisdiction Crypto Ban",
                description="Major country bans cryptocurrency trading",
                scenario_type=StressTestType.REGULATORY,
                severity=ScenarioSeverity.SEVERE,
                asset_shocks={
                    "BTC": -0.40,
                    "ETH": -0.50,
                    "ADA": -0.60,
                    "DOT": -0.55,
                    "DEFI_TOKENS": -0.70
                },
                duration_days=90,
                probability=0.15,
                metadata={
                    "regulatory_type": "trading_ban",
                    "jurisdiction": "major_economy"
                }
            ),
            
            "stablecoin_regulation": StressScenario(
                name="Strict Stablecoin Regulation",
                description="New regulations severely restrict stablecoin usage",
                scenario_type=StressTestType.REGULATORY,
                severity=ScenarioSeverity.MODERATE,
                asset_shocks={
                    "USDT": -0.05,
                    "USDC": -0.03,
                    "BTC": -0.15,  # Indirect impact
                    "ETH": -0.18,
                    "DEFI_TOKENS": -0.35  # Direct impact on DeFi
                },
                duration_days=60,
                probability=0.25,
                metadata={
                    "regulatory_type": "stablecoin_restrictions"
                }
            )
        }
        
        self.historical_scenarios.update(regulatory_scenarios)
    
    async def run_historical_stress_test(self, 
                                       positions: Dict[str, float],
                                       scenario_names: Optional[List[str]] = None) -> List[StressTestResult]:
        """Run historical stress test scenarios."""
        try:
            results = []
            scenarios_to_test = scenario_names or list(self.historical_scenarios.keys())
            
            for scenario_name in scenarios_to_test:
                if scenario_name not in self.historical_scenarios:
                    logger.warning(f"Unknown scenario: {scenario_name}")
                    continue
                
                scenario = self.historical_scenarios[scenario_name]
                result = await self._execute_stress_scenario(positions, scenario)
                results.append(result)
            
            logger.info(f"Historical stress testing completed for {len(results)} scenarios")
            return results
            
        except Exception as e:
            logger.error(f"Historical stress testing failed: {e}")
            raise
    
    async def run_monte_carlo_stress_test(self,
                                        positions: Dict[str, float],
                                        config: MonteCarloStressConfig = None) -> Dict[str, Any]:
        """Run Monte Carlo stress testing framework."""
        try:
            if config is None:
                config = MonteCarloStressConfig()
            
            symbols = list(positions.keys())
            n_assets = len(symbols)
            
            # Generate random stress scenarios
            stress_results = []
            
            for i in range(config.n_simulations):
                # Generate random shocks
                scenario_shocks = {}
                
                for symbol in symbols:
                    # Base shock from normal distribution
                    base_shock = np.random.normal(0, 0.1)
                    
                    # Add fat tail events
                    if np.random.random() < config.fat_tail_probability:
                        # Extreme event
                        extreme_shock = np.random.normal(0, 0.3) * config.extreme_event_multiplier
                        base_shock += extreme_shock
                    
                    # Apply shock range limits
                    shock = np.clip(base_shock, 
                                  config.return_shock_range[0], 
                                  config.return_shock_range[1])
                    scenario_shocks[symbol] = shock
                
                # Create temporary scenario
                temp_scenario = StressScenario(
                    name=f"Monte Carlo Scenario {i+1}",
                    description=f"Random stress scenario {i+1}",
                    scenario_type=StressTestType.MONTE_CARLO,
                    severity=ScenarioSeverity.MODERATE,
                    asset_shocks=scenario_shocks
                )
                
                # Execute scenario
                result = await self._execute_stress_scenario(positions, temp_scenario)
                stress_results.append(result.total_pnl)
            
            # Analyze results
            stress_array = np.array(stress_results)
            
            # Calculate percentiles
            percentiles = {}
            for confidence in config.confidence_levels:
                percentile = (1 - confidence) * 100
                percentiles[f"VaR_{confidence*100:.1f}"] = -np.percentile(stress_array, percentile)
            
            # Calculate tail metrics
            tail_5pct = stress_array[stress_array <= np.percentile(stress_array, 5)]
            tail_1pct = stress_array[stress_array <= np.percentile(stress_array, 1)]
            
            analysis = {
                "total_simulations": config.n_simulations,
                "worst_case_pnl": np.min(stress_array),
                "best_case_pnl": np.max(stress_array),
                "mean_pnl": np.mean(stress_array),
                "std_pnl": np.std(stress_array),
                "percentiles": percentiles,
                "tail_expectations": {
                    "tail_5pct_mean": np.mean(tail_5pct),
                    "tail_1pct_mean": np.mean(tail_1pct),
                    "tail_0_1pct_mean": np.mean(stress_array[stress_array <= np.percentile(stress_array, 0.1)])
                },
                "extreme_loss_count": np.sum(stress_array < -np.std(stress_array) * 3),
                "probability_of_ruin": np.sum(stress_array < -sum(abs(v) for v in positions.values()) * 0.5) / config.n_simulations
            }
            
            logger.info(f"Monte Carlo stress testing completed: {config.n_simulations} simulations")
            return analysis
            
        except Exception as e:
            logger.error(f"Monte Carlo stress testing failed: {e}")
            raise
    
    async def analyze_tail_risk(self, 
                              positions: Dict[str, float],
                              returns_data: Dict[str, List[float]]) -> TailRiskMetrics:
        """Analyze tail risk and extreme event characteristics."""
        try:
            # Combine all returns for portfolio analysis
            symbols = list(positions.keys())
            total_value = sum(abs(v) for v in positions.values())
            weights = {symbol: positions[symbol] / total_value for symbol in symbols}
            
            # Calculate portfolio returns
            min_length = min(len(returns_data[symbol]) for symbol in symbols)
            portfolio_returns = []
            
            for i in range(min_length):
                portfolio_return = sum(weights[symbol] * returns_data[symbol][i] for symbol in symbols)
                portfolio_returns.append(portfolio_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Tail analysis
            tail_threshold_5pct = np.percentile(portfolio_returns, 5)
            tail_threshold_1pct = np.percentile(portfolio_returns, 1)
            
            tail_returns_5pct = portfolio_returns[portfolio_returns <= tail_threshold_5pct]
            tail_returns_1pct = portfolio_returns[portfolio_returns <= tail_threshold_1pct]
            
            # Tail expectation and variance
            tail_expectation = np.mean(tail_returns_5pct) * total_value
            tail_variance = np.var(tail_returns_5pct) * (total_value ** 2)
            
            # Extreme loss probability (beyond 3 standard deviations)
            std_dev = np.std(portfolio_returns)
            extreme_threshold = np.mean(portfolio_returns) - 3 * std_dev
            extreme_loss_probability = np.sum(portfolio_returns < extreme_threshold) / len(portfolio_returns)
            
            # Black swan threshold (6 sigma event)
            black_swan_threshold = (np.mean(portfolio_returns) - 6 * std_dev) * total_value
            
            # Maximum credible loss (worst historical + buffer)
            historical_worst = np.min(portfolio_returns) * total_value
            maximum_credible_loss = historical_worst * 1.5  # 50% buffer
            
            # Fat tail indicator (kurtosis)
            fat_tail_indicator = stats.kurtosis(portfolio_returns)
            
            # Tail correlations (how assets correlate in tail events)
            tail_correlations = {}
            for symbol in symbols:
                symbol_returns = np.array(returns_data[symbol][-min_length:])
                # Correlation during tail events
                tail_mask = portfolio_returns <= tail_threshold_5pct
                if np.sum(tail_mask) > 10:  # Need sufficient tail observations
                    tail_corr = np.corrcoef(portfolio_returns[tail_mask], symbol_returns[tail_mask])[0, 1]
                    tail_correlations[symbol] = tail_corr if not np.isnan(tail_corr) else 0.0
                else:
                    tail_correlations[symbol] = 0.0
            
            # Tail dependency (copula-based measure, simplified)
            tail_dependency = np.mean([abs(corr) for corr in tail_correlations.values()])
            
            return TailRiskMetrics(
                tail_expectation=tail_expectation,
                tail_variance=tail_variance,
                extreme_loss_probability=extreme_loss_probability,
                black_swan_threshold=black_swan_threshold,
                maximum_credible_loss=maximum_credible_loss,
                tail_correlation=tail_correlations,
                fat_tail_indicator=fat_tail_indicator,
                tail_dependency=tail_dependency
            )
            
        except Exception as e:
            logger.error(f"Tail risk analysis failed: {e}")
            raise
    
    async def detect_black_swan_indicators(self,
                                         market_data: Dict[str, Any],
                                         positions: Dict[str, float]) -> Dict[str, Any]:
        """Detect potential black swan event indicators."""
        try:
            indicators = {}
            
            # Volatility clustering
            if 'volatility_data' in market_data:
                vol_data = market_data['volatility_data']
                recent_vol = np.mean(vol_data[-5:])  # Last 5 periods
                historical_vol = np.mean(vol_data[:-5])
                vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
                
                indicators['volatility_spike'] = {
                    'ratio': vol_ratio,
                    'alert': vol_ratio > 2.0,  # 2x normal volatility
                    'severity': 'high' if vol_ratio > 3.0 else 'medium' if vol_ratio > 2.0 else 'low'
                }
            
            # Correlation breakdown
            if 'correlation_data' in market_data:
                corr_data = market_data['correlation_data']
                recent_corr = np.mean(corr_data[-5:])
                historical_corr = np.mean(corr_data[:-5])
                corr_change = abs(recent_corr - historical_corr)
                
                indicators['correlation_breakdown'] = {
                    'change': corr_change,
                    'alert': corr_change > 0.3,  # 30% correlation change
                    'severity': 'high' if corr_change > 0.5 else 'medium' if corr_change > 0.3 else 'low'
                }
            
            # Liquidity stress
            if 'volume_data' in market_data:
                volume_data = market_data['volume_data']
                recent_volume = np.mean(volume_data[-5:])
                historical_volume = np.mean(volume_data[:-5])
                volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
                
                indicators['liquidity_stress'] = {
                    'ratio': volume_ratio,
                    'alert': volume_ratio < 0.5,  # 50% volume drop
                    'severity': 'high' if volume_ratio < 0.3 else 'medium' if volume_ratio < 0.5 else 'low'
                }
            
            # Market structure indicators
            if 'price_data' in market_data:
                price_data = market_data['price_data']
                returns = np.diff(price_data) / price_data[:-1]
                
                # Jump detection
                jump_threshold = 3 * np.std(returns)
                recent_jumps = np.sum(np.abs(returns[-10:]) > jump_threshold)
                
                indicators['price_jumps'] = {
                    'recent_jumps': recent_jumps,
                    'alert': recent_jumps > 2,
                    'severity': 'high' if recent_jumps > 3 else 'medium' if recent_jumps > 2 else 'low'
                }
            
            # Aggregate risk score
            alert_count = sum(1 for ind in indicators.values() if ind.get('alert', False))
            high_severity_count = sum(1 for ind in indicators.values() if ind.get('severity') == 'high')
            
            overall_risk_score = (alert_count * 0.3 + high_severity_count * 0.7) / len(indicators)
            
            indicators['overall_assessment'] = {
                'risk_score': overall_risk_score,
                'alert_level': 'critical' if overall_risk_score > 0.7 else 'high' if overall_risk_score > 0.5 else 'medium' if overall_risk_score > 0.3 else 'low',
                'recommendation': self._get_black_swan_recommendation(overall_risk_score)
            }
            
            logger.info(f"Black swan indicators analyzed: risk score {overall_risk_score:.2f}")
            return indicators
            
        except Exception as e:
            logger.error(f"Black swan indicator detection failed: {e}")
            raise
    
    def _get_black_swan_recommendation(self, risk_score: float) -> str:
        """Get recommendation based on black swan risk score."""
        if risk_score > 0.8:
            return "IMMEDIATE_RISK_REDUCTION: Consider significant position reduction and hedging"
        elif risk_score > 0.6:
            return "HIGH_ALERT: Increase monitoring frequency and prepare contingency plans"
        elif risk_score > 0.4:
            return "ELEVATED_CAUTION: Monitor closely and review risk limits"
        elif risk_score > 0.2:
            return "NORMAL_VIGILANCE: Continue standard risk monitoring"
        else:
            return "LOW_RISK: Normal market conditions"
    
    async def _execute_stress_scenario(self, 
                                     positions: Dict[str, float],
                                     scenario: StressScenario) -> StressTestResult:
        """Execute a single stress test scenario."""
        try:
            total_pnl = 0.0
            asset_pnl = {}
            
            # Calculate P&L for each position
            for symbol, position_value in positions.items():
                if symbol in scenario.asset_shocks:
                    shock = scenario.asset_shocks[symbol]
                    pnl = position_value * shock
                    asset_pnl[symbol] = pnl
                    total_pnl += pnl
                else:
                    # No shock specified, assume no impact
                    asset_pnl[symbol] = 0.0
            
            # Calculate percentage impact
            total_position_value = sum(abs(v) for v in positions.values())
            pnl_percentage = total_pnl / total_position_value if total_position_value > 0 else 0.0
            
            # Check if this breaches typical VaR limits
            var_breach = abs(pnl_percentage) > 0.05  # 5% loss threshold
            
            # Estimate liquidity impact (simplified)
            liquidity_impact = 0.0
            if scenario.severity in [ScenarioSeverity.SEVERE, ScenarioSeverity.EXTREME]:
                # Assume higher liquidity costs during severe stress
                liquidity_impact = abs(total_pnl) * 0.02  # 2% of loss as liquidity cost
            
            # Estimate recovery time
            recovery_time = None
            if scenario.duration_days:
                if scenario.severity == ScenarioSeverity.EXTREME:
                    recovery_time = scenario.duration_days * 3  # 3x longer to recover
                elif scenario.severity == ScenarioSeverity.SEVERE:
                    recovery_time = scenario.duration_days * 2
                else:
                    recovery_time = scenario.duration_days
            
            # Additional risk metrics
            risk_metrics = {
                'max_individual_loss': min(asset_pnl.values()) if asset_pnl else 0.0,
                'diversification_benefit': total_pnl - sum(asset_pnl.values()),
                'concentration_risk': max(abs(pnl) for pnl in asset_pnl.values()) / abs(total_pnl) if total_pnl != 0 else 0.0
            }
            
            return StressTestResult(
                scenario_name=scenario.name,
                scenario_type=scenario.scenario_type,
                total_pnl=total_pnl,
                pnl_percentage=pnl_percentage,
                asset_pnl=asset_pnl,
                var_breach=var_breach,
                liquidity_impact=liquidity_impact,
                recovery_time_days=recovery_time,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Stress scenario execution failed: {e}")
            raise
    
    async def create_custom_scenario(self,
                                   name: str,
                                   description: str,
                                   asset_shocks: Dict[str, float],
                                   severity: ScenarioSeverity = ScenarioSeverity.MODERATE) -> str:
        """Create a custom stress test scenario."""
        try:
            scenario_id = f"custom_{name.lower().replace(' ', '_')}"
            
            scenario = StressScenario(
                name=name,
                description=description,
                scenario_type=StressTestType.CUSTOM_SCENARIO,
                severity=severity,
                asset_shocks=asset_shocks,
                metadata={'created_by': 'user', 'created_at': datetime.utcnow().isoformat()}
            )
            
            self.custom_scenarios[scenario_id] = scenario
            
            logger.info(f"Custom scenario created: {scenario_id}")
            return scenario_id
            
        except Exception as e:
            logger.error(f"Custom scenario creation failed: {e}")
            raise
    
    async def get_scenario_recommendations(self,
                                         positions: Dict[str, float],
                                         risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """Get recommendations for stress testing based on portfolio."""
        try:
            recommendations = {
                'recommended_scenarios': [],
                'risk_assessment': {},
                'hedging_suggestions': [],
                'monitoring_priorities': []
            }
            
            # Analyze portfolio composition
            total_value = sum(abs(v) for v in positions.values())
            crypto_exposure = sum(abs(v) for k, v in positions.items() 
                                if k in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']) / total_value
            
            # Recommend scenarios based on portfolio
            if crypto_exposure > 0.5:
                recommendations['recommended_scenarios'].extend([
                    'covid_crash_2020',
                    'crypto_winter_2018',
                    'crypto_ban'
                ])
            
            if crypto_exposure > 0.8:
                recommendations['recommended_scenarios'].append('flash_crash_2010')
            
            # Risk assessment
            recommendations['risk_assessment'] = {
                'crypto_concentration': crypto_exposure,
                'risk_level': 'high' if crypto_exposure > 0.7 else 'medium' if crypto_exposure > 0.3 else 'low',
                'diversification_score': 1 - crypto_exposure
            }
            
            # Hedging suggestions
            if crypto_exposure > 0.6:
                recommendations['hedging_suggestions'].extend([
                    'Consider adding traditional assets (stocks, bonds)',
                    'Implement options-based hedging strategies',
                    'Maintain higher cash reserves'
                ])
            
            # Monitoring priorities
            recommendations['monitoring_priorities'] = [
                'Regulatory developments in major jurisdictions',
                'Cryptocurrency market sentiment indicators',
                'Traditional market correlation changes'
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Scenario recommendations failed: {e}")
            raise
    
    def get_available_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get all available stress test scenarios."""
        all_scenarios = {}
        
        # Historical scenarios
        for scenario_id, scenario in self.historical_scenarios.items():
            all_scenarios[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type.value,
                'severity': scenario.severity.value,
                'probability': scenario.probability,
                'historical_date': scenario.historical_date.isoformat() if scenario.historical_date else None
            }
        
        # Custom scenarios
        for scenario_id, scenario in self.custom_scenarios.items():
            all_scenarios[scenario_id] = {
                'name': scenario.name,
                'description': scenario.description,
                'type': scenario.scenario_type.value,
                'severity': scenario.severity.value,
                'custom': True
            }
        
        return all_scenarios


# Factory function
def create_stress_testing_engine() -> StressTestingEngine:
    """Create a stress testing engine instance."""
    return StressTestingEngine()


# Example usage
if __name__ == "__main__":
    async def test_stress_engine():
        """Test the stress testing engine."""
        print("Testing Stress Testing Engine...")
        
        # Create engine
        engine = create_stress_testing_engine()
        
        # Test portfolio
        positions = {
            'BTC': 50000,
            'ETH': 30000,
            'ADA': 20000
        }
        
        # Run historical stress tests
        historical_results = await engine.run_historical_stress_test(
            positions=positions,
            scenario_names=['covid_crash_2020', 'crypto_winter_2018']
        )
        
        print(f"Historical stress test results:")
        for result in historical_results:
            print(f"  {result.scenario_name}: ${result.total_pnl:,.2f} ({result.pnl_percentage:.1%})")
        
        # Run Monte Carlo stress test
        mc_config = MonteCarloStressConfig(n_simulations=1000)
        mc_results = await engine.run_monte_carlo_stress_test(positions, mc_config)
        
        print(f"\nMonte Carlo stress test results:")
        print(f"  Worst case: ${mc_results['worst_case_pnl']:,.2f}")
        print(f"  VaR 95%: ${mc_results['percentiles']['VaR_95.0']:,.2f}")
        print(f"  VaR 99%: ${mc_results['percentiles']['VaR_99.0']:,.2f}")
        
        print("Stress Testing Engine test completed successfully!")
    
    asyncio.run(test_stress_engine())