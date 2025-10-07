"""
Multi-Timeframe Analysis System
Implements signal combination from different timeframes, timeframe-specific strategy allocation,
and adaptive timeframe selection based on market conditions.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field

from backtesting_engine import (
    MarketData, TradingStrategy, Order, OrderSide, OrderType,
    BacktestingEngine, BacktestingConfig
)

# Configure logging
logger = structlog.get_logger("multi-timeframe-analyzer")


class TimeframeType(str, Enum):
    """Supported timeframe types."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class SignalStrength(str, Enum):
    """Signal strength levels."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class MarketRegime(str, Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe."""
    timeframe: TimeframeType
    timestamp: datetime
    signal_type: str  # "buy", "sell", "hold"
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price: float
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedSignal:
    """Combined signal from multiple timeframes."""
    timestamp: datetime
    primary_signal: str  # "buy", "sell", "hold"
    overall_strength: float
    overall_confidence: float
    timeframe_signals: List[TimeframeSignal]
    weight_distribution: Dict[TimeframeType, float]
    market_regime: MarketRegime
    recommended_position_size: float


@dataclass
class MarketCondition:
    """Current market condition assessment."""
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_profile: str  # "high", "medium", "low"
    support_resistance_levels: List[float]
    optimal_timeframes: List[TimeframeType]


class TimeframeWeightCalculator:
    """Calculate dynamic weights for different timeframes based on market conditions."""
    
    def __init__(self):
        self.base_weights = {
            TimeframeType.M1: 0.05,
            TimeframeType.M5: 0.10,
            TimeframeType.M15: 0.15,
            TimeframeType.M30: 0.20,
            TimeframeType.H1: 0.25,
            TimeframeType.H4: 0.20,
            TimeframeType.D1: 0.05
        }
    
    def calculate_weights(self, 
                         market_condition: MarketCondition,
                         available_timeframes: List[TimeframeType]) -> Dict[TimeframeType, float]:
        """Calculate dynamic weights based on market conditions."""
        weights = {}
        
        for timeframe in available_timeframes:
            base_weight = self.base_weights.get(timeframe, 0.1)
            
            # Adjust based on market regime
            regime_adjustment = self._get_regime_adjustment(market_condition.regime, timeframe)
            
            # Adjust based on volatility
            volatility_adjustment = self._get_volatility_adjustment(market_condition.volatility, timeframe)
            
            # Adjust based on trend strength
            trend_adjustment = self._get_trend_adjustment(market_condition.trend_strength, timeframe)
            
            # Calculate final weight
            final_weight = base_weight * regime_adjustment * volatility_adjustment * trend_adjustment
            weights[timeframe] = final_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {tf: w / total_weight for tf, w in weights.items()}
        
        return weights
    
    def _get_regime_adjustment(self, regime: MarketRegime, timeframe: TimeframeType) -> float:
        """Get weight adjustment based on market regime."""
        adjustments = {
            MarketRegime.TRENDING_UP: {
                TimeframeType.M1: 0.8,
                TimeframeType.M5: 0.9,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 1.1,
                TimeframeType.H1: 1.2,
                TimeframeType.H4: 1.3,
                TimeframeType.D1: 1.1
            },
            MarketRegime.TRENDING_DOWN: {
                TimeframeType.M1: 0.8,
                TimeframeType.M5: 0.9,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 1.1,
                TimeframeType.H1: 1.2,
                TimeframeType.H4: 1.3,
                TimeframeType.D1: 1.1
            },
            MarketRegime.SIDEWAYS: {
                TimeframeType.M1: 1.2,
                TimeframeType.M5: 1.1,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 0.9,
                TimeframeType.H1: 0.8,
                TimeframeType.H4: 0.7,
                TimeframeType.D1: 0.6
            },
            MarketRegime.VOLATILE: {
                TimeframeType.M1: 1.3,
                TimeframeType.M5: 1.2,
                TimeframeType.M15: 1.1,
                TimeframeType.M30: 1.0,
                TimeframeType.H1: 0.9,
                TimeframeType.H4: 0.8,
                TimeframeType.D1: 0.7
            },
            MarketRegime.LOW_VOLATILITY: {
                TimeframeType.M1: 0.7,
                TimeframeType.M5: 0.8,
                TimeframeType.M15: 0.9,
                TimeframeType.M30: 1.0,
                TimeframeType.H1: 1.1,
                TimeframeType.H4: 1.2,
                TimeframeType.D1: 1.3
            }
        }
        
        return adjustments.get(regime, {}).get(timeframe, 1.0)
    
    def _get_volatility_adjustment(self, volatility: float, timeframe: TimeframeType) -> float:
        """Get weight adjustment based on volatility."""
        if volatility > 0.05:  # High volatility
            short_term_boost = {
                TimeframeType.M1: 1.2,
                TimeframeType.M5: 1.1,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 0.9,
                TimeframeType.H1: 0.8,
                TimeframeType.H4: 0.7,
                TimeframeType.D1: 0.6
            }
            return short_term_boost.get(timeframe, 1.0)
        elif volatility < 0.01:  # Low volatility
            long_term_boost = {
                TimeframeType.M1: 0.6,
                TimeframeType.M5: 0.7,
                TimeframeType.M15: 0.8,
                TimeframeType.M30: 0.9,
                TimeframeType.H1: 1.0,
                TimeframeType.H4: 1.1,
                TimeframeType.D1: 1.2
            }
            return long_term_boost.get(timeframe, 1.0)
        else:
            return 1.0  # Normal volatility
    
    def _get_trend_adjustment(self, trend_strength: float, timeframe: TimeframeType) -> float:
        """Get weight adjustment based on trend strength."""
        if trend_strength > 0.7:  # Strong trend
            # Favor longer timeframes for strong trends
            trend_boost = {
                TimeframeType.M1: 0.8,
                TimeframeType.M5: 0.9,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 1.1,
                TimeframeType.H1: 1.2,
                TimeframeType.H4: 1.3,
                TimeframeType.D1: 1.2
            }
            return trend_boost.get(timeframe, 1.0)
        elif trend_strength < 0.3:  # Weak trend
            # Favor shorter timeframes for weak trends
            weak_trend_boost = {
                TimeframeType.M1: 1.2,
                TimeframeType.M5: 1.1,
                TimeframeType.M15: 1.0,
                TimeframeType.M30: 0.9,
                TimeframeType.H1: 0.8,
                TimeframeType.H4: 0.7,
                TimeframeType.D1: 0.6
            }
            return weak_trend_boost.get(timeframe, 1.0)
        else:
            return 1.0  # Moderate trend


class MarketRegimeDetector:
    """Detect current market regime based on price action and indicators."""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
    
    def detect_regime(self, market_data: List[MarketData]) -> MarketCondition:
        """Detect current market regime and conditions."""
        if len(market_data) < self.lookback_periods:
            return MarketCondition(
                timestamp=market_data[-1].timestamp,
                regime=MarketRegime.SIDEWAYS,
                volatility=0.02,
                trend_strength=0.5,
                volume_profile="medium",
                support_resistance_levels=[],
                optimal_timeframes=[TimeframeType.H1, TimeframeType.H4]
            )
        
        recent_data = market_data[-self.lookback_periods:]
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(recent_data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(recent_data)
        
        # Determine regime
        regime = self._determine_regime(recent_data, trend_strength, volatility)
        
        # Calculate volume profile
        volume_profile = self._analyze_volume_profile(recent_data)
        
        # Find support/resistance levels
        support_resistance = self._find_support_resistance_levels(recent_data)
        
        # Determine optimal timeframes
        optimal_timeframes = self._determine_optimal_timeframes(regime, volatility, trend_strength)
        
        return MarketCondition(
            timestamp=recent_data[-1].timestamp,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            support_resistance_levels=support_resistance,
            optimal_timeframes=optimal_timeframes
        )
    
    def _calculate_trend_strength(self, data: List[MarketData]) -> float:
        """Calculate trend strength using linear regression."""
        prices = [d.close for d in data]
        x = np.arange(len(prices))
        
        # Linear regression
        slope, _ = np.polyfit(x, prices, 1)
        
        # Calculate R-squared
        y_pred = slope * x + np.mean(prices)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return max(0, min(1, r_squared))
    
    def _calculate_volatility(self, data: List[MarketData]) -> float:
        """Calculate price volatility."""
        prices = [d.close for d in data]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) if returns else 0.02
    
    def _determine_regime(self, data: List[MarketData], trend_strength: float, volatility: float) -> MarketRegime:
        """Determine market regime based on trend and volatility."""
        prices = [d.close for d in data]
        
        # Calculate price direction
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        if volatility > 0.05:
            return MarketRegime.VOLATILE
        elif volatility < 0.01:
            return MarketRegime.LOW_VOLATILITY
        elif trend_strength > 0.6:
            if price_change > 0.02:
                return MarketRegime.TRENDING_UP
            elif price_change < -0.02:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS
        else:
            return MarketRegime.SIDEWAYS
    
    def _analyze_volume_profile(self, data: List[MarketData]) -> str:
        """Analyze volume profile."""
        volumes = [d.volume for d in data]
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])  # Last 10 periods
        
        if recent_volume > avg_volume * 1.5:
            return "high"
        elif recent_volume < avg_volume * 0.5:
            return "low"
        else:
            return "medium"
    
    def _find_support_resistance_levels(self, data: List[MarketData]) -> List[float]:
        """Find key support and resistance levels."""
        highs = [d.high for d in data]
        lows = [d.low for d in data]
        
        # Simple approach: find local maxima and minima
        levels = []
        
        # Find resistance levels (local maxima)
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                levels.append(highs[i])
        
        # Find support levels (local minima)
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                levels.append(lows[i])
        
        # Return unique levels sorted
        return sorted(list(set(levels)))
    
    def _determine_optimal_timeframes(self, regime: MarketRegime, volatility: float, trend_strength: float) -> List[TimeframeType]:
        """Determine optimal timeframes for current market conditions."""
        if regime == MarketRegime.VOLATILE:
            return [TimeframeType.M1, TimeframeType.M5, TimeframeType.M15]
        elif regime == MarketRegime.LOW_VOLATILITY:
            return [TimeframeType.H1, TimeframeType.H4, TimeframeType.D1]
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if trend_strength > 0.7:
                return [TimeframeType.M30, TimeframeType.H1, TimeframeType.H4]
            else:
                return [TimeframeType.M15, TimeframeType.M30, TimeframeType.H1]
        else:  # SIDEWAYS
            return [TimeframeType.M5, TimeframeType.M15, TimeframeType.M30]


class MultiTimeframeStrategy(TradingStrategy):
    """Base class for multi-timeframe trading strategies."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.timeframes = parameters.get('timeframes', [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4])
        self.timeframe_data = {tf: [] for tf in self.timeframes}
        self.timeframe_signals = {tf: [] for tf in self.timeframes}
        
    @abstractmethod
    async def generate_timeframe_signal(self, timeframe: TimeframeType, data: List[MarketData]) -> Optional[TimeframeSignal]:
        """Generate signal for a specific timeframe."""
        pass
    
    async def generate_signals(self, market_data: List[MarketData]) -> List[Dict[str, Any]]:
        """Generate signals from all timeframes."""
        signals = []
        
        # Generate signals for each timeframe
        for timeframe in self.timeframes:
            # Convert data to appropriate timeframe
            tf_data = self._convert_to_timeframe(market_data, timeframe)
            
            if len(tf_data) > 0:
                signal = await self.generate_timeframe_signal(timeframe, tf_data)
                if signal:
                    self.timeframe_signals[timeframe].append(signal)
                    signals.append({
                        'timestamp': signal.timestamp,
                        'timeframe': signal.timeframe.value,
                        'signal': signal.signal_type,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'price': signal.price,
                        'indicators': signal.indicators
                    })
        
        return signals
    
    def _convert_to_timeframe(self, data: List[MarketData], timeframe: TimeframeType) -> List[MarketData]:
        """Convert 1-minute data to specified timeframe."""
        if not data:
            return []
        
        # Get timeframe in minutes
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        if timeframe_minutes == 1:
            return data  # Already 1-minute data
        
        # Group data by timeframe periods
        grouped_data = []
        current_group = []
        
        for i, point in enumerate(data):
            current_group.append(point)
            
            # Check if we should close this timeframe period
            if (i + 1) % timeframe_minutes == 0 or i == len(data) - 1:
                if current_group:
                    # Create aggregated data point
                    aggregated = self._aggregate_data_points(current_group, timeframe)
                    grouped_data.append(aggregated)
                    current_group = []
        
        return grouped_data
    
    def _get_timeframe_minutes(self, timeframe: TimeframeType) -> int:
        """Get timeframe duration in minutes."""
        timeframe_map = {
            TimeframeType.M1: 1,
            TimeframeType.M5: 5,
            TimeframeType.M15: 15,
            TimeframeType.M30: 30,
            TimeframeType.H1: 60,
            TimeframeType.H4: 240,
            TimeframeType.D1: 1440,
            TimeframeType.W1: 10080
        }
        return timeframe_map.get(timeframe, 60)
    
    def _aggregate_data_points(self, data_points: List[MarketData], timeframe: TimeframeType) -> MarketData:
        """Aggregate multiple data points into a single timeframe bar."""
        if not data_points:
            raise ValueError("Cannot aggregate empty data points")
        
        # OHLCV aggregation
        open_price = data_points[0].open
        high_price = max(point.high for point in data_points)
        low_price = min(point.low for point in data_points)
        close_price = data_points[-1].close
        total_volume = sum(point.volume for point in data_points)
        
        # Use the timestamp of the last data point
        timestamp = data_points[-1].timestamp
        symbol = data_points[0].symbol
        
        return MarketData(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume
        )


class MultiTimeframeMovingAverageStrategy(MultiTimeframeStrategy):
    """Multi-timeframe moving average strategy."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.short_window = parameters.get('short_window', 10)
        self.long_window = parameters.get('long_window', 30)
        self.timeframe_histories = {tf: [] for tf in self.timeframes}
    
    async def generate_timeframe_signal(self, timeframe: TimeframeType, data: List[MarketData]) -> Optional[TimeframeSignal]:
        """Generate moving average signal for specific timeframe."""
        if len(data) < self.long_window:
            return None
        
        # Calculate moving averages
        prices = [d.close for d in data]
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Determine signal
        if short_ma > long_ma:
            signal_type = "buy"
            strength = min(1.0, (short_ma - long_ma) / long_ma)
        elif short_ma < long_ma:
            signal_type = "sell"
            strength = max(-1.0, (short_ma - long_ma) / long_ma)
        else:
            signal_type = "hold"
            strength = 0.0
        
        # Calculate confidence based on trend consistency
        recent_prices = prices[-5:]  # Last 5 periods
        trend_consistency = self._calculate_trend_consistency(recent_prices)
        confidence = trend_consistency
        
        return TimeframeSignal(
            timeframe=timeframe,
            timestamp=data[-1].timestamp,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=data[-1].close,
            indicators={
                'short_ma': short_ma,
                'long_ma': long_ma,
                'ma_diff': short_ma - long_ma,
                'ma_diff_pct': (short_ma - long_ma) / long_ma
            }
        )
    
    def _calculate_trend_consistency(self, prices: List[float]) -> float:
        """Calculate trend consistency for confidence measure."""
        if len(prices) < 2:
            return 0.5
        
        # Count consistent moves
        consistent_moves = 0
        total_moves = len(prices) - 1
        
        for i in range(1, len(prices)):
            if i == 1:
                continue
            
            current_move = prices[i] - prices[i-1]
            previous_move = prices[i-1] - prices[i-2]
            
            # Check if moves are in same direction
            if (current_move > 0 and previous_move > 0) or (current_move < 0 and previous_move < 0):
                consistent_moves += 1
        
        return consistent_moves / max(1, total_moves - 1)
    
    async def on_market_data(self, data: MarketData) -> Optional[Order]:
        """Process market data and generate orders based on multi-timeframe analysis."""
        # This would be implemented by the MultiTimeframeAnalyzer
        return None


class SignalCombiner:
    """Combine signals from multiple timeframes into a single trading decision."""
    
    def __init__(self):
        self.weight_calculator = TimeframeWeightCalculator()
    
    def combine_signals(self, 
                       timeframe_signals: List[TimeframeSignal],
                       market_condition: MarketCondition) -> CombinedSignal:
        """Combine multiple timeframe signals into a single decision."""
        if not timeframe_signals:
            return CombinedSignal(
                timestamp=datetime.now(),
                primary_signal="hold",
                overall_strength=0.0,
                overall_confidence=0.0,
                timeframe_signals=[],
                weight_distribution={},
                market_regime=market_condition.regime,
                recommended_position_size=0.0
            )
        
        # Calculate weights for each timeframe
        available_timeframes = [signal.timeframe for signal in timeframe_signals]
        weights = self.weight_calculator.calculate_weights(market_condition, available_timeframes)
        
        # Calculate weighted signals
        weighted_strength = 0.0
        weighted_confidence = 0.0
        signal_votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for signal in timeframe_signals:
            weight = weights.get(signal.timeframe, 0.0)
            
            # Accumulate weighted strength and confidence
            weighted_strength += signal.strength * weight * signal.confidence
            weighted_confidence += signal.confidence * weight
            
            # Vote for signal type
            signal_votes[signal.signal_type] += weight * signal.confidence
        
        # Determine primary signal
        primary_signal = max(signal_votes, key=signal_votes.get)
        
        # Normalize confidence
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_confidence /= total_weight
        
        # Calculate recommended position size
        position_size = self._calculate_position_size(
            primary_signal, abs(weighted_strength), weighted_confidence, market_condition
        )
        
        return CombinedSignal(
            timestamp=timeframe_signals[-1].timestamp,
            primary_signal=primary_signal,
            overall_strength=weighted_strength,
            overall_confidence=weighted_confidence,
            timeframe_signals=timeframe_signals,
            weight_distribution=weights,
            market_regime=market_condition.regime,
            recommended_position_size=position_size
        )
    
    def _calculate_position_size(self, 
                               signal: str, 
                               strength: float, 
                               confidence: float,
                               market_condition: MarketCondition) -> float:
        """Calculate recommended position size based on signal quality and market conditions."""
        if signal == "hold":
            return 0.0
        
        # Base position size from signal strength and confidence
        base_size = strength * confidence
        
        # Adjust for market regime
        regime_adjustment = {
            MarketRegime.TRENDING_UP: 1.2 if signal == "buy" else 0.8,
            MarketRegime.TRENDING_DOWN: 1.2 if signal == "sell" else 0.8,
            MarketRegime.SIDEWAYS: 0.7,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.LOW_VOLATILITY: 1.1
        }
        
        adjustment = regime_adjustment.get(market_condition.regime, 1.0)
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.3, 1.0 - market_condition.volatility * 10)
        
        final_size = base_size * adjustment * volatility_adjustment
        
        # Cap at reasonable limits
        return max(0.0, min(1.0, final_size))


class MultiTimeframeAnalyzer:
    """Main multi-timeframe analysis system."""
    
    def __init__(self, 
                 timeframes: List[TimeframeType] = None,
                 lookback_periods: int = 100):
        self.timeframes = timeframes or [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4]
        self.regime_detector = MarketRegimeDetector(lookback_periods)
        self.signal_combiner = SignalCombiner()
        self.strategies = {}
        
    def add_strategy(self, strategy: MultiTimeframeStrategy):
        """Add a multi-timeframe strategy."""
        self.strategies[strategy.name] = strategy
    
    async def analyze(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Perform comprehensive multi-timeframe analysis."""
        # Detect market conditions
        market_condition = self.regime_detector.detect_regime(market_data)
        
        # Generate signals from all strategies and timeframes
        all_signals = []
        strategy_results = {}
        
        for strategy_name, strategy in self.strategies.items():
            signals = await strategy.generate_signals(market_data)
            
            # Convert to TimeframeSignal objects
            timeframe_signals = []
            for signal_dict in signals:
                tf_signal = TimeframeSignal(
                    timeframe=TimeframeType(signal_dict['timeframe']),
                    timestamp=signal_dict['timestamp'],
                    signal_type=signal_dict['signal'],
                    strength=signal_dict['strength'],
                    confidence=signal_dict['confidence'],
                    price=signal_dict['price'],
                    indicators=signal_dict.get('indicators', {}),
                    metadata={'strategy': strategy_name}
                )
                timeframe_signals.append(tf_signal)
                all_signals.append(tf_signal)
            
            # Combine signals for this strategy
            if timeframe_signals:
                combined_signal = self.signal_combiner.combine_signals(timeframe_signals, market_condition)
                strategy_results[strategy_name] = combined_signal
        
        # Overall combined signal from all strategies
        overall_combined = None
        if all_signals:
            overall_combined = self.signal_combiner.combine_signals(all_signals, market_condition)
        
        return {
            'market_condition': market_condition,
            'strategy_results': strategy_results,
            'overall_signal': overall_combined,
            'timeframe_analysis': self._analyze_timeframe_performance(all_signals),
            'recommendations': self._generate_recommendations(market_condition, overall_combined)
        }
    
    def _analyze_timeframe_performance(self, signals: List[TimeframeSignal]) -> Dict[str, Any]:
        """Analyze performance of different timeframes."""
        timeframe_stats = defaultdict(list)
        
        for signal in signals:
            timeframe_stats[signal.timeframe].append({
                'strength': signal.strength,
                'confidence': signal.confidence,
                'signal_type': signal.signal_type
            })
        
        analysis = {}
        for timeframe, stats in timeframe_stats.items():
            if stats:
                analysis[timeframe.value] = {
                    'signal_count': len(stats),
                    'avg_strength': np.mean([s['strength'] for s in stats]),
                    'avg_confidence': np.mean([s['confidence'] for s in stats]),
                    'buy_signals': len([s for s in stats if s['signal_type'] == 'buy']),
                    'sell_signals': len([s for s in stats if s['signal_type'] == 'sell']),
                    'hold_signals': len([s for s in stats if s['signal_type'] == 'hold'])
                }
        
        return analysis
    
    def _generate_recommendations(self, 
                                market_condition: MarketCondition,
                                combined_signal: Optional[CombinedSignal]) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis."""
        recommendations = {
            'market_regime': market_condition.regime.value,
            'optimal_timeframes': [tf.value for tf in market_condition.optimal_timeframes],
            'volatility_level': 'high' if market_condition.volatility > 0.05 else 'low' if market_condition.volatility < 0.01 else 'medium',
            'trend_strength': market_condition.trend_strength
        }
        
        if combined_signal:
            recommendations.update({
                'primary_signal': combined_signal.primary_signal,
                'signal_strength': combined_signal.overall_strength,
                'signal_confidence': combined_signal.overall_confidence,
                'recommended_position_size': combined_signal.recommended_position_size,
                'timeframe_weights': {tf.value: weight for tf, weight in combined_signal.weight_distribution.items()}
            })
            
            # Risk warnings
            warnings = []
            if combined_signal.overall_confidence < 0.5:
                warnings.append("Low signal confidence - consider reducing position size")
            if market_condition.volatility > 0.05:
                warnings.append("High volatility detected - use tighter stops")
            if market_condition.regime == MarketRegime.SIDEWAYS:
                warnings.append("Sideways market - consider range trading strategies")
            
            recommendations['warnings'] = warnings
        
        return recommendations


# Example usage and testing
async def demo_multi_timeframe_analysis():
    """Demonstrate multi-timeframe analysis system."""
    print("=== Multi-Timeframe Analysis Demo ===")
    
    # Create sample market data
    market_data = []
    base_time = datetime(2023, 1, 1)
    base_price = 50000.0
    
    for i in range(500):  # 500 minutes of data
        # Create complex price pattern
        trend = i * 10  # Uptrend
        cycle1 = 500 * np.sin(i * 0.1)  # Long cycle
        cycle2 = 200 * np.sin(i * 0.3)  # Short cycle
        noise = np.random.normal(0, 50)  # Random noise
        
        price = base_price + trend + cycle1 + cycle2 + noise
        
        market_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            symbol="BTCUSDT",
            open=price,
            high=price + abs(noise) * 0.5,
            low=price - abs(noise) * 0.5,
            close=price,
            volume=1000 + abs(noise) * 10
        ))
    
    # Create multi-timeframe analyzer
    analyzer = MultiTimeframeAnalyzer(
        timeframes=[TimeframeType.M15, TimeframeType.H1, TimeframeType.H4]
    )
    
    # Add multi-timeframe strategy
    strategy = MultiTimeframeMovingAverageStrategy(
        "multi_tf_ma",
        {
            'timeframes': [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4],
            'short_window': 10,
            'long_window': 30
        }
    )
    analyzer.add_strategy(strategy)
    
    # Run analysis
    result = await analyzer.analyze(market_data)
    
    # Display results
    print(f"\nMarket Condition:")
    print(f"  Regime: {result['market_condition'].regime.value}")
    print(f"  Volatility: {result['market_condition'].volatility:.4f}")
    print(f"  Trend Strength: {result['market_condition'].trend_strength:.4f}")
    print(f"  Optimal Timeframes: {[tf.value for tf in result['market_condition'].optimal_timeframes]}")
    
    if result['overall_signal']:
        print(f"\nOverall Signal:")
        print(f"  Primary Signal: {result['overall_signal'].primary_signal}")
        print(f"  Strength: {result['overall_signal'].overall_strength:.4f}")
        print(f"  Confidence: {result['overall_signal'].overall_confidence:.4f}")
        print(f"  Recommended Position Size: {result['overall_signal'].recommended_position_size:.4f}")
        
        print(f"\nTimeframe Weights:")
        for tf, weight in result['overall_signal'].weight_distribution.items():
            print(f"  {tf.value}: {weight:.3f}")
    
    print(f"\nTimeframe Analysis:")
    for tf, stats in result['timeframe_analysis'].items():
        print(f"  {tf}:")
        print(f"    Signals: {stats['signal_count']}")
        print(f"    Avg Strength: {stats['avg_strength']:.3f}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"    Buy/Sell/Hold: {stats['buy_signals']}/{stats['sell_signals']}/{stats['hold_signals']}")
    
    print(f"\nRecommendations:")
    for key, value in result['recommendations'].items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    asyncio.run(demo_multi_timeframe_analysis())