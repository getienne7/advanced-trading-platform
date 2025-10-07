"""
Tests for the multi-timeframe analysis system.
"""
import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List

from multi_timeframe_analyzer import (
    TimeframeType, MarketRegime, TimeframeSignal, CombinedSignal,
    MarketCondition, TimeframeWeightCalculator, MarketRegimeDetector,
    MultiTimeframeMovingAverageStrategy, SignalCombiner, MultiTimeframeAnalyzer
)
from backtesting_engine import MarketData


class TestTimeframeWeightCalculator:
    """Test suite for TimeframeWeightCalculator."""
    
    @pytest.fixture
    def weight_calculator(self) -> TimeframeWeightCalculator:
        """Create weight calculator for testing."""
        return TimeframeWeightCalculator()
    
    @pytest.fixture
    def sample_market_condition(self) -> MarketCondition:
        """Create sample market condition."""
        return MarketCondition(
            timestamp=datetime.now(),
            regime=MarketRegime.TRENDING_UP,
            volatility=0.02,
            trend_strength=0.8,
            volume_profile="medium",
            support_resistance_levels=[50000, 51000, 52000],
            optimal_timeframes=[TimeframeType.H1, TimeframeType.H4]
        )
    
    def test_weight_calculation(self, weight_calculator, sample_market_condition):
        """Test weight calculation for different timeframes."""
        timeframes = [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4]
        
        weights = weight_calculator.calculate_weights(sample_market_condition, timeframes)
        
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.001
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
        
        # Should have weights for all requested timeframes
        assert len(weights) == len(timeframes)
        for tf in timeframes:
            assert tf in weights
    
    def test_regime_adjustment(self, weight_calculator):
        """Test regime-based weight adjustments."""
        timeframes = [TimeframeType.M15, TimeframeType.H1, TimeframeType.H4]
        
        # Trending market should favor longer timeframes
        trending_condition = MarketCondition(
            timestamp=datetime.now(),
            regime=MarketRegime.TRENDING_UP,
            volatility=0.02,
            trend_strength=0.8,
            volume_profile="medium",
            support_resistance_levels=[],
            optimal_timeframes=timeframes
        )
        
        trending_weights = weight_calculator.calculate_weights(trending_condition, timeframes)
        
        # Volatile market should favor shorter timeframes
        volatile_condition = MarketCondition(
            timestamp=datetime.now(),
            regime=MarketRegime.VOLATILE,
            volatility=0.08,
            trend_strength=0.3,
            volume_profile="high",
            support_resistance_levels=[],
            optimal_timeframes=timeframes
        )
        
        volatile_weights = weight_calculator.calculate_weights(volatile_condition, timeframes)
        
        # In trending market, H4 should have higher weight than M15
        assert trending_weights[TimeframeType.H4] > trending_weights[TimeframeType.M15]
        
        # In volatile market, M15 should have higher weight than H4
        assert volatile_weights[TimeframeType.M15] > volatile_weights[TimeframeType.H4]


class TestMarketRegimeDetector:
    """Test suite for MarketRegimeDetector."""
    
    @pytest.fixture
    def regime_detector(self) -> MarketRegimeDetector:
        """Create regime detector for testing."""
        return MarketRegimeDetector(lookback_periods=50)
    
    @pytest.fixture
    def trending_up_data(self) -> List[MarketData]:
        """Create trending up market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # Strong uptrend
            price = base_price + i * 100 + np.random.normal(0, 50)
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
    def sideways_data(self) -> List[MarketData]:
        """Create sideways market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # Sideways with noise
            price = base_price + np.random.normal(0, 200)
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
    
    def test_trending_up_detection(self, regime_detector, trending_up_data):
        """Test detection of uptrending market."""
        condition = regime_detector.detect_regime(trending_up_data)
        
        assert condition.regime == MarketRegime.TRENDING_UP
        assert condition.trend_strength > 0.5
        assert condition.volatility > 0
    
    def test_sideways_detection(self, regime_detector, sideways_data):
        """Test detection of sideways market."""
        condition = regime_detector.detect_regime(sideways_data)
        
        assert condition.regime in [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]
        assert condition.trend_strength < 0.7
    
    def test_volatility_calculation(self, regime_detector):
        """Test volatility calculation."""
        # High volatility data
        high_vol_data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # High volatility
            price = base_price + np.random.normal(0, 2000)  # Large swings
            high_vol_data.append(MarketData(
                timestamp=base_time + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 500,
                low=price - 500,
                close=price,
                volume=1000.0
            ))
        
        condition = regime_detector.detect_regime(high_vol_data)
        assert condition.volatility > 0.02  # Should detect high volatility
    
    def test_support_resistance_detection(self, regime_detector, trending_up_data):
        """Test support and resistance level detection."""
        condition = regime_detector.detect_regime(trending_up_data)
        
        # Should find some support/resistance levels
        assert len(condition.support_resistance_levels) >= 0
        
        # Levels should be sorted
        levels = condition.support_resistance_levels
        if len(levels) > 1:
            assert levels == sorted(levels)


class TestMultiTimeframeMovingAverageStrategy:
    """Test suite for MultiTimeframeMovingAverageStrategy."""
    
    @pytest.fixture
    def strategy(self) -> MultiTimeframeMovingAverageStrategy:
        """Create multi-timeframe MA strategy for testing."""
        return MultiTimeframeMovingAverageStrategy(
            "test_multi_ma",
            {
                'timeframes': [TimeframeType.M15, TimeframeType.H1],
                'short_window': 5,
                'long_window': 10
            }
        )
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(100):
            # Trending data
            price = base_price + i * 50
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                symbol="BTCUSDT",
                open=price,
                high=price + 25,
                low=price - 25,
                close=price,
                volume=1000.0
            ))
        
        return data
    
    def test_timeframe_conversion(self, strategy, sample_market_data):
        """Test conversion of 1-minute data to different timeframes."""
        # Convert to 15-minute timeframe
        tf_data = strategy._convert_to_timeframe(sample_market_data, TimeframeType.M15)
        
        # Should have fewer data points
        assert len(tf_data) < len(sample_market_data)
        
        # Each bar should represent 15 minutes of data
        expected_bars = len(sample_market_data) // 15
        assert len(tf_data) <= expected_bars + 1  # +1 for partial bar
    
    def test_data_aggregation(self, strategy):
        """Test OHLCV data aggregation."""
        # Create test data points
        base_time = datetime(2023, 1, 1)
        data_points = [
            MarketData(base_time, "BTCUSDT", 100, 105, 95, 102, 1000),
            MarketData(base_time + timedelta(minutes=1), "BTCUSDT", 102, 108, 98, 106, 1200),
            MarketData(base_time + timedelta(minutes=2), "BTCUSDT", 106, 110, 104, 108, 800)
        ]
        
        aggregated = strategy._aggregate_data_points(data_points, TimeframeType.M15)
        
        # Check OHLCV values
        assert aggregated.open == 100  # First open
        assert aggregated.high == 110  # Highest high
        assert aggregated.low == 95    # Lowest low
        assert aggregated.close == 108 # Last close
        assert aggregated.volume == 3000  # Sum of volumes
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, strategy, sample_market_data):
        """Test signal generation for multiple timeframes."""
        signals = await strategy.generate_signals(sample_market_data)
        
        # Should generate signals for configured timeframes
        assert len(signals) > 0
        
        # Check signal structure
        for signal in signals:
            assert 'timestamp' in signal
            assert 'timeframe' in signal
            assert 'signal' in signal
            assert 'strength' in signal
            assert 'confidence' in signal
            assert 'price' in signal
            assert 'indicators' in signal
            
            # Signal should be valid
            assert signal['signal'] in ['buy', 'sell', 'hold']
            assert -1.0 <= signal['strength'] <= 1.0
            assert 0.0 <= signal['confidence'] <= 1.0


class TestSignalCombiner:
    """Test suite for SignalCombiner."""
    
    @pytest.fixture
    def signal_combiner(self) -> SignalCombiner:
        """Create signal combiner for testing."""
        return SignalCombiner()
    
    @pytest.fixture
    def sample_timeframe_signals(self) -> List[TimeframeSignal]:
        """Create sample timeframe signals."""
        return [
            TimeframeSignal(
                timeframe=TimeframeType.M15,
                timestamp=datetime.now(),
                signal_type="buy",
                strength=0.7,
                confidence=0.8,
                price=50000.0,
                indicators={'ma_diff': 100}
            ),
            TimeframeSignal(
                timeframe=TimeframeType.H1,
                timestamp=datetime.now(),
                signal_type="buy",
                strength=0.5,
                confidence=0.9,
                price=50000.0,
                indicators={'ma_diff': 200}
            ),
            TimeframeSignal(
                timeframe=TimeframeType.H4,
                timestamp=datetime.now(),
                signal_type="sell",
                strength=-0.3,
                confidence=0.6,
                price=50000.0,
                indicators={'ma_diff': -50}
            )
        ]
    
    @pytest.fixture
    def sample_market_condition(self) -> MarketCondition:
        """Create sample market condition."""
        return MarketCondition(
            timestamp=datetime.now(),
            regime=MarketRegime.TRENDING_UP,
            volatility=0.02,
            trend_strength=0.7,
            volume_profile="medium",
            support_resistance_levels=[],
            optimal_timeframes=[TimeframeType.H1, TimeframeType.H4]
        )
    
    def test_signal_combination(self, signal_combiner, sample_timeframe_signals, sample_market_condition):
        """Test combination of multiple timeframe signals."""
        combined = signal_combiner.combine_signals(sample_timeframe_signals, sample_market_condition)
        
        assert isinstance(combined, CombinedSignal)
        assert combined.primary_signal in ['buy', 'sell', 'hold']
        assert -1.0 <= combined.overall_strength <= 1.0
        assert 0.0 <= combined.overall_confidence <= 1.0
        assert len(combined.timeframe_signals) == len(sample_timeframe_signals)
        assert len(combined.weight_distribution) > 0
        assert 0.0 <= combined.recommended_position_size <= 1.0
    
    def test_empty_signals(self, signal_combiner, sample_market_condition):
        """Test handling of empty signal list."""
        combined = signal_combiner.combine_signals([], sample_market_condition)
        
        assert combined.primary_signal == "hold"
        assert combined.overall_strength == 0.0
        assert combined.overall_confidence == 0.0
        assert combined.recommended_position_size == 0.0
    
    def test_position_size_calculation(self, signal_combiner, sample_market_condition):
        """Test position size calculation logic."""
        # Strong buy signal
        strong_buy_signal = [TimeframeSignal(
            timeframe=TimeframeType.H1,
            timestamp=datetime.now(),
            signal_type="buy",
            strength=0.9,
            confidence=0.9,
            price=50000.0
        )]
        
        combined = signal_combiner.combine_signals(strong_buy_signal, sample_market_condition)
        assert combined.recommended_position_size > 0
        
        # Hold signal
        hold_signal = [TimeframeSignal(
            timeframe=TimeframeType.H1,
            timestamp=datetime.now(),
            signal_type="hold",
            strength=0.0,
            confidence=0.5,
            price=50000.0
        )]
        
        combined_hold = signal_combiner.combine_signals(hold_signal, sample_market_condition)
        assert combined_hold.recommended_position_size == 0.0


class TestMultiTimeframeAnalyzer:
    """Test suite for MultiTimeframeAnalyzer."""
    
    @pytest.fixture
    def analyzer(self) -> MultiTimeframeAnalyzer:
        """Create multi-timeframe analyzer for testing."""
        return MultiTimeframeAnalyzer(
            timeframes=[TimeframeType.M15, TimeframeType.H1],
            lookback_periods=50
        )
    
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Generate sample market data."""
        data = []
        base_time = datetime(2023, 1, 1)
        base_price = 50000.0
        
        for i in range(200):
            # Complex price pattern
            trend = i * 25
            cycle = 500 * np.sin(i * 0.1)
            noise = np.random.normal(0, 100)
            price = base_price + trend + cycle + noise
            
            data.append(MarketData(
                timestamp=base_time + timedelta(minutes=i),
                symbol="BTCUSDT",
                open=price,
                high=price + abs(noise) * 0.5,
                low=price - abs(noise) * 0.5,
                close=price,
                volume=1000 + abs(noise) * 5
            ))
        
        return data
    
    def test_strategy_addition(self, analyzer):
        """Test adding strategies to analyzer."""
        strategy = MultiTimeframeMovingAverageStrategy(
            "test_strategy",
            {'timeframes': [TimeframeType.M15, TimeframeType.H1]}
        )
        
        analyzer.add_strategy(strategy)
        
        assert "test_strategy" in analyzer.strategies
        assert analyzer.strategies["test_strategy"] == strategy
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, analyzer, sample_market_data):
        """Test comprehensive multi-timeframe analysis."""
        # Add a strategy
        strategy = MultiTimeframeMovingAverageStrategy(
            "ma_strategy",
            {
                'timeframes': [TimeframeType.M15, TimeframeType.H1],
                'short_window': 5,
                'long_window': 15
            }
        )
        analyzer.add_strategy(strategy)
        
        # Run analysis
        result = await analyzer.analyze(sample_market_data)
        
        # Verify result structure
        assert 'market_condition' in result
        assert 'strategy_results' in result
        assert 'overall_signal' in result
        assert 'timeframe_analysis' in result
        assert 'recommendations' in result
        
        # Verify market condition
        market_condition = result['market_condition']
        assert isinstance(market_condition, MarketCondition)
        assert isinstance(market_condition.regime, MarketRegime)
        assert market_condition.volatility >= 0
        assert 0 <= market_condition.trend_strength <= 1
        
        # Verify strategy results
        strategy_results = result['strategy_results']
        if strategy_results:
            assert 'ma_strategy' in strategy_results
            combined_signal = strategy_results['ma_strategy']
            assert isinstance(combined_signal, CombinedSignal)
        
        # Verify recommendations
        recommendations = result['recommendations']
        assert 'market_regime' in recommendations
        assert 'optimal_timeframes' in recommendations
        assert 'volatility_level' in recommendations
    
    def test_timeframe_performance_analysis(self, analyzer):
        """Test timeframe performance analysis."""
        # Create sample signals
        signals = [
            TimeframeSignal(
                timeframe=TimeframeType.M15,
                timestamp=datetime.now(),
                signal_type="buy",
                strength=0.7,
                confidence=0.8,
                price=50000.0
            ),
            TimeframeSignal(
                timeframe=TimeframeType.M15,
                timestamp=datetime.now(),
                signal_type="sell",
                strength=-0.5,
                confidence=0.6,
                price=50000.0
            ),
            TimeframeSignal(
                timeframe=TimeframeType.H1,
                timestamp=datetime.now(),
                signal_type="buy",
                strength=0.8,
                confidence=0.9,
                price=50000.0
            )
        ]
        
        analysis = analyzer._analyze_timeframe_performance(signals)
        
        # Should have analysis for both timeframes
        assert 'm15' in analysis
        assert '1h' in analysis
        
        # M15 should have 2 signals
        assert analysis['m15']['signal_count'] == 2
        assert analysis['m15']['buy_signals'] == 1
        assert analysis['m15']['sell_signals'] == 1
        
        # H1 should have 1 signal
        assert analysis['1h']['signal_count'] == 1
        assert analysis['1h']['buy_signals'] == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])