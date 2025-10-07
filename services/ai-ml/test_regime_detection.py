"""
Test suite for the Market Regime Detection System.
Tests HMM regime identification, GARCH volatility forecasting, and strategy selection.
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from regime_detection import MarketRegimeDetector


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.regime_model_path = "test_models/regime_detection"


@pytest.fixture
def regime_detector():
    """Create a regime detector for testing."""
    config = MockConfig()
    detector = MarketRegimeDetector(config)
    detector.ready = True
    return detector


@pytest.mark.asyncio
async def test_regime_detector_initialization():
    """Test regime detector initialization."""
    config = MockConfig()
    detector = MarketRegimeDetector(config)
    
    # Test that detector is created with correct configuration
    assert detector.config == config
    assert not detector.ready
    assert detector.regime_config['n_regimes'] == 4
    assert len(detector.regime_labels) == 4


@pytest.mark.asyncio
async def test_synthetic_data_generation(regime_detector):
    """Test synthetic historical data generation with regime changes."""
    
    # Test BTC data generation
    btc_data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    
    assert len(btc_data) == 499, "Should generate 499 data points (500 - 1 for returns)"
    assert all(col in btc_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert all(btc_data['high'] >= btc_data['low']), "High should be >= Low"
    assert all(btc_data['high'] >= btc_data['close']), "High should be >= Close"
    assert all(btc_data['low'] <= btc_data['close']), "Low should be <= Close"
    assert all(btc_data['volume'] > 0), "Volume should be positive"
    
    # Check for price variation (should not be constant)
    price_std = btc_data['close'].std()
    assert price_std > 1000, "Prices should have reasonable variation"


@pytest.mark.asyncio
async def test_feature_extraction(regime_detector):
    """Test regime feature extraction."""
    
    # Generate test data
    data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    
    # Extract features
    features = regime_detector._extract_regime_features(data)
    
    # Check that all expected features are present
    expected_features = ['returns', 'volatility', 'volume_ratio', 'momentum', 'rsi', 'bb_position', 'trend_strength']
    for feature in expected_features:
        assert feature in features.columns, f"{feature} should be present"
    
    # Check feature bounds
    rsi_values = features['rsi'].dropna()
    assert all(0 <= rsi <= 100 for rsi in rsi_values), "RSI should be between 0 and 100"
    
    # Check that features don't have excessive NaN values
    for feature in expected_features:
        nan_ratio = features[feature].isna().sum() / len(features)
        assert nan_ratio < 0.1, f"{feature} should have less than 10% NaN values"


@pytest.mark.asyncio
async def test_regime_model_training(regime_detector):
    """Test regime model training."""
    
    # Generate test data
    data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    features = regime_detector._extract_regime_features(data)
    
    # Train regime model
    await regime_detector._train_regime_model("BTC/USDT", features)
    
    # Check that model was created
    model_key = "regime_BTC/USDT"
    scaler_key = "regime_scaler_BTC/USDT"
    
    assert model_key in regime_detector.regime_models, "Regime model should be created"
    assert scaler_key in regime_detector.scalers, "Feature scaler should be created"
    
    # Check model properties
    model = regime_detector.regime_models[model_key]
    assert hasattr(model, 'n_components'), "Model should be a Gaussian Mixture Model"
    assert model.n_components == 4, "Model should have 4 components (regimes)"


@pytest.mark.asyncio
async def test_regime_prediction(regime_detector):
    """Test regime prediction functionality."""
    
    # Generate test data and train model
    data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    features = regime_detector._extract_regime_features(data)
    await regime_detector._train_regime_model("BTC/USDT", features)
    
    # Get trained model
    model = regime_detector.regime_models["regime_BTC/USDT"]
    
    # Predict current regime
    regime_id, probability = regime_detector._predict_current_regime(model, features)
    
    # Check prediction validity
    assert 0 <= regime_id < 4, "Regime ID should be between 0 and 3"
    assert 0 <= probability <= 1, "Probability should be between 0 and 1"
    assert regime_id in regime_detector.regime_labels, "Regime ID should be valid"
    
    # Check regime label
    regime_label = regime_detector.regime_labels[regime_id]
    assert regime_label in ['bull', 'bear', 'sideways', 'volatile'], "Regime label should be valid"


@pytest.mark.asyncio
async def test_garch_model_training(regime_detector):
    """Test GARCH model training for volatility forecasting."""
    
    # Generate test data
    data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    returns = data['close'].pct_change().dropna()
    
    # Train GARCH model
    await regime_detector._train_garch_model("BTC/USDT", returns)
    
    # Check that model was created
    model_key = "garch_BTC/USDT"
    assert model_key in regime_detector.garch_models, "GARCH model should be created"
    
    # Check model parameters
    garch_model = regime_detector.garch_models[model_key]
    assert 'omega' in garch_model, "GARCH model should have omega parameter"
    assert 'alpha' in garch_model, "GARCH model should have alpha parameter"
    assert 'beta' in garch_model, "GARCH model should have beta parameter"
    assert 'last_variance' in garch_model, "GARCH model should have last variance"
    
    # Check parameter bounds (typical GARCH constraints)
    assert garch_model['omega'] > 0, "Omega should be positive"
    assert 0 <= garch_model['alpha'] <= 1, "Alpha should be between 0 and 1"
    assert 0 <= garch_model['beta'] <= 1, "Beta should be between 0 and 1"
    assert garch_model['alpha'] + garch_model['beta'] < 1, "Alpha + Beta should be less than 1 for stationarity"


@pytest.mark.asyncio
async def test_volatility_forecasting(regime_detector):
    """Test volatility forecasting using GARCH."""
    
    # Generate test data
    data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    
    # Forecast volatility
    volatility_forecast = await regime_detector._forecast_volatility("BTC/USDT", data)
    
    # Check forecast validity
    assert isinstance(volatility_forecast, float), "Volatility forecast should be a float"
    assert volatility_forecast > 0, "Volatility forecast should be positive"
    assert volatility_forecast < 2.0, "Volatility forecast should be reasonable (< 200%)"
    
    # Test with insufficient data
    small_data = data.head(50)  # Less than minimum required
    small_volatility = await regime_detector._forecast_volatility("TEST/USDT", small_data)
    assert small_volatility > 0, "Should return fallback volatility for insufficient data"


@pytest.mark.asyncio
async def test_strategy_recommendation(regime_detector):
    """Test strategy recommendation based on regime."""
    
    # Test each regime
    regimes = ['bull', 'bear', 'sideways', 'volatile']
    
    for regime in regimes:
        strategy = regime_detector._get_strategy_recommendation(regime, 0.2)  # 20% volatility
        
        # Check strategy structure
        assert 'primary' in strategy, "Strategy should have primary component"
        assert 'risk_level' in strategy, "Strategy should have risk level"
        assert 'position_size' in strategy, "Strategy should have position size"
        assert 'stop_loss' in strategy, "Strategy should have stop loss"
        assert 'take_profit' in strategy, "Strategy should have take profit"
        assert 'volatility_forecast' in strategy, "Strategy should include volatility forecast"
        assert 'volatility_regime' in strategy, "Strategy should include volatility regime"
        
        # Check parameter bounds
        assert 0 < strategy['position_size'] <= 1, "Position size should be between 0 and 1"
        assert 0 < strategy['stop_loss'] < 1, "Stop loss should be between 0 and 1"
        assert 0 < strategy['take_profit'] < 1, "Take profit should be between 0 and 1"
        
        # Check regime-specific strategies
        if regime == 'bull':
            assert strategy['primary'] == 'trend_following', "Bull regime should use trend following"
        elif regime == 'bear':
            assert strategy['primary'] == 'short_selling', "Bear regime should use short selling"
        elif regime == 'sideways':
            assert strategy['primary'] == 'mean_reversion', "Sideways regime should use mean reversion"
        elif regime == 'volatile':
            assert strategy['primary'] == 'volatility_trading', "Volatile regime should use volatility trading"


@pytest.mark.asyncio
async def test_volatility_regime_classification(regime_detector):
    """Test volatility regime classification."""
    
    # Test different volatility levels
    low_vol = regime_detector._classify_volatility_regime(0.1)
    medium_vol = regime_detector._classify_volatility_regime(0.3)
    high_vol = regime_detector._classify_volatility_regime(0.5)
    
    assert low_vol == "low", "Low volatility should be classified as 'low'"
    assert medium_vol == "medium", "Medium volatility should be classified as 'medium'"
    assert high_vol == "high", "High volatility should be classified as 'high'"


@pytest.mark.asyncio
async def test_transition_probabilities(regime_detector):
    """Test regime transition probability calculation."""
    
    # Create a mock Gaussian Mixture Model
    from sklearn.mixture import GaussianMixture
    
    # Generate some test data for training
    np.random.seed(42)
    test_data = np.random.randn(100, 5)
    
    model = GaussianMixture(n_components=4, random_state=42)
    model.fit(test_data)
    
    # Get transition probabilities
    transition_probs = regime_detector._get_transition_probabilities(model)
    
    # Check structure
    assert len(transition_probs) == 4, "Should have 4 regimes"
    
    for from_regime in regime_detector.regime_labels.values():
        assert from_regime in transition_probs, f"Should have transitions from {from_regime}"
        
        # Check that probabilities sum to 1
        total_prob = sum(transition_probs[from_regime].values())
        assert abs(total_prob - 1.0) < 1e-6, f"Transition probabilities from {from_regime} should sum to 1"
        
        # Check that all probabilities are valid
        for to_regime, prob in transition_probs[from_regime].items():
            assert 0 <= prob <= 1, f"Transition probability should be between 0 and 1"


@pytest.mark.asyncio
async def test_confidence_score_calculation(regime_detector):
    """Test confidence score calculation."""
    
    # Test different scenarios
    high_conf = regime_detector._calculate_confidence_score(0.9, 300)  # High prob, sufficient data
    medium_conf = regime_detector._calculate_confidence_score(0.7, 150)  # Medium prob, some data
    low_conf = regime_detector._calculate_confidence_score(0.5, 50)     # Low prob, little data
    
    assert 0 <= high_conf <= 1, "Confidence should be between 0 and 1"
    assert 0 <= medium_conf <= 1, "Confidence should be between 0 and 1"
    assert 0 <= low_conf <= 1, "Confidence should be between 0 and 1"
    
    assert high_conf > medium_conf > low_conf, "Confidence should decrease with lower probability and less data"


@pytest.mark.asyncio
async def test_regime_history_update(regime_detector):
    """Test regime history tracking."""
    
    symbol = "BTC/USDT"
    
    # Initially no history
    assert symbol not in regime_detector.regime_history
    
    # Update history
    regime_detector._update_regime_history(symbol, "bull", 0.8)
    regime_detector._update_regime_history(symbol, "bull", 0.85)
    regime_detector._update_regime_history(symbol, "volatile", 0.7)
    
    # Check history
    assert symbol in regime_detector.regime_history
    history = regime_detector.regime_history[symbol]
    assert len(history) == 3, "Should have 3 history entries"
    
    # Check latest entry
    latest = history[-1]
    assert latest['regime'] == 'volatile', "Latest regime should be volatile"
    assert latest['probability'] == 0.7, "Latest probability should be 0.7"
    assert 'timestamp' in latest, "Should have timestamp"


@pytest.mark.asyncio
async def test_comprehensive_regime_detection(regime_detector):
    """Test the main regime detection function."""
    
    # Mock the data fetching to use controlled data
    mock_data = await regime_detector._get_historical_data("BTC/USDT", "1d")
    regime_detector._get_historical_data = AsyncMock(return_value=mock_data)
    
    # Test comprehensive regime detection
    result = await regime_detector.detect_regime("BTC/USDT", "1d")
    
    # Verify result structure
    assert "symbol" in result
    assert "current_regime" in result
    assert "regime_probability" in result
    assert "regime_history" in result
    assert "volatility_forecast" in result
    assert "strategy_recommendation" in result
    assert "regime_transition_matrix" in result
    assert "confidence_score" in result
    assert "timestamp" in result
    
    # Verify data types and bounds
    assert isinstance(result["current_regime"], str)
    assert result["current_regime"] in ['bull', 'bear', 'sideways', 'volatile']
    assert 0 <= result["regime_probability"] <= 1
    assert result["volatility_forecast"] > 0
    assert 0 <= result["confidence_score"] <= 1
    assert isinstance(result["regime_history"], list)
    assert isinstance(result["strategy_recommendation"], dict)
    assert isinstance(result["regime_transition_matrix"], dict)


@pytest.mark.asyncio
async def test_error_handling(regime_detector):
    """Test error handling in regime detection."""
    
    # Test with detector not ready
    regime_detector.ready = False
    
    with pytest.raises(RuntimeError, match="Regime detection system not initialized"):
        await regime_detector.detect_regime("BTC/USDT")
    
    # Reset detector state
    regime_detector.ready = True
    
    # Test with insufficient data
    insufficient_data = pd.DataFrame({
        'open': [100] * 10, 'high': [101] * 10, 'low': [99] * 10, 
        'close': [100.5] * 10, 'volume': [1000] * 10
    })
    
    regime_detector._get_historical_data = AsyncMock(return_value=insufficient_data)
    
    # Should return default response without raising error
    result = await regime_detector.detect_regime("BTC/USDT")
    assert result["current_regime"] == "sideways", "Should return default regime for insufficient data"
    assert result["confidence_score"] < 0.5, "Should have low confidence for insufficient data"


@pytest.mark.asyncio
async def test_regime_statistics(regime_detector):
    """Test regime statistics functionality."""
    
    symbol = "BTC/USDT"
    
    # Add some regime history
    regime_detector._update_regime_history(symbol, "bull", 0.8)
    regime_detector._update_regime_history(symbol, "bull", 0.85)
    regime_detector._update_regime_history(symbol, "volatile", 0.7)
    regime_detector._update_regime_history(symbol, "bear", 0.75)
    
    # Get statistics
    stats = regime_detector.get_regime_statistics(symbol)
    
    # Check statistics structure
    assert "symbol" in stats
    assert "total_observations" in stats
    assert "regime_distribution" in stats
    assert "recent_regimes" in stats
    
    # Check values
    assert stats["symbol"] == symbol
    assert stats["total_observations"] == 4
    assert "bull" in stats["regime_distribution"]
    assert stats["regime_distribution"]["bull"] == 2  # Two bull observations
    
    # Test global statistics
    global_stats = regime_detector.get_regime_statistics()
    assert "total_symbols" in global_stats
    assert "symbols" in global_stats
    assert global_stats["total_symbols"] == 1
    assert symbol in global_stats["symbols"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])