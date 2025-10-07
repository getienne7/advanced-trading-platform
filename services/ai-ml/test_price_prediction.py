"""
Test suite for the Price Prediction Engine.
Tests LSTM, Transformer models, and ensemble prediction system.
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from price_prediction import PricePredictionEngine


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.price_model_path = "test_models/price_prediction"
        self.sentiment_cache_ttl = 300
        self.prediction_cache_ttl = 900


@pytest.fixture
def price_engine():
    """Create a price prediction engine for testing."""
    config = MockConfig()
    engine = PricePredictionEngine(config)
    engine.ready = True
    return engine


@pytest.mark.asyncio
async def test_price_engine_initialization():
    """Test price prediction engine initialization."""
    config = MockConfig()
    engine = PricePredictionEngine(config)
    
    # Test that engine is created with correct configuration
    assert engine.config == config
    assert not engine.ready
    assert engine.lstm_config['sequence_length'] == 60
    assert engine.transformer_config['sequence_length'] == 100


@pytest.mark.asyncio
async def test_synthetic_data_generation(price_engine):
    """Test synthetic historical data generation."""
    
    # Test BTC data generation
    btc_data = await price_engine._get_historical_data("BTC/USDT", "1h")
    
    assert len(btc_data) == 1000, "Should generate 1000 data points"
    assert all(col in btc_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert all(btc_data['high'] >= btc_data['low']), "High should be >= Low"
    assert all(btc_data['high'] >= btc_data['close']), "High should be >= Close"
    assert all(btc_data['low'] <= btc_data['close']), "Low should be <= Close"
    assert all(btc_data['volume'] > 0), "Volume should be positive"
    
    # Test ETH data generation
    eth_data = await price_engine._get_historical_data("ETH/USDT", "1h")
    
    assert len(eth_data) == 1000, "Should generate 1000 data points"
    # ETH should have different base price than BTC
    assert abs(btc_data['close'].mean() - eth_data['close'].mean()) > 1000


@pytest.mark.asyncio
async def test_technical_indicators(price_engine):
    """Test technical indicator calculation."""
    
    # Generate test data
    data = await price_engine._get_historical_data("BTC/USDT", "1h")
    
    # Add technical indicators
    data_with_indicators = price_engine._add_technical_indicators(data)
    
    # Check that indicators were added
    expected_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower']
    for indicator in expected_indicators:
        assert indicator in data_with_indicators.columns, f"{indicator} should be added"
    
    # Check RSI bounds
    rsi_values = data_with_indicators['rsi'].dropna()
    assert all(0 <= rsi <= 100 for rsi in rsi_values), "RSI should be between 0 and 100"
    
    # Check Bollinger Bands relationship
    assert all(data_with_indicators['bb_upper'] >= data_with_indicators['bb_lower']), "BB upper should be >= BB lower"


@pytest.mark.asyncio
async def test_sequence_creation(price_engine):
    """Test sequence creation for time series models."""
    
    # Create test data
    data = np.random.rand(100, 5)  # 100 time steps, 5 features
    sequence_length = 10
    
    X, y = price_engine._create_sequences(data, sequence_length, target_col=-1)
    
    # Check shapes
    expected_samples = 100 - sequence_length
    assert X.shape == (expected_samples, sequence_length, 5), f"X shape should be ({expected_samples}, {sequence_length}, 5)"
    assert y.shape == (expected_samples,), f"y shape should be ({expected_samples},)"
    
    # Check that sequences are correct
    assert np.array_equal(X[0], data[0:sequence_length]), "First sequence should match first 10 rows"
    assert y[0] == data[sequence_length, -1], "First target should be close price at sequence_length"


@pytest.mark.asyncio
async def test_linear_prediction(price_engine):
    """Test linear regression baseline prediction."""
    
    # Generate test data with clear trend
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    prices = np.linspace(100, 200, 100)  # Clear upward trend
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 1, 100),
        'high': prices + np.random.normal(2, 1, 100),
        'low': prices + np.random.normal(-2, 1, 100),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    predictions, confidence_intervals = await price_engine._predict_with_linear(data, 24)
    
    # Check prediction structure
    expected_horizons = ['1h', '6h', '12h', '24h']
    for horizon in expected_horizons:
        assert horizon in predictions, f"{horizon} prediction should be present"
        assert horizon in confidence_intervals, f"{horizon} confidence interval should be present"
        assert len(confidence_intervals[horizon]) == 2, "Confidence interval should have lower and upper bounds"
    
    # Check that predictions follow the trend (should be increasing)
    current_price = data['close'].iloc[-1]
    assert predictions['24h'] > current_price, "24h prediction should be higher than current price (upward trend)"


@pytest.mark.asyncio
async def test_ensemble_prediction(price_engine):
    """Test ensemble prediction combination."""
    
    # Mock predictions from different models
    predictions = {
        'lstm': {'1h': 100.5, '6h': 102.0, '12h': 104.0, '24h': 106.0},
        'transformer': {'1h': 101.0, '6h': 103.0, '12h': 105.0, '24h': 107.0},
        'linear': {'1h': 99.5, '6h': 101.0, '12h': 103.0, '24h': 105.0}
    }
    
    ensemble_pred = price_engine._create_ensemble_prediction(predictions, "BTC/USDT")
    
    # Ensemble should be between min and max predictions
    min_pred = min(pred['24h'] for pred in predictions.values())
    max_pred = max(pred['24h'] for pred in predictions.values())
    
    assert min_pred <= ensemble_pred <= max_pred, "Ensemble prediction should be within range of individual predictions"
    
    # Test with empty predictions
    empty_ensemble = price_engine._create_ensemble_prediction({}, "BTC/USDT")
    assert empty_ensemble == 0.0, "Empty predictions should return 0.0"


@pytest.mark.asyncio
async def test_prediction_timeframes(price_engine):
    """Test prediction timeframe generation."""
    
    # Test different timeframes
    timeframes_1h = price_engine._generate_prediction_timeframes("1h", 24)
    timeframes_4h = price_engine._generate_prediction_timeframes("4h", 24)
    
    assert len(timeframes_1h) == 4, "Should generate 4 prediction timeframes"
    assert len(timeframes_4h) == 4, "Should generate 4 prediction timeframes"
    
    # Parse and check that times are in the future
    base_time = datetime.utcnow()
    for timeframe_str in timeframes_1h:
        pred_time = datetime.fromisoformat(timeframe_str.replace('Z', '+00:00'))
        assert pred_time > base_time, "Prediction time should be in the future"


@pytest.mark.asyncio
async def test_model_performance_tracking(price_engine):
    """Test model performance tracking."""
    
    symbol = "BTC/USDT"
    
    # Initially no performance data
    assert price_engine.get_model_performance(symbol) == {}
    
    # Add performance data
    price_engine.model_performance[symbol] = {
        'lstm': {'mse': 0.01, 'mae': 0.05, 'trained_at': datetime.utcnow().isoformat()},
        'transformer': {'mse': 0.008, 'mae': 0.04, 'trained_at': datetime.utcnow().isoformat()}
    }
    
    perf = price_engine.get_model_performance(symbol)
    assert 'lstm' in perf
    assert 'transformer' in perf
    assert perf['lstm']['mse'] == 0.01
    assert perf['transformer']['mse'] == 0.008


@pytest.mark.asyncio
async def test_ensemble_weights_update(price_engine):
    """Test ensemble weights updating."""
    
    symbol = "BTC/USDT"
    original_weights = price_engine.ensemble_weights.copy()
    
    # Update weights
    new_weights = {'lstm': 0.5, 'transformer': 0.3, 'linear': 0.2}
    price_engine.update_ensemble_weights(symbol, new_weights)
    
    # Check that weights were normalized and updated
    total_weight = sum(price_engine.ensemble_weights.values())
    assert abs(total_weight - 1.0) < 1e-6, "Weights should sum to 1.0"


@pytest.mark.asyncio
async def test_comprehensive_price_prediction(price_engine):
    """Test the main price prediction function."""
    
    # Mock the data fetching to avoid long computation
    mock_data = pd.DataFrame({
        'open': np.random.uniform(45000, 55000, 200),
        'high': np.random.uniform(46000, 56000, 200),
        'low': np.random.uniform(44000, 54000, 200),
        'close': np.random.uniform(45000, 55000, 200),
        'volume': np.random.uniform(1000, 10000, 200)
    })
    
    # Mock the historical data method
    price_engine._get_historical_data = AsyncMock(return_value=mock_data)
    
    # Mock the individual prediction methods to avoid training
    price_engine._predict_with_lstm = AsyncMock(return_value=(
        {'1h': 50100, '6h': 50500, '12h': 51000, '24h': 51500},
        {'1h': [49900, 50300], '6h': [50200, 50800], '12h': [50600, 51400], '24h': [51000, 52000]}
    ))
    
    price_engine._predict_with_transformer = AsyncMock(return_value=(
        {'1h': 50200, '6h': 50600, '12h': 51100, '24h': 51600},
        {'1h': [49950, 50450], '6h': [50250, 50950], '12h': [50650, 51550], '24h': [51050, 52150]}
    ))
    
    price_engine._predict_with_linear = AsyncMock(return_value=(
        {'1h': 50050, '6h': 50400, '12h': 50900, '24h': 51400},
        {'1h': [49850, 50250], '6h': [50100, 50700], '12h': [50500, 51300], '24h': [50900, 51900]}
    ))
    
    # Test comprehensive prediction
    result = await price_engine.predict_price("BTC/USDT", "1h", 24, ["lstm", "transformer", "linear"])
    
    # Verify result structure
    assert "symbol" in result
    assert "current_price" in result
    assert "predictions" in result
    assert "confidence_intervals" in result
    assert "ensemble_prediction" in result
    assert "prediction_times" in result
    assert "timestamp" in result
    
    # Verify predictions from all models
    assert "lstm" in result["predictions"]
    assert "transformer" in result["predictions"]
    assert "linear" in result["predictions"]
    
    # Verify confidence intervals
    assert "lstm" in result["confidence_intervals"]
    assert "transformer" in result["confidence_intervals"]
    assert "linear" in result["confidence_intervals"]
    
    # Verify ensemble prediction is reasonable
    assert isinstance(result["ensemble_prediction"], (int, float))
    assert result["ensemble_prediction"] > 0


@pytest.mark.asyncio
async def test_error_handling(price_engine):
    """Test error handling in price prediction."""
    
    # Test with engine not ready
    price_engine.ready = False
    
    with pytest.raises(RuntimeError, match="Price prediction engine not initialized"):
        await price_engine.predict_price("BTC/USDT")
    
    # Reset engine state
    price_engine.ready = True
    
    # Test with insufficient data
    insufficient_data = pd.DataFrame({
        'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
    })
    
    price_engine._get_historical_data = AsyncMock(return_value=insufficient_data)
    
    with pytest.raises(ValueError, match="Insufficient historical data"):
        await price_engine.predict_price("BTC/USDT")


@pytest.mark.asyncio
async def test_model_configurations(price_engine):
    """Test model configuration parameters."""
    
    # Test LSTM configuration
    lstm_config = price_engine.lstm_config
    assert lstm_config['sequence_length'] == 60
    assert len(lstm_config['features']) == 5
    assert len(lstm_config['lstm_units']) == 3
    assert 0 < lstm_config['dropout_rate'] < 1
    assert lstm_config['learning_rate'] > 0
    
    # Test Transformer configuration
    transformer_config = price_engine.transformer_config
    assert transformer_config['sequence_length'] == 100
    assert len(transformer_config['features']) == 9  # Including technical indicators
    assert transformer_config['embed_dim'] > 0
    assert transformer_config['num_heads'] > 0
    assert transformer_config['ff_dim'] > 0
    assert transformer_config['num_layers'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])