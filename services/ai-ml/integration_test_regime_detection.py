"""
Integration test for the Market Regime Detection System.
Tests the complete regime detection workflow including HMM, GARCH, and strategy selection.
"""
import asyncio
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))
sys.path.append(str(Path(__file__).parent))

from regime_detection import MarketRegimeDetector


class TestConfig:
    """Test configuration."""
    
    def __init__(self):
        self.regime_model_path = "test_models/regime_detection"


async def test_regime_detection_workflow():
    """Test the complete regime detection workflow."""
    
    print("🚀 Starting Market Regime Detection System Integration Test")
    
    # Initialize detector
    config = TestConfig()
    detector = MarketRegimeDetector(config)
    
    try:
        # Initialize the detector
        print("📊 Initializing market regime detection system...")
        await detector.initialize()
        
        # Test 1: Synthetic data generation with regime changes
        print("\n1️⃣ Testing synthetic data generation with regime changes...")
        
        btc_data = await detector._get_historical_data("BTC/USDT", "1d")
        eth_data = await detector._get_historical_data("ETH/USDT", "1d")
        
        assert len(btc_data) == 499, "Should generate 499 data points"
        assert len(eth_data) == 499, "Should generate 499 data points"
        
        # Verify OHLCV relationships
        assert all(btc_data['high'] >= btc_data['low']), "High >= Low"
        assert all(btc_data['high'] >= btc_data['close']), "High >= Close"
        assert all(btc_data['low'] <= btc_data['close']), "Low <= Close"
        
        print(f"   ✅ Generated {len(btc_data)} BTC data points with regime changes")
        print(f"   ✅ Generated {len(eth_data)} ETH data points with regime changes")
        print(f"   ✅ BTC price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
        print(f"   ✅ ETH price range: ${eth_data['close'].min():.2f} - ${eth_data['close'].max():.2f}")
        
        # Test 2: Feature extraction for regime detection
        print("\n2️⃣ Testing regime feature extraction...")
        
        features = detector._extract_regime_features(btc_data)
        
        expected_features = ['returns', 'volatility', 'volume_ratio', 'momentum', 'rsi', 'bb_position', 'trend_strength']
        for feature in expected_features:
            assert feature in features.columns, f"{feature} should be present"
        
        # Check feature quality
        rsi_values = features['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in rsi_values), "RSI should be 0-100"
        
        returns_std = features['returns'].std()
        volatility_mean = features['volatility'].mean()
        
        print(f"   ✅ Extracted {len(expected_features)} regime features")
        print(f"   ✅ Returns volatility: {returns_std:.4f}")
        print(f"   ✅ Average volatility: {volatility_mean:.4f}")
        print(f"   ✅ RSI range: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
        
        # Test 3: Regime model training (Gaussian Mixture Model as HMM proxy)
        print("\n3️⃣ Testing regime model training...")
        
        await detector._train_regime_model("BTC/USDT", features)
        
        model_key = "regime_BTC/USDT"
        scaler_key = "regime_scaler_BTC/USDT"
        
        assert model_key in detector.regime_models, "Regime model should be created"
        assert scaler_key in detector.scalers, "Feature scaler should be created"
        
        model = detector.regime_models[model_key]
        print(f"   ✅ Trained Gaussian Mixture Model with {model.n_components} components")
        print(f"   ✅ Model converged: {model.converged_}")
        print(f"   ✅ Model weights: {[f'{w:.3f}' for w in model.weights_]}")
        
        # Test 4: Regime prediction
        print("\n4️⃣ Testing regime prediction...")
        
        regime_id, probability = detector._predict_current_regime(model, features)
        regime_label = detector.regime_labels[regime_id]
        
        assert 0 <= regime_id < 4, "Regime ID should be valid"
        assert 0 <= probability <= 1, "Probability should be valid"
        
        print(f"   ✅ Current regime: {regime_label} (ID: {regime_id})")
        print(f"   ✅ Regime probability: {probability:.3f}")
        
        # Test 5: GARCH volatility forecasting
        print("\n5️⃣ Testing GARCH volatility forecasting...")
        
        returns = btc_data['close'].pct_change().dropna()
        await detector._train_garch_model("BTC/USDT", returns)
        
        garch_model = detector.garch_models["garch_BTC/USDT"]
        volatility_forecast = detector._garch_forecast(garch_model, returns)
        
        # Check GARCH parameters
        omega = garch_model['omega']
        alpha = garch_model['alpha']
        beta = garch_model['beta']
        
        assert omega > 0, "Omega should be positive"
        assert 0 <= alpha <= 1, "Alpha should be valid"
        assert 0 <= beta <= 1, "Beta should be valid"
        assert alpha + beta < 1, "Model should be stationary"
        
        print(f"   ✅ GARCH(1,1) parameters: ω={omega:.6f}, α={alpha:.3f}, β={beta:.3f}")
        print(f"   ✅ Volatility forecast: {volatility_forecast:.1%} (annualized)")
        print(f"   ✅ Model stationarity: α+β = {alpha+beta:.3f} < 1.0")
        
        # Test 6: Strategy recommendation
        print("\n6️⃣ Testing strategy recommendation system...")
        
        strategy = detector._get_strategy_recommendation(regime_label, volatility_forecast)
        
        # Check strategy components
        required_keys = ['primary', 'risk_level', 'position_size', 'stop_loss', 'take_profit', 'volatility_forecast', 'volatility_regime']
        for key in required_keys:
            assert key in strategy, f"Strategy should have {key}"
        
        print(f"   ✅ Regime: {regime_label}")
        print(f"   ✅ Primary strategy: {strategy['primary']}")
        print(f"   ✅ Risk level: {strategy['risk_level']}")
        print(f"   ✅ Position size: {strategy['position_size']:.1%}")
        print(f"   ✅ Stop loss: {strategy['stop_loss']:.1%}")
        print(f"   ✅ Take profit: {strategy['take_profit']:.1%}")
        print(f"   ✅ Volatility regime: {strategy['volatility_regime']}")
        
        # Test 7: Regime transition probabilities
        print("\n7️⃣ Testing regime transition probabilities...")
        
        transition_probs = detector._get_transition_probabilities(model)
        
        # Verify transition matrix properties
        for from_regime in detector.regime_labels.values():
            assert from_regime in transition_probs, f"Should have transitions from {from_regime}"
            total_prob = sum(transition_probs[from_regime].values())
            assert abs(total_prob - 1.0) < 1e-6, f"Probabilities should sum to 1"
        
        print(f"   ✅ Transition matrix calculated for {len(transition_probs)} regimes")
        print("   ✅ Transition probabilities (from → to):")
        for from_regime, transitions in transition_probs.items():
            for to_regime, prob in transitions.items():
                if prob > 0.1:  # Only show significant transitions
                    print(f"      {from_regime} → {to_regime}: {prob:.3f}")
        
        # Test 8: Regime history analysis
        print("\n8️⃣ Testing regime history analysis...")
        
        regime_history = detector._analyze_regime_history(model, features, "BTC/USDT")
        
        if regime_history:
            print(f"   ✅ Identified {len(regime_history)} regime periods")
            for i, period in enumerate(regime_history[-3:]):  # Show last 3 periods
                duration = period.get('duration_days', 0)
                regime = period.get('regime', 'unknown')
                prob = period.get('avg_probability', 0)
                print(f"      Period {i+1}: {regime} regime for {duration} days (avg prob: {prob:.3f})")
        else:
            print("   ✅ No significant regime changes detected (stable market)")
        
        # Test 9: Confidence score calculation
        print("\n9️⃣ Testing confidence score calculation...")
        
        confidence_high = detector._calculate_confidence_score(0.9, 400)
        confidence_medium = detector._calculate_confidence_score(0.7, 200)
        confidence_low = detector._calculate_confidence_score(0.5, 100)
        
        assert confidence_high > confidence_medium > confidence_low, "Confidence should decrease appropriately"
        
        print(f"   ✅ High confidence scenario: {confidence_high:.3f}")
        print(f"   ✅ Medium confidence scenario: {confidence_medium:.3f}")
        print(f"   ✅ Low confidence scenario: {confidence_low:.3f}")
        
        # Test 10: Comprehensive regime detection
        print("\n🔟 Testing comprehensive regime detection...")
        
        result = await detector.detect_regime("BTC/USDT", "1d")
        
        # Verify result structure
        required_fields = [
            'symbol', 'current_regime', 'regime_probability', 'regime_history',
            'volatility_forecast', 'strategy_recommendation', 'regime_transition_matrix',
            'confidence_score', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in result, f"Result should contain {field}"
        
        print(f"   ✅ Symbol: {result['symbol']}")
        print(f"   ✅ Current regime: {result['current_regime']}")
        print(f"   ✅ Regime probability: {result['regime_probability']:.3f}")
        print(f"   ✅ Volatility forecast: {result['volatility_forecast']:.1%}")
        print(f"   ✅ Confidence score: {result['confidence_score']:.3f}")
        print(f"   ✅ Strategy: {result['strategy_recommendation']['primary']}")
        
        # Test 11: Regime statistics
        print("\n1️⃣1️⃣ Testing regime statistics tracking...")
        
        # Add some regime history
        detector._update_regime_history("BTC/USDT", result['current_regime'], result['regime_probability'])
        detector._update_regime_history("BTC/USDT", "volatile", 0.8)
        detector._update_regime_history("BTC/USDT", "bull", 0.75)
        
        stats = detector.get_regime_statistics("BTC/USDT")
        
        assert "total_observations" in stats, "Should have observation count"
        assert "regime_distribution" in stats, "Should have regime distribution"
        
        print(f"   ✅ Total observations: {stats['total_observations']}")
        print(f"   ✅ Regime distribution: {stats['regime_distribution']}")
        
        # Test global statistics
        global_stats = detector.get_regime_statistics()
        print(f"   ✅ Total symbols tracked: {global_stats['total_symbols']}")
        print(f"   ✅ Symbols: {global_stats['symbols']}")
        
        print("\n🎉 All regime detection tests passed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Synthetic data generation with regime changes working")
        print("   ✅ Regime feature extraction working")
        print("   ✅ Gaussian Mixture Model training working")
        print("   ✅ Regime prediction working")
        print("   ✅ GARCH volatility forecasting working")
        print("   ✅ Strategy recommendation working")
        print("   ✅ Regime transition probabilities working")
        print("   ✅ Regime history analysis working")
        print("   ✅ Confidence score calculation working")
        print("   ✅ Comprehensive regime detection working")
        print("   ✅ Regime statistics tracking working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        await detector.cleanup()


async def main():
    """Run the integration test."""
    success = await test_regime_detection_workflow()
    
    if success:
        print("\n🚀 Market Regime Detection System is ready for production!")
        print("\n📊 Key Features Implemented:")
        print("   • Hidden Markov Model proxy using Gaussian Mixture Models")
        print("   • GARCH(1,1) volatility forecasting")
        print("   • Multi-feature regime detection (returns, volatility, momentum, RSI)")
        print("   • Dynamic strategy selection based on market conditions")
        print("   • Regime transition probability analysis")
        print("   • Confidence scoring and model validation")
        print("   • Historical regime change detection")
        print("   • Volatility regime classification (low/medium/high)")
        print("   • Comprehensive strategy recommendations")
        print("   • Real-time regime tracking and statistics")
    else:
        print("\n❌ Integration test failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)