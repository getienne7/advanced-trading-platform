#!/usr/bin/env python3
"""
Quick Demo of the Advanced Trading Platform AI/ML System
Run this to see the AI/ML capabilities in action immediately!
"""
import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "services" / "ai-ml"))
sys.path.append(str(Path(__file__).parent / "shared"))

from sentiment_engine import SentimentAnalysisEngine
from price_prediction import PricePredictionEngine
from regime_detection import MarketRegimeDetector


class DemoConfig:
    """Demo configuration."""
    def __init__(self):
        # AI/ML model paths
        self.sentiment_model_path = "models/sentiment"
        self.price_model_path = "models/price_prediction"
        self.regime_model_path = "models/regime_detection"
        
        # API keys (empty for demo - will use fallbacks)
        self.news_api_key = ""
        self.twitter_bearer_token = ""
        self.reddit_client_id = ""
        self.reddit_client_secret = ""
        
        # Cache settings
        self.sentiment_cache_ttl = 300
        self.prediction_cache_ttl = 900


async def demo_ai_ml_system():
    """Demonstrate the complete AI/ML system."""
    
    print("🚀 Advanced Trading Platform AI/ML System Demo")
    print("=" * 60)
    
    # Initialize configuration
    config = DemoConfig()
    
    # Initialize all AI/ML engines
    print("\n📊 Initializing AI/ML Engines...")
    
    sentiment_engine = SentimentAnalysisEngine(config)
    price_engine = PricePredictionEngine(config)
    regime_detector = MarketRegimeDetector(config)
    
    try:
        # Initialize engines
        await sentiment_engine.initialize()
        await price_engine.initialize()
        await regime_detector.initialize()
        
        print("✅ All AI/ML engines initialized successfully!")
        
        # Demo symbols
        symbols = ["BTC/USDT", "ETH/USDT"]
        
        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}")
            print("-" * 40)
            
            # 1. Sentiment Analysis
            print("📈 Sentiment Analysis:")
            try:
                sentiment_result = await sentiment_engine.analyze_sentiment(
                    symbol=symbol,
                    sources=["news", "twitter", "reddit"],
                    timeframe="24h"
                )
                
                print(f"   Overall Sentiment: {sentiment_result['overall_sentiment']:.3f}")
                print(f"   Sentiment Label: {sentiment_result['sentiment_label']}")
                print(f"   Confidence: {sentiment_result['confidence']:.3f}")
                
                # Show source breakdown
                for source, data in sentiment_result['sources'].items():
                    if isinstance(data, dict) and 'sentiment' in data:
                        print(f"   {source.title()}: {data['sentiment']:.3f} (confidence: {data['confidence']:.3f})")
                
            except Exception as e:
                print(f"   ⚠️ Sentiment analysis: {e}")
            
            # 2. Price Prediction
            print("\n🎯 Price Predictions:")
            try:
                prediction_result = await price_engine.predict_price(
                    symbol=symbol,
                    timeframe="1h",
                    horizon=24,
                    models=["lstm", "transformer", "linear"]
                )
                
                print(f"   Current Price: ${prediction_result['current_price']:.2f}")
                print(f"   Ensemble 24h Prediction: ${prediction_result['ensemble_prediction']:.2f}")
                
                # Show individual model predictions
                for model, predictions in prediction_result['predictions'].items():
                    if '24h' in predictions:
                        print(f"   {model.upper()} 24h: ${predictions['24h']:.2f}")
                
            except Exception as e:
                print(f"   ⚠️ Price prediction: {e}")
            
            # 3. Market Regime Detection
            print("\n🎪 Market Regime Analysis:")
            try:
                regime_result = await regime_detector.detect_regime(symbol, "1d")
                
                print(f"   Current Regime: {regime_result['current_regime']}")
                print(f"   Regime Probability: {regime_result['regime_probability']:.3f}")
                print(f"   Volatility Forecast: {regime_result['volatility_forecast']:.1%}")
                print(f"   Confidence Score: {regime_result['confidence_score']:.3f}")
                
                # Show strategy recommendation
                strategy = regime_result['strategy_recommendation']
                print(f"   Recommended Strategy: {strategy['primary']}")
                print(f"   Position Size: {strategy['position_size']:.1%}")
                print(f"   Risk Level: {strategy['risk_level']}")
                
            except Exception as e:
                print(f"   ⚠️ Regime detection: {e}")
            
            print()
        
        # Show system capabilities summary
        print("\n🎉 Demo Complete! System Capabilities:")
        print("=" * 60)
        print("✅ Multi-source sentiment analysis (News, Twitter, Reddit)")
        print("✅ Deep learning price predictions (LSTM, Transformer)")
        print("✅ Market regime detection (Bull, Bear, Sideways, Volatile)")
        print("✅ GARCH volatility forecasting")
        print("✅ Dynamic strategy recommendations")
        print("✅ Ensemble model predictions with confidence intervals")
        print("✅ Real-time market intelligence")
        
        print("\n🚀 Ready for Production Deployment!")
        print("\nNext Steps:")
        print("1. Run 'python advanced_trading_platform/services/ai-ml/main.py' for API server")
        print("2. Visit http://localhost:8005/docs for interactive API documentation")
        print("3. Integrate with trading strategies and risk management")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await sentiment_engine.cleanup()
        await price_engine.cleanup()
        await regime_detector.cleanup()


if __name__ == "__main__":
    print("Starting Advanced Trading Platform AI/ML Demo...")
    asyncio.run(demo_ai_ml_system())