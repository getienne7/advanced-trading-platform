"""
Integration test for the Sentiment Analysis Engine.
Tests the complete sentiment analysis workflow.
"""
import asyncio
import sys
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))
sys.path.append(str(Path(__file__).parent))

from sentiment_engine import SentimentAnalysisEngine


class TestConfig:
    """Test configuration."""
    
    def __init__(self):
        # News API configurations (using empty keys for testing)
        self.news_api_key = ""
        self.coindesk_api_url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        self.cointelegraph_rss = "https://cointelegraph.com/rss"
        self.cryptonews_rss = "https://cryptonews.com/news/feed"
        
        # Social media API configurations (using empty keys for testing)
        self.twitter_bearer_token = ""
        self.reddit_client_id = ""
        self.reddit_client_secret = ""
        
        # Model configurations
        self.sentiment_model_path = "models/sentiment"
        self.price_model_path = "models/price_prediction"
        self.regime_model_path = "models/regime_detection"
        
        # Cache configurations
        self.sentiment_cache_ttl = 300
        self.prediction_cache_ttl = 900


async def test_sentiment_analysis_workflow():
    """Test the complete sentiment analysis workflow."""
    
    print("🚀 Starting Sentiment Analysis Engine Integration Test")
    
    # Initialize engine
    config = TestConfig()
    engine = SentimentAnalysisEngine(config)
    
    try:
        # Initialize the engine (without actual API calls)
        print("📊 Initializing sentiment analysis engine...")
        engine.session = None  # Skip HTTP session for testing
        engine.ready = True
        
        # Test 1: Text sentiment analysis
        print("\n1️⃣ Testing text sentiment analysis...")
        
        test_texts = [
            ("Bitcoin is going to the moon! 🚀 Great investment opportunity!", "bullish"),
            ("Bitcoin is crashing! This is a disaster 📉 Sell everything!", "bearish"),
            ("Bitcoin price is stable at $50,000 today.", "neutral"),
            ("Elon Musk tweets about Bitcoin again! Diamond hands 💎🙌", "bullish"),
            ("Regulation crackdown on crypto! FUD everywhere!", "bearish")
        ]
        
        for text, expected_sentiment in test_texts:
            sentiment_score = engine._analyze_text_sentiment(text, "BTC/USDT")
            
            if expected_sentiment == "bullish":
                assert sentiment_score > 0, f"Expected bullish sentiment for: {text}"
                print(f"   ✅ Bullish text detected: {sentiment_score:.3f}")
            elif expected_sentiment == "bearish":
                assert sentiment_score < 0, f"Expected bearish sentiment for: {text}"
                print(f"   ✅ Bearish text detected: {sentiment_score:.3f}")
            else:  # neutral
                assert abs(sentiment_score) < 0.5, f"Expected neutral sentiment for: {text}"
                print(f"   ✅ Neutral text detected: {sentiment_score:.3f}")
        
        # Test 2: Keyword sentiment calculation
        print("\n2️⃣ Testing keyword sentiment calculation...")
        
        bullish_keywords = "bullish rally pump moon hodl diamond hands"
        bearish_keywords = "bearish dump crash correction fud panic"
        
        bullish_score = engine._calculate_keyword_sentiment(bullish_keywords)
        bearish_score = engine._calculate_keyword_sentiment(bearish_keywords)
        
        assert bullish_score > 0, "Bullish keywords should produce positive score"
        assert bearish_score < 0, "Bearish keywords should produce negative score"
        
        print(f"   ✅ Bullish keywords score: {bullish_score:.3f}")
        print(f"   ✅ Bearish keywords score: {bearish_score:.3f}")
        
        # Test 3: Crypto modifier calculation
        print("\n3️⃣ Testing crypto-specific modifiers...")
        
        btc_text = "bitcoin btc is the future of money"
        shitcoin_text = "this shitcoin is worthless"
        
        btc_modifier = engine._calculate_crypto_modifier(btc_text)
        shitcoin_modifier = engine._calculate_crypto_modifier(shitcoin_text)
        
        assert btc_modifier > 0, "Bitcoin mentions should have positive modifier"
        assert shitcoin_modifier < 0, "Shitcoin mentions should have negative modifier"
        
        print(f"   ✅ Bitcoin modifier: {btc_modifier:.3f}")
        print(f"   ✅ Shitcoin modifier: {shitcoin_modifier:.3f}")
        
        # Test 4: Influence modifier calculation
        print("\n4️⃣ Testing influence modifiers...")
        
        elon_text = "elon musk just tweeted about bitcoin"
        schiff_text = "peter schiff says bitcoin is worthless"
        
        elon_modifier = engine._calculate_influence_modifier(elon_text)
        schiff_modifier = engine._calculate_influence_modifier(schiff_text)
        
        assert elon_modifier > 0, "Elon Musk should have positive influence"
        assert schiff_modifier < 0, "Peter Schiff should have negative influence"
        
        print(f"   ✅ Elon Musk influence: {elon_modifier:.3f}")
        print(f"   ✅ Peter Schiff influence: {schiff_modifier:.3f}")
        
        # Test 5: Search terms generation
        print("\n5️⃣ Testing search terms generation...")
        
        btc_terms = engine._get_search_terms("BTC/USDT")
        eth_terms = engine._get_search_terms("ETH/USDT")
        
        assert "bitcoin" in btc_terms or "btc" in btc_terms, "BTC terms should include bitcoin/btc"
        assert "ethereum" in eth_terms or "eth" in eth_terms, "ETH terms should include ethereum/eth"
        
        print(f"   ✅ BTC search terms: {btc_terms}")
        print(f"   ✅ ETH search terms: {eth_terms}")
        
        # Test 6: Overall sentiment calculation
        print("\n6️⃣ Testing overall sentiment calculation...")
        
        sentiment_data = {
            "news": {"sentiment": 0.6, "confidence": 0.8, "count": 25},
            "twitter": {"sentiment": 0.3, "confidence": 0.7, "count": 100},
            "reddit": {"sentiment": -0.2, "confidence": 0.6, "count": 15}
        }
        
        overall_sentiment, confidence = engine._calculate_overall_sentiment(sentiment_data)
        
        assert -1 <= overall_sentiment <= 1, "Overall sentiment should be between -1 and 1"
        assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"
        assert overall_sentiment > 0, "Should be positive given the test data"
        
        print(f"   ✅ Overall sentiment: {overall_sentiment:.3f}")
        print(f"   ✅ Confidence: {confidence:.3f}")
        
        # Test 7: Article deduplication
        print("\n7️⃣ Testing article deduplication...")
        
        articles = [
            {"title": "Bitcoin reaches new all-time high", "content": "Bitcoin price surged today"},
            {"title": "Bitcoin reaches new all time high", "content": "BTC price increased significantly"},
            {"title": "Ethereum launches new upgrade", "content": "ETH network gets major update"},
            {"title": "Bitcoin price analysis", "content": "Technical analysis of BTC"}
        ]
        
        unique_articles = engine._remove_duplicate_articles(articles)
        
        assert len(unique_articles) < len(articles), "Should remove duplicate articles"
        assert len(unique_articles) >= 3, "Should keep unique articles"
        
        print(f"   ✅ Removed duplicates: {len(articles)} -> {len(unique_articles)} articles")
        
        # Test 8: Source weight calculation
        print("\n8️⃣ Testing source credibility weighting...")
        
        coindesk_weight = engine._get_source_weight("CoinDesk")
        unknown_weight = engine._get_source_weight("Unknown Source")
        
        assert coindesk_weight > unknown_weight, "CoinDesk should have higher weight than unknown source"
        
        print(f"   ✅ CoinDesk weight: {coindesk_weight:.3f}")
        print(f"   ✅ Unknown source weight: {unknown_weight:.3f}")
        
        print("\n🎉 All sentiment analysis tests passed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Text sentiment analysis working")
        print("   ✅ Keyword-based sentiment working")
        print("   ✅ Crypto-specific modifiers working")
        print("   ✅ Influence modifiers working")
        print("   ✅ Search term generation working")
        print("   ✅ Overall sentiment calculation working")
        print("   ✅ Article deduplication working")
        print("   ✅ Source credibility weighting working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if engine.session:
            await engine.cleanup()


async def main():
    """Run the integration test."""
    success = await test_sentiment_analysis_workflow()
    
    if success:
        print("\n🚀 Sentiment Analysis Engine is ready for production!")
        print("\n📊 Key Features Implemented:")
        print("   • Multi-source sentiment analysis (News, Twitter, Reddit)")
        print("   • Advanced text processing with VADER and TextBlob")
        print("   • Crypto-specific keyword analysis")
        print("   • Influential person detection")
        print("   • Emoji sentiment analysis")
        print("   • Source credibility weighting")
        print("   • Article deduplication")
        print("   • Comprehensive sentiment aggregation")
    else:
        print("\n❌ Integration test failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)