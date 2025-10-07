"""
Test suite for the Sentiment Analysis Engine.
Tests news API integration, social media sentiment analysis, and sentiment scoring.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from sentiment_engine import SentimentAnalysisEngine


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.news_api_key = "test_news_api_key"
        self.coindesk_api_url = "https://api.coindesk.com/v1/news/articles"
        self.cointelegraph_rss = "https://cointelegraph.com/rss"
        self.cryptonews_rss = "https://cryptonews.com/news/feed"
        
        self.twitter_bearer_token = "test_twitter_token"
        self.reddit_client_id = "test_reddit_id"
        self.reddit_client_secret = "test_reddit_secret"
        
        self.sentiment_model_path = "models/sentiment"
        self.price_model_path = "models/price_prediction"
        self.regime_model_path = "models/regime_detection"
        
        self.sentiment_cache_ttl = 300
        self.prediction_cache_ttl = 900


@pytest.fixture
def sentiment_engine():
    """Create a sentiment analysis engine for testing."""
    config = MockConfig()
    engine = SentimentAnalysisEngine(config)
    
    # Mock the session to avoid actual HTTP calls
    engine.session = AsyncMock()
    
    # Mock API clients
    engine.twitter_api = Mock()
    engine.reddit_api = Mock()
    
    engine.ready = True
    return engine


@pytest.mark.asyncio
async def test_sentiment_engine_initialization():
    """Test sentiment engine initialization."""
    config = MockConfig()
    engine = SentimentAnalysisEngine(config)
    
    # Test that engine is created with correct configuration
    assert engine.config == config
    assert not engine.ready
    assert engine.session is None


@pytest.mark.asyncio
async def test_text_sentiment_analysis(sentiment_engine):
    """Test text sentiment analysis with various inputs."""
    
    # Test bullish text
    bullish_text = "Bitcoin is going to the moon! 🚀 This is a great investment opportunity with strong fundamentals."
    bullish_sentiment = sentiment_engine._analyze_text_sentiment(bullish_text, "BTC/USDT")
    assert bullish_sentiment > 0, "Bullish text should have positive sentiment"
    
    # Test bearish text
    bearish_text = "Bitcoin is crashing! This is a bubble and will dump to zero. Panic selling everywhere 📉"
    bearish_sentiment = sentiment_engine._analyze_text_sentiment(bearish_text, "BTC/USDT")
    assert bearish_sentiment < 0, "Bearish text should have negative sentiment"
    
    # Test neutral text
    neutral_text = "Bitcoin price is currently at $50,000. The market is stable today."
    neutral_sentiment = sentiment_engine._analyze_text_sentiment(neutral_text, "BTC/USDT")
    assert abs(neutral_sentiment) < 0.3, "Neutral text should have low sentiment score"


@pytest.mark.asyncio
async def test_keyword_sentiment_calculation(sentiment_engine):
    """Test keyword-based sentiment calculation."""
    
    # Test bullish keywords
    bullish_text = "bullish rally pump moon hodl diamond hands"
    bullish_score = sentiment_engine._calculate_keyword_sentiment(bullish_text)
    assert bullish_score > 0, "Bullish keywords should produce positive score"
    
    # Test bearish keywords
    bearish_text = "bearish dump crash correction fud panic sell"
    bearish_score = sentiment_engine._calculate_keyword_sentiment(bearish_text)
    assert bearish_score < 0, "Bearish keywords should produce negative score"
    
    # Test mixed keywords
    mixed_text = "bullish pump but also bearish dump"
    mixed_score = sentiment_engine._calculate_keyword_sentiment(mixed_text)
    assert abs(mixed_score) < 0.5, "Mixed keywords should produce moderate score"


@pytest.mark.asyncio
async def test_crypto_modifier_calculation(sentiment_engine):
    """Test crypto-specific sentiment modifiers."""
    
    # Test Bitcoin modifier
    btc_text = "bitcoin btc is the future"
    btc_modifier = sentiment_engine._calculate_crypto_modifier(btc_text)
    assert btc_modifier > 0, "Bitcoin mentions should have positive modifier"
    
    # Test negative crypto terms
    negative_text = "shitcoin scam rekt paper hands"
    negative_modifier = sentiment_engine._calculate_crypto_modifier(negative_text)
    assert negative_modifier < 0, "Negative crypto terms should have negative modifier"


@pytest.mark.asyncio
async def test_influence_modifier_calculation(sentiment_engine):
    """Test influence modifier for influential people."""
    
    # Test Elon Musk mention
    elon_text = "elon musk just tweeted about bitcoin"
    elon_modifier = sentiment_engine._calculate_influence_modifier(elon_text)
    assert elon_modifier > 0, "Elon Musk mention should have positive influence modifier"
    
    # Test Peter Schiff mention (bearish on crypto)
    schiff_text = "peter schiff says bitcoin is worthless"
    schiff_modifier = sentiment_engine._calculate_influence_modifier(schiff_text)
    assert schiff_modifier < 0, "Peter Schiff mention should have negative influence modifier"


@pytest.mark.asyncio
async def test_emoji_sentiment_calculation(sentiment_engine):
    """Test emoji-based sentiment calculation."""
    
    # Test positive emojis
    positive_text = "Bitcoin to the moon! 🚀🌙💎🙌"
    positive_score = sentiment_engine._calculate_emoji_sentiment(positive_text)
    assert positive_score > 0, "Positive emojis should produce positive score"
    
    # Test negative emojis
    negative_text = "Bitcoin is crashing 📉💸😭🤮"
    negative_score = sentiment_engine._calculate_emoji_sentiment(negative_text)
    assert negative_score < 0, "Negative emojis should produce negative score"


@pytest.mark.asyncio
async def test_search_terms_generation(sentiment_engine):
    """Test search terms generation for different symbols."""
    
    # Test Bitcoin
    btc_terms = sentiment_engine._get_search_terms("BTC/USDT")
    assert "bitcoin" in btc_terms
    assert "btc" in btc_terms
    assert "$btc" in btc_terms
    
    # Test Ethereum
    eth_terms = sentiment_engine._get_search_terms("ETH/USDT")
    assert "ethereum" in eth_terms
    assert "eth" in eth_terms
    
    # Test unknown symbol
    unknown_terms = sentiment_engine._get_search_terms("UNKNOWN/USDT")
    assert "unknown" in unknown_terms
    assert "$unknown" in unknown_terms


@pytest.mark.asyncio
async def test_source_weight_calculation(sentiment_engine):
    """Test source credibility weighting."""
    
    # Test high credibility sources
    coindesk_weight = sentiment_engine._get_source_weight("CoinDesk")
    assert coindesk_weight > 1.0, "CoinDesk should have high credibility weight"
    
    # Test medium credibility sources
    cointelegraph_weight = sentiment_engine._get_source_weight("Cointelegraph")
    assert cointelegraph_weight >= 1.0, "Cointelegraph should have good credibility weight"
    
    # Test unknown sources
    unknown_weight = sentiment_engine._get_source_weight("Unknown Source")
    assert unknown_weight < 1.0, "Unknown sources should have lower credibility weight"


@pytest.mark.asyncio
async def test_duplicate_article_removal(sentiment_engine):
    """Test duplicate article removal."""
    
    articles = [
        {"title": "Bitcoin reaches new all-time high", "content": "Bitcoin price surged today"},
        {"title": "Bitcoin reaches new all time high", "content": "BTC price increased significantly"},  # Similar title
        {"title": "Ethereum launches new upgrade", "content": "ETH network gets major update"},
        {"title": "Bitcoin price analysis", "content": "Technical analysis of BTC"}
    ]
    
    unique_articles = sentiment_engine._remove_duplicate_articles(articles)
    
    # Should remove the duplicate Bitcoin article
    assert len(unique_articles) == 3, "Should remove duplicate articles"
    
    # Check that the Ethereum and Bitcoin analysis articles remain
    titles = [article["title"] for article in unique_articles]
    assert any("ethereum" in title.lower() for title in titles)
    assert any("analysis" in title.lower() for title in titles)


@pytest.mark.asyncio
async def test_overall_sentiment_calculation(sentiment_engine):
    """Test overall sentiment calculation from multiple sources."""
    
    sentiment_data = {
        "news": {
            "sentiment": 0.6,
            "confidence": 0.8,
            "count": 25
        },
        "twitter": {
            "sentiment": 0.3,
            "confidence": 0.7,
            "count": 100
        },
        "reddit": {
            "sentiment": -0.2,
            "confidence": 0.6,
            "count": 15
        }
    }
    
    overall_sentiment, confidence = sentiment_engine._calculate_overall_sentiment(sentiment_data)
    
    # Should be positive but moderated by Reddit negativity
    assert overall_sentiment > 0, "Overall sentiment should be positive"
    assert overall_sentiment < 0.6, "Overall sentiment should be moderated"
    assert 0 < confidence <= 1, "Confidence should be between 0 and 1"


@pytest.mark.asyncio
async def test_timeframe_parsing(sentiment_engine):
    """Test timeframe parsing and cutoff time calculation."""
    
    # Test various timeframes
    timeframes = ["1h", "4h", "12h", "24h", "7d"]
    
    for timeframe in timeframes:
        cutoff_time = sentiment_engine._get_cutoff_time(timeframe)
        timedelta_obj = sentiment_engine._get_timedelta(timeframe)
        
        assert isinstance(cutoff_time, datetime), "Cutoff time should be datetime object"
        assert isinstance(timedelta_obj, timedelta), "Should return timedelta object"
        assert cutoff_time < datetime.utcnow(), "Cutoff time should be in the past"


@pytest.mark.asyncio
async def test_comprehensive_sentiment_analysis(sentiment_engine):
    """Test the main sentiment analysis function."""
    
    # Mock the individual analysis methods
    sentiment_engine._analyze_news_sentiment = AsyncMock(return_value={
        "sentiment": 0.5,
        "confidence": 0.8,
        "count": 20,
        "articles": []
    })
    
    sentiment_engine._analyze_twitter_sentiment = AsyncMock(return_value={
        "sentiment": 0.3,
        "confidence": 0.7,
        "count": 50,
        "tweets": []
    })
    
    sentiment_engine._analyze_reddit_sentiment = AsyncMock(return_value={
        "sentiment": 0.1,
        "confidence": 0.6,
        "count": 10,
        "posts": []
    })
    
    # Test comprehensive analysis
    result = await sentiment_engine.analyze_sentiment("BTC/USDT", ["news", "twitter", "reddit"], "24h")
    
    # Verify result structure
    assert "symbol" in result
    assert "overall_sentiment" in result
    assert "sentiment_label" in result
    assert "confidence" in result
    assert "sources" in result
    assert "timestamp" in result
    
    # Verify sentiment label
    assert result["sentiment_label"] in ["bullish", "bearish", "neutral"]
    
    # Verify all sources are included
    assert "news" in result["sources"]
    assert "twitter" in result["sources"]
    assert "reddit" in result["sources"]


@pytest.mark.asyncio
async def test_error_handling(sentiment_engine):
    """Test error handling in sentiment analysis."""
    
    # Test with engine not ready
    sentiment_engine.ready = False
    
    with pytest.raises(RuntimeError, match="Sentiment engine not initialized"):
        await sentiment_engine.analyze_sentiment("BTC/USDT")
    
    # Reset engine state
    sentiment_engine.ready = True
    
    # Test with failing news analysis
    sentiment_engine._analyze_news_sentiment = AsyncMock(side_effect=Exception("News API error"))
    sentiment_engine._analyze_twitter_sentiment = AsyncMock(return_value={
        "sentiment": 0.2, "confidence": 0.5, "count": 10
    })
    sentiment_engine._analyze_reddit_sentiment = AsyncMock(return_value={
        "sentiment": 0.1, "confidence": 0.4, "count": 5
    })
    
    # Should still work with other sources
    result = await sentiment_engine.analyze_sentiment("BTC/USDT")
    assert "sources" in result
    assert "error" in result["sources"]["news"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])