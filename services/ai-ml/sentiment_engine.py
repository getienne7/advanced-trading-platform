"""
Sentiment Analysis Engine for cryptocurrency market intelligence.
Integrates news APIs, social media, and provides sentiment scoring.
"""
import asyncio
import aiohttp
import feedparser
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
import praw
import structlog

logger = structlog.get_logger(__name__)

class SentimentAnalysisEngine:
    """Advanced sentiment analysis engine for cryptocurrency markets."""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        self.ready = False
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.twitter_api = None
        self.reddit_api = None
        
        # News API endpoints
        self.news_apis = {
            'coindesk': 'https://api.coindesk.com/v1/news/articles',
            'cointelegraph': 'https://cointelegraph.com/api/v1/content',
            'cryptonews': 'https://cryptonews.com/api/news',
            'newsapi': 'https://newsapi.org/v2/everything'
        }
        
        # RSS feeds for backup
        self.rss_feeds = [
            'https://cointelegraph.com/rss',
            'https://cryptonews.com/news/feed',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://bitcoinmagazine.com/.rss/full/',
            'https://decrypt.co/feed',
            'https://www.theblockcrypto.com/rss.xml'
        ]
        
        # Sentiment keywords and weights (enhanced)
        self.bullish_keywords = {
            # Price action
            'moon', 'bullish', 'pump', 'rally', 'surge', 'breakout', 'bull run', 'parabolic',
            'hodl', 'diamond hands', 'to the moon', 'green', 'profit', 'gains', 'mooning',
            'buy the dip', 'accumulate', 'strong support', 'resistance broken', 'all time high',
            
            # Adoption and development
            'institutional adoption', 'mainstream adoption', 'partnership', 'integration',
            'upgrade', 'innovation', 'breakthrough', 'positive', 'optimistic', 'bullish news',
            'mass adoption', 'enterprise', 'corporate', 'investment', 'fund', 'etf approved',
            
            # Technical indicators
            'golden cross', 'cup and handle', 'ascending triangle', 'bull flag',
            'higher highs', 'higher lows', 'uptrend', 'momentum', 'volume spike',
            
            # Market sentiment
            'fomo', 'euphoria', 'greed', 'confidence', 'strength', 'resilience'
        }
        
        self.bearish_keywords = {
            # Price action
            'bearish', 'dump', 'crash', 'correction', 'bear market', 'red', 'decline',
            'sell off', 'panic', 'fear', 'uncertainty', 'doubt', 'fud', 'capitulation',
            'paper hands', 'weak hands', 'resistance', 'support broken', 'death cross',
            
            # Negative events
            'regulation', 'ban', 'crackdown', 'hack', 'scam', 'bubble', 'ponzi',
            'overvalued', 'correction', 'decline', 'negative', 'pessimistic', 'bearish news',
            'lawsuit', 'investigation', 'fraud', 'manipulation', 'whale dump',
            
            # Technical indicators
            'death cross', 'head and shoulders', 'descending triangle', 'bear flag',
            'lower highs', 'lower lows', 'downtrend', 'breakdown', 'volume decline',
            
            # Market sentiment
            'capitulation', 'despair', 'fear', 'panic selling', 'blood bath', 'rekt'
        }
        
        # Crypto-specific sentiment modifiers (enhanced)
        self.crypto_multipliers = {
            'bitcoin': 1.3, 'btc': 1.3, '$btc': 1.3,
            'ethereum': 1.2, 'eth': 1.2, '$eth': 1.2,
            'altcoin': 0.9, 'altcoins': 0.9,
            'shitcoin': -0.8, 'shitcoins': -0.8,
            'defi': 1.1, 'nft': 0.7, 'nfts': 0.7,
            'whale': 1.2, 'whales': 1.2,
            'retail': 0.8, 'institutions': 1.3,
            'degen': -0.3, 'ape': -0.2, 'rekt': -1.0,
            'diamond hands': 1.5, 'paper hands': -1.2,
            'hodl': 1.1, 'hodling': 1.1,
            'lambo': 0.8, 'moon': 1.0, 'mars': 1.2
        }
        
        # Influence multipliers for different sources
        self.influence_multipliers = {
            'elon musk': 2.0, 'michael saylor': 1.8, 'cathie wood': 1.5,
            'vitalik buterin': 1.7, 'changpeng zhao': 1.6, 'brian armstrong': 1.4,
            'jack dorsey': 1.5, 'tim draper': 1.3, 'anthony pompliano': 1.2,
            'max keiser': 1.1, 'peter schiff': -1.2  # Peter Schiff is typically bearish on crypto
        }
    
    async def initialize(self):
        """Initialize the sentiment analysis engine."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Advanced-Trading-Platform/1.0',
                    'Accept': 'application/json, text/html, application/rss+xml'
                }
            )
            
            # Initialize Twitter API if credentials are available
            if self.config.twitter_bearer_token:
                try:
                    self.twitter_api = tweepy.Client(
                        bearer_token=self.config.twitter_bearer_token,
                        wait_on_rate_limit=True
                    )
                    logger.info("Twitter API initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Twitter API: {e}")
            
            # Initialize Reddit API if credentials are available
            if self.config.reddit_client_id and self.config.reddit_client_secret:
                try:
                    self.reddit_api = praw.Reddit(
                        client_id=self.config.reddit_client_id,
                        client_secret=self.config.reddit_client_secret,
                        user_agent='Advanced-Trading-Platform/1.0'
                    )
                    logger.info("Reddit API initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Reddit API: {e}")
            
            # Test API connections
            await self._test_connections()
            
            self.ready = True
            logger.info("Sentiment analysis engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment engine: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.ready = False
        logger.info("Sentiment analysis engine cleaned up")
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.ready
    
    async def analyze_sentiment(
        self,
        symbol: str,
        sources: List[str] = None,
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """Analyze sentiment for a given symbol from multiple sources."""
        
        if not self.ready:
            raise RuntimeError("Sentiment engine not initialized")
        
        if sources is None:
            sources = ["news", "twitter", "reddit"]
        
        logger.info("Starting sentiment analysis", symbol=symbol, sources=sources)
        
        # Collect data from all sources concurrently
        tasks = []
        if "news" in sources:
            tasks.append(self._analyze_news_sentiment(symbol, timeframe))
        if "twitter" in sources:
            tasks.append(self._analyze_twitter_sentiment(symbol, timeframe))
        if "reddit" in sources:
            tasks.append(self._analyze_reddit_sentiment(symbol, timeframe))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sentiment_data = {}
        for i, source in enumerate([s for s in ["news", "twitter", "reddit"] if s in sources]):
            if i < len(results) and not isinstance(results[i], Exception):
                sentiment_data[source] = results[i]
            else:
                logger.warning(f"Failed to get {source} sentiment", error=str(results[i]) if i < len(results) else "No result")
                sentiment_data[source] = {
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "count": 0,
                    "error": str(results[i]) if i < len(results) and isinstance(results[i], Exception) else "No data"
                }
        
        # Calculate overall sentiment
        overall_sentiment, confidence = self._calculate_overall_sentiment(sentiment_data)
        
        # Determine sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = "bullish"
        elif overall_sentiment < -0.2:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        return {
            "symbol": symbol,
            "overall_sentiment": round(overall_sentiment, 3),
            "sentiment_label": sentiment_label,
            "confidence": round(confidence, 3),
            "sources": sentiment_data,
            "timestamp": datetime.utcnow()
        }
    
    async def _analyze_news_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze sentiment from news sources."""
        try:
            articles = []
            
            # Try API endpoints first, then fall back to RSS feeds
            api_articles = await self._fetch_news_from_apis(symbol, timeframe)
            articles.extend(api_articles)
            
            # Get articles from RSS feeds as backup/supplement
            for feed_url in self.rss_feeds:
                try:
                    feed_articles = await self._fetch_rss_feed(feed_url, symbol, timeframe)
                    articles.extend(feed_articles)
                except Exception as e:
                    logger.warning(f"Failed to fetch from {feed_url}: {e}")
            
            if not articles:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "articles": []}
            
            # Remove duplicates based on title similarity
            articles = self._remove_duplicate_articles(articles)
            
            # Analyze sentiment of articles
            sentiments = []
            processed_articles = []
            
            for article in articles[:100]:  # Limit to 100 most recent articles
                try:
                    text = f"{article.get('title', '')} {article.get('summary', '')} {article.get('content', '')}"
                    sentiment_score = self._analyze_text_sentiment(text, symbol)
                    
                    # Weight by source credibility
                    source_weight = self._get_source_weight(article.get('source', ''))
                    weighted_sentiment = sentiment_score * source_weight
                    
                    sentiments.append(weighted_sentiment)
                    
                    processed_articles.append({
                        "title": article.get('title', ''),
                        "url": article.get('link', ''),
                        "published": article.get('published', ''),
                        "source": article.get('source', ''),
                        "sentiment": round(sentiment_score, 3),
                        "weighted_sentiment": round(weighted_sentiment, 3)
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze article sentiment: {e}")
            
            if not sentiments:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "articles": []}
            
            # Calculate weighted average sentiment and confidence
            avg_sentiment = np.mean(sentiments)
            confidence = min(1.0, len(sentiments) / 30)  # Higher confidence with more articles
            
            # Sort articles by sentiment strength for response
            processed_articles.sort(key=lambda x: abs(x['sentiment']), reverse=True)
            
            return {
                "sentiment": round(avg_sentiment, 3),
                "confidence": round(confidence, 3),
                "count": len(sentiments),
                "articles": processed_articles[:15]  # Return top 15 for response
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            raise
    
    async def _analyze_twitter_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze sentiment from Twitter/X."""
        try:
            if not self.config.twitter_bearer_token:
                logger.warning("Twitter bearer token not configured")
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "error": "No API token"}
            
            # Search for tweets about the symbol
            search_terms = self._get_search_terms(symbol)
            tweets = []
            
            for term in search_terms:
                try:
                    term_tweets = await self._fetch_tweets(term, timeframe)
                    tweets.extend(term_tweets)
                except Exception as e:
                    logger.warning(f"Failed to fetch tweets for {term}: {e}")
            
            if not tweets:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "tweets": []}
            
            # Analyze tweet sentiments
            sentiments = []
            processed_tweets = []
            
            for tweet in tweets[:100]:  # Limit to 100 most recent tweets
                try:
                    sentiment_score = self._analyze_text_sentiment(tweet.get('text', ''), symbol)
                    # Weight by engagement (likes, retweets)
                    engagement_weight = min(2.0, 1 + (tweet.get('public_metrics', {}).get('like_count', 0) / 100))
                    weighted_sentiment = sentiment_score * engagement_weight
                    
                    sentiments.append(weighted_sentiment)
                    processed_tweets.append({
                        "text": tweet.get('text', '')[:200],  # Truncate for response
                        "author": tweet.get('author_id', ''),
                        "created_at": tweet.get('created_at', ''),
                        "sentiment": round(sentiment_score, 3),
                        "engagement": tweet.get('public_metrics', {})
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze tweet sentiment: {e}")
            
            if not sentiments:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "tweets": []}
            
            # Calculate weighted average sentiment
            avg_sentiment = np.mean(sentiments)
            confidence = min(1.0, len(sentiments) / 50)  # Higher confidence with more tweets
            
            return {
                "sentiment": round(avg_sentiment, 3),
                "confidence": round(confidence, 3),
                "count": len(sentiments),
                "tweets": processed_tweets[:10]  # Return top 10 for response
            }
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            raise
    
    async def _analyze_reddit_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze sentiment from Reddit."""
        try:
            if not self.config.reddit_client_id or not self.config.reddit_client_secret:
                logger.warning("Reddit API credentials not configured")
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "error": "No API credentials"}
            
            # Get Reddit posts from crypto subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets', 'altcoin']
            posts = []
            
            for subreddit in subreddits:
                try:
                    subreddit_posts = await self._fetch_reddit_posts(subreddit, symbol, timeframe)
                    posts.extend(subreddit_posts)
                except Exception as e:
                    logger.warning(f"Failed to fetch from r/{subreddit}: {e}")
            
            if not posts:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "posts": []}
            
            # Analyze post sentiments
            sentiments = []
            processed_posts = []
            
            for post in posts[:50]:  # Limit to 50 most recent posts
                try:
                    text = f"{post.get('title', '')} {post.get('selftext', '')}"
                    sentiment_score = self._analyze_text_sentiment(text, symbol)
                    # Weight by upvotes and comments
                    engagement_weight = min(2.0, 1 + (post.get('score', 0) / 100) + (post.get('num_comments', 0) / 50))
                    weighted_sentiment = sentiment_score * engagement_weight
                    
                    sentiments.append(weighted_sentiment)
                    processed_posts.append({
                        "title": post.get('title', ''),
                        "subreddit": post.get('subreddit', ''),
                        "created_utc": post.get('created_utc', ''),
                        "sentiment": round(sentiment_score, 3),
                        "score": post.get('score', 0),
                        "num_comments": post.get('num_comments', 0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze Reddit post sentiment: {e}")
            
            if not sentiments:
                return {"sentiment": 0.0, "confidence": 0.0, "count": 0, "posts": []}
            
            # Calculate weighted average sentiment
            avg_sentiment = np.mean(sentiments)
            confidence = min(1.0, len(sentiments) / 30)  # Higher confidence with more posts
            
            return {
                "sentiment": round(avg_sentiment, 3),
                "confidence": round(confidence, 3),
                "count": len(sentiments),
                "posts": processed_posts[:10]  # Return top 10 for response
            }
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            raise
    
    def _analyze_text_sentiment(self, text: str, symbol: str) -> float:
        """Analyze sentiment of a text using multiple methods."""
        if not text:
            return 0.0
        
        original_text = text
        
        # Clean and normalize text
        text_clean = text.lower()
        text_clean = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text_clean)
        text_clean = re.sub(r'[^a-zA-Z0-9\s#$@]', ' ', text_clean)
        
        # VADER sentiment analysis (good for social media text)
        vader_scores = self.vader_analyzer.polarity_scores(original_text)
        vader_sentiment = vader_scores['compound']
        
        # TextBlob sentiment analysis
        blob = TextBlob(text_clean)
        textblob_sentiment = blob.sentiment.polarity
        
        # Keyword-based sentiment analysis
        keyword_sentiment = self._calculate_keyword_sentiment(text_clean)
        
        # Crypto-specific sentiment modifiers
        crypto_modifier = self._calculate_crypto_modifier(text_clean)
        
        # Influence modifier (check for influential people)
        influence_modifier = self._calculate_influence_modifier(text_clean)
        
        # Emoji sentiment (basic implementation)
        emoji_sentiment = self._calculate_emoji_sentiment(original_text)
        
        # Combine sentiments with weights
        combined_sentiment = (
            vader_sentiment * 0.35 +
            textblob_sentiment * 0.25 +
            keyword_sentiment * 0.25 +
            crypto_modifier * 0.1 +
            emoji_sentiment * 0.05
        )
        
        # Apply influence modifier
        combined_sentiment *= (1 + influence_modifier)
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, combined_sentiment))
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on keyword matching."""
        words = text.split()
        bullish_count = sum(1 for word in words if word in self.bullish_keywords)
        bearish_count = sum(1 for word in words if word in self.bearish_keywords)
        
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total_keywords
    
    def _calculate_crypto_modifier(self, text: str) -> float:
        """Calculate crypto-specific sentiment modifiers."""
        modifier = 0.0
        words = text.split()
        
        for word in words:
            if word in self.crypto_multipliers:
                modifier += self.crypto_multipliers[word] * 0.1
        
        return max(-0.5, min(0.5, modifier))
    
    def _calculate_overall_sentiment(self, sentiment_data: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate overall sentiment from multiple sources."""
        if not sentiment_data:
            return 0.0, 0.0
        
        # Source weights based on reliability and volume
        source_weights = {
            "news": 0.4,
            "twitter": 0.35,
            "reddit": 0.25
        }
        
        weighted_sentiments = []
        total_confidence = 0.0
        
        for source, data in sentiment_data.items():
            if isinstance(data, dict) and "sentiment" in data and "confidence" in data:
                weight = source_weights.get(source, 0.33)
                confidence = data["confidence"]
                sentiment = data["sentiment"]
                
                weighted_sentiments.append(sentiment * weight * confidence)
                total_confidence += confidence * weight
        
        if not weighted_sentiments or total_confidence == 0:
            return 0.0, 0.0
        
        overall_sentiment = sum(weighted_sentiments) / total_confidence
        overall_confidence = min(1.0, total_confidence)
        
        return overall_sentiment, overall_confidence
    
    def _get_search_terms(self, symbol: str) -> List[str]:
        """Get search terms for a trading symbol."""
        base_symbol = symbol.split('/')[0].upper()
        
        # Common cryptocurrency names and symbols
        crypto_names = {
            'BTC': ['bitcoin', 'btc', '$btc'],
            'ETH': ['ethereum', 'eth', '$eth'],
            'ADA': ['cardano', 'ada', '$ada'],
            'DOT': ['polkadot', 'dot', '$dot'],
            'LINK': ['chainlink', 'link', '$link'],
            'UNI': ['uniswap', 'uni', '$uni'],
            'AAVE': ['aave', '$aave'],
            'SUSHI': ['sushiswap', 'sushi', '$sushi']
        }
        
        terms = crypto_names.get(base_symbol, [base_symbol.lower(), f'${base_symbol.lower()}'])
        return terms[:3]  # Limit to 3 terms to avoid rate limits
    
    async def _test_connections(self):
        """Test API connections."""
        try:
            # Test RSS feed access
            async with self.session.get(self.config.cointelegraph_rss) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed test failed: {response.status}")
            
            logger.info("API connections tested successfully")
            
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")
    
    async def _fetch_rss_feed(self, feed_url: str, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed."""
        try:
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                # Filter articles by timeframe and relevance
                cutoff_time = self._get_cutoff_time(timeframe)
                search_terms = self._get_search_terms(symbol)
                
                relevant_articles = []
                for entry in feed.entries:
                    # Check if article is recent enough
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        article_time = datetime(*entry.published_parsed[:6])
                        if article_time < cutoff_time:
                            continue
                    
                    # Check if article is relevant to the symbol
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    content = f"{title} {summary}"
                    
                    if any(term in content for term in search_terms):
                        relevant_articles.append({
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'summary': entry.get('summary', ''),
                            'published': entry.get('published', '')
                        })
                
                return relevant_articles
                
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    async def _fetch_tweets(self, search_term: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetch tweets using Twitter API v2."""
        if not self.twitter_api:
            logger.warning("Twitter API not initialized")
            return []
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - self._get_timedelta(timeframe)
            
            # Search for tweets
            query = f"{search_term} -is:retweet lang:en"
            tweets = []
            
            # Use asyncio to run the synchronous tweepy call
            def fetch_tweets_sync():
                try:
                    return self.twitter_api.search_recent_tweets(
                        query=query,
                        max_results=100,
                        start_time=start_time,
                        end_time=end_time,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                        user_fields=['username', 'verified', 'public_metrics']
                    )
                except Exception as e:
                    logger.error(f"Twitter API error: {e}")
                    return None
            
            # Run in thread pool to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(fetch_tweets_sync)
                response = future.result(timeout=30)
            
            if response and response.data:
                for tweet in response.data:
                    tweets.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                        'author_id': tweet.author_id,
                        'public_metrics': tweet.public_metrics or {}
                    })
            
            logger.info(f"Fetched {len(tweets)} tweets for {search_term}")
            return tweets
            
        except Exception as e:
            logger.error(f"Failed to fetch tweets for {search_term}: {e}")
            return []
    
    async def _fetch_reddit_posts(self, subreddit: str, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetch Reddit posts using Reddit API."""
        if not self.reddit_api:
            logger.warning("Reddit API not initialized")
            return []
        
        try:
            # Calculate time range
            cutoff_time = self._get_cutoff_time(timeframe)
            search_terms = self._get_search_terms(symbol)
            
            posts = []
            
            # Use asyncio to run the synchronous praw calls
            def fetch_posts_sync():
                try:
                    subreddit_obj = self.reddit_api.subreddit(subreddit)
                    
                    # Get hot posts
                    hot_posts = list(subreddit_obj.hot(limit=50))
                    
                    # Get new posts
                    new_posts = list(subreddit_obj.new(limit=50))
                    
                    # Combine and filter posts
                    all_posts = hot_posts + new_posts
                    relevant_posts = []
                    
                    for post in all_posts:
                        # Check if post is recent enough
                        post_time = datetime.utcfromtimestamp(post.created_utc)
                        if post_time < cutoff_time:
                            continue
                        
                        # Check if post is relevant to the symbol
                        title = post.title.lower()
                        selftext = post.selftext.lower()
                        content = f"{title} {selftext}"
                        
                        if any(term in content for term in search_terms):
                            relevant_posts.append({
                                'id': post.id,
                                'title': post.title,
                                'selftext': post.selftext,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc,
                                'subreddit': post.subreddit.display_name,
                                'author': str(post.author) if post.author else '[deleted]',
                                'url': post.url,
                                'permalink': f"https://reddit.com{post.permalink}"
                            })
                    
                    return relevant_posts
                    
                except Exception as e:
                    logger.error(f"Reddit API error for r/{subreddit}: {e}")
                    return []
            
            # Run in thread pool to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(fetch_posts_sync)
                posts = future.result(timeout=30)
            
            logger.info(f"Fetched {len(posts)} relevant posts from r/{subreddit} for {symbol}")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch Reddit posts from r/{subreddit}: {e}")
            return []
    
    def _calculate_influence_modifier(self, text: str) -> float:
        """Calculate influence modifier based on influential people mentioned."""
        modifier = 0.0
        words = text.split()
        
        # Check for influential people (case insensitive)
        text_lower = text.lower()
        for person, multiplier in self.influence_multipliers.items():
            if person in text_lower:
                modifier += multiplier * 0.1
        
        return max(-0.5, min(0.5, modifier))
    
    def _calculate_emoji_sentiment(self, text: str) -> float:
        """Calculate sentiment based on emojis (basic implementation)."""
        # Positive emojis
        positive_emojis = ['🚀', '🌙', '💎', '🙌', '💰', '📈', '🔥', '💪', '🎉', '✅', '👍', '😊', '😎', '🤑']
        # Negative emojis  
        negative_emojis = ['📉', '💸', '😭', '😱', '🤮', '💩', '👎', '😢', '😰', '🔻', '❌', '⚠️']
        
        positive_count = sum(1 for emoji in positive_emojis if emoji in text)
        negative_count = sum(1 for emoji in negative_emojis if emoji in text)
        
        total_emojis = positive_count + negative_count
        if total_emojis == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_emojis * 0.3  # Scale down emoji impact
    
    def _get_timedelta(self, timeframe: str) -> timedelta:
        """Get timedelta object based on timeframe string."""
        if timeframe == "1h":
            return timedelta(hours=1)
        elif timeframe == "4h":
            return timedelta(hours=4)
        elif timeframe == "12h":
            return timedelta(hours=12)
        elif timeframe == "24h":
            return timedelta(hours=24)
        elif timeframe == "7d":
            return timedelta(days=7)
        else:
            return timedelta(hours=24)  # Default to 24h
    
    def _get_cutoff_time(self, timeframe: str) -> datetime:
        """Get cutoff time based on timeframe."""
        now = datetime.utcnow()
        return now - self._get_timedelta(timeframe)
    
    async def _fetch_news_from_apis(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Fetch news from various news APIs."""
        articles = []
        search_terms = self._get_search_terms(symbol)
        
        # CoinDesk API (if available)
        try:
            coindesk_articles = await self._fetch_coindesk_news(search_terms, timeframe)
            articles.extend(coindesk_articles)
        except Exception as e:
            logger.warning(f"Failed to fetch CoinDesk news: {e}")
        
        # NewsAPI (if API key available)
        if hasattr(self.config, 'news_api_key') and self.config.news_api_key:
            try:
                newsapi_articles = await self._fetch_newsapi_articles(search_terms, timeframe)
                articles.extend(newsapi_articles)
            except Exception as e:
                logger.warning(f"Failed to fetch NewsAPI articles: {e}")
        
        return articles
    
    async def _fetch_coindesk_news(self, search_terms: List[str], timeframe: str) -> List[Dict[str, Any]]:
        """Fetch news from CoinDesk API."""
        articles = []
        
        try:
            # CoinDesk doesn't have a public API for articles, so we'll use RSS
            # This is a placeholder for when they might have an API
            return []
        except Exception as e:
            logger.error(f"CoinDesk API error: {e}")
            return []
    
    async def _fetch_newsapi_articles(self, search_terms: List[str], timeframe: str) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI."""
        articles = []
        
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - self._get_timedelta(timeframe)
            
            for term in search_terms[:2]:  # Limit to avoid rate limits
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'{term} cryptocurrency',
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 50,
                    'apiKey': self.config.news_api_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            articles.append({
                                'title': article.get('title', ''),
                                'summary': article.get('description', ''),
                                'content': article.get('content', ''),
                                'link': article.get('url', ''),
                                'published': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', 'NewsAPI')
                            })
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return articles
    
    def _remove_duplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = []
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            if not title:
                continue
            
            # Normalize title for better comparison
            normalized_title = re.sub(r'[^\w\s]', ' ', title)  # Remove punctuation
            normalized_title = re.sub(r'\s+', ' ', normalized_title)  # Normalize whitespace
            title_words = set(normalized_title.split())
            
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_normalized = re.sub(r'[^\w\s]', ' ', seen_title)
                seen_normalized = re.sub(r'\s+', ' ', seen_normalized)
                seen_words = set(seen_normalized.split())
                
                # Calculate Jaccard similarity
                intersection = len(title_words & seen_words)
                union = len(title_words | seen_words)
                
                if union > 0:
                    similarity = intersection / union
                    # If 70% similarity or more, consider it a duplicate
                    if similarity >= 0.7:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.append(title)
        
        return unique_articles
    
    def _get_source_weight(self, source: str) -> float:
        """Get credibility weight for news source."""
        source_weights = {
            'coindesk': 1.2,
            'cointelegraph': 1.1,
            'decrypt': 1.1,
            'the block': 1.2,
            'bitcoin magazine': 1.0,
            'cryptonews': 0.9,
            'newsapi': 0.8,
            'unknown': 0.7
        }
        
        source_lower = source.lower()
        for known_source, weight in source_weights.items():
            if known_source in source_lower:
                return weight
        
        return source_weights['unknown']