"""
AI/ML Market Intelligence Service for Advanced Trading Platform.
Handles sentiment analysis, price prediction, and market regime detection.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from database import get_postgres_session, get_redis
from utils import setup_logging

# Configure logging
logger = setup_logging("ai-ml-service")

# Metrics
SENTIMENT_ANALYSIS_COUNT = Counter('sentiment_analysis_total', 'Total sentiment analyses', ['source', 'symbol'])
PREDICTION_COUNT = Counter('price_predictions_total', 'Total price predictions', ['model', 'symbol'])
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time')

# Configuration
class AIMLConfig:
    """AI/ML service configuration."""
    
    def __init__(self):
        # News API configurations
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.coindesk_api_url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        self.cointelegraph_rss = "https://cointelegraph.com/rss"
        self.cryptonews_rss = "https://cryptonews.com/news/feed"
        
        # Social media API configurations
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        
        # Model configurations
        self.sentiment_model_path = os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment")
        self.price_model_path = os.getenv("PRICE_MODEL_PATH", "models/price_prediction")
        self.regime_model_path = os.getenv("REGIME_MODEL_PATH", "models/regime_detection")
        
        # Cache configurations
        self.sentiment_cache_ttl = int(os.getenv("SENTIMENT_CACHE_TTL", "300"))  # 5 minutes
        self.prediction_cache_ttl = int(os.getenv("PREDICTION_CACHE_TTL", "900"))  # 15 minutes

config = AIMLConfig()

# Pydantic models
class SentimentRequest(BaseModel):
    """Sentiment analysis request model."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTC/USDT)")
    sources: List[str] = Field(default=["news", "twitter", "reddit"], description="Data sources to analyze")
    timeframe: str = Field(default="24h", description="Time frame for analysis")

class SentimentResponse(BaseModel):
    """Sentiment analysis response model."""
    symbol: str
    overall_sentiment: float = Field(..., description="Overall sentiment score (-1 to 1)")
    sentiment_label: str = Field(..., description="Sentiment label (bearish/neutral/bullish)")
    confidence: float = Field(..., description="Confidence score (0 to 1)")
    sources: Dict[str, Dict[str, Any]] = Field(..., description="Sentiment by source")
    timestamp: datetime
    
class PricePredictionRequest(BaseModel):
    """Price prediction request model."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(default="1h", description="Prediction timeframe")
    horizon: int = Field(default=24, description="Prediction horizon in hours")
    models: List[str] = Field(default=["lstm", "transformer"], description="Models to use")

class PricePredictionResponse(BaseModel):
    """Price prediction response model."""
    symbol: str
    current_price: float
    predictions: Dict[str, Dict[str, float]] = Field(..., description="Predictions by model and timeframe")
    confidence_intervals: Dict[str, Dict[str, List[float]]] = Field(..., description="Confidence intervals")
    ensemble_prediction: float = Field(..., description="Ensemble model prediction")
    timestamp: datetime

class MarketRegimeResponse(BaseModel):
    """Market regime detection response model."""
    symbol: str
    current_regime: str = Field(..., description="Current market regime")
    regime_probability: float = Field(..., description="Probability of current regime")
    regime_history: List[Dict[str, Any]] = Field(..., description="Historical regime changes")
    volatility_forecast: float = Field(..., description="Forecasted volatility")
    timestamp: datetime

# FastAPI app
app = FastAPI(
    title="AI/ML Market Intelligence Service",
    description="AI-powered market analysis and prediction service",
    version="1.0.0"
)

# Import AI/ML components
from sentiment_engine import SentimentAnalysisEngine
from price_prediction import PricePredictionEngine
from regime_detection import MarketRegimeDetector
from mlops_pipeline import MLOpsManager
from mlops_service import create_mlops_service
from model_scheduler import create_model_scheduler

# Initialize engines
sentiment_engine = SentimentAnalysisEngine(config)
prediction_engine = PricePredictionEngine(config)
regime_detector = MarketRegimeDetector(config)

# Initialize MLOps components
mlops_manager = MLOpsManager(config)
mlops_service = create_mlops_service(config)
model_scheduler = create_model_scheduler(config, mlops_manager)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "sentiment": sentiment_engine.is_ready(),
            "prediction": prediction_engine.is_ready(),
            "regime": regime_detector.is_ready()
        },
        "mlops": {
            "manager": "ready",
            "scheduler": "ready" if model_scheduler else "not_initialized"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(generate_latest(), media_type="text/plain")

# Sentiment Analysis Endpoints
@app.post("/api/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks
):
    """Analyze market sentiment for a given symbol."""
    try:
        # Check cache first
        redis = await get_redis()
        cache_key = f"sentiment:{request.symbol}:{':'.join(request.sources)}:{request.timeframe}"
        cached_result = await redis.get(cache_key)
        
        if cached_result:
            import json
            logger.info("Returning cached sentiment analysis", symbol=request.symbol)
            return SentimentResponse(**json.loads(cached_result))
        
        # Perform sentiment analysis
        logger.info("Performing sentiment analysis", symbol=request.symbol, sources=request.sources)
        
        with MODEL_INFERENCE_TIME.time():
            sentiment_result = await sentiment_engine.analyze_sentiment(
                symbol=request.symbol,
                sources=request.sources,
                timeframe=request.timeframe
            )
        
        # Update metrics
        for source in request.sources:
            SENTIMENT_ANALYSIS_COUNT.labels(source=source, symbol=request.symbol).inc()
        
        # Cache result
        background_tasks.add_task(
            cache_sentiment_result,
            cache_key,
            sentiment_result,
            config.sentiment_cache_ttl
        )
        
        return sentiment_result
        
    except Exception as e:
        logger.error("Sentiment analysis failed", error=str(e), symbol=request.symbol)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/api/sentiment/{symbol}", response_model=SentimentResponse)
async def get_sentiment(symbol: str):
    """Get latest sentiment analysis for a symbol."""
    request = SentimentRequest(symbol=symbol)
    return await analyze_sentiment(request, BackgroundTasks())

# Price Prediction Endpoints
@app.post("/api/predictions/price", response_model=PricePredictionResponse)
async def predict_price(
    request: PricePredictionRequest,
    background_tasks: BackgroundTasks
):
    """Generate price predictions for a given symbol."""
    try:
        # Check cache first
        redis = await get_redis()
        cache_key = f"prediction:{request.symbol}:{request.timeframe}:{request.horizon}"
        cached_result = await redis.get(cache_key)
        
        if cached_result:
            import json
            logger.info("Returning cached price prediction", symbol=request.symbol)
            return PricePredictionResponse(**json.loads(cached_result))
        
        # Generate predictions
        logger.info("Generating price predictions", symbol=request.symbol, models=request.models)
        
        with MODEL_INFERENCE_TIME.time():
            prediction_result = await prediction_engine.predict_price(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                models=request.models
            )
        
        # Update metrics
        for model in request.models:
            PREDICTION_COUNT.labels(model=model, symbol=request.symbol).inc()
        
        # Cache result
        background_tasks.add_task(
            cache_prediction_result,
            cache_key,
            prediction_result,
            config.prediction_cache_ttl
        )
        
        return prediction_result
        
    except Exception as e:
        logger.error("Price prediction failed", error=str(e), symbol=request.symbol)
        raise HTTPException(status_code=500, detail=f"Price prediction failed: {str(e)}")

@app.get("/api/predictions/{symbol}", response_model=PricePredictionResponse)
async def get_price_prediction(symbol: str):
    """Get latest price prediction for a symbol."""
    request = PricePredictionRequest(symbol=symbol)
    return await predict_price(request, BackgroundTasks())

# Market Regime Detection Endpoints
@app.get("/api/regime/{symbol}", response_model=MarketRegimeResponse)
async def detect_market_regime(symbol: str):
    """Detect current market regime for a symbol."""
    try:
        logger.info("Detecting market regime", symbol=symbol)
        
        with MODEL_INFERENCE_TIME.time():
            regime_result = await regime_detector.detect_regime(symbol=symbol)
        
        return regime_result
        
    except Exception as e:
        logger.error("Market regime detection failed", error=str(e), symbol=symbol)
        raise HTTPException(status_code=500, detail=f"Market regime detection failed: {str(e)}")

# Combined Analysis Endpoint
@app.get("/api/analysis/{symbol}")
async def comprehensive_analysis(symbol: str):
    """Get comprehensive AI analysis including sentiment, predictions, and regime."""
    try:
        logger.info("Performing comprehensive analysis", symbol=symbol)
        
        # Run all analyses concurrently
        sentiment_task = analyze_sentiment(SentimentRequest(symbol=symbol), BackgroundTasks())
        prediction_task = predict_price(PricePredictionRequest(symbol=symbol), BackgroundTasks())
        regime_task = detect_market_regime(symbol)
        
        sentiment_result, prediction_result, regime_result = await asyncio.gather(
            sentiment_task, prediction_task, regime_task
        )
        
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": sentiment_result,
            "price_prediction": prediction_result,
            "market_regime": regime_result,
            "trading_signal": generate_trading_signal(sentiment_result, prediction_result, regime_result)
        }
        
    except Exception as e:
        logger.error("Comprehensive analysis failed", error=str(e), symbol=symbol)
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

# Helper functions
async def cache_sentiment_result(cache_key: str, result: SentimentResponse, ttl: int):
    """Cache sentiment analysis result."""
    try:
        redis = await get_redis()
        import json
        await redis.setex(cache_key, ttl, json.dumps(result.dict(), default=str))
    except Exception as e:
        logger.error("Failed to cache sentiment result", error=str(e))

async def cache_prediction_result(cache_key: str, result: PricePredictionResponse, ttl: int):
    """Cache price prediction result."""
    try:
        redis = await get_redis()
        import json
        await redis.setex(cache_key, ttl, json.dumps(result.dict(), default=str))
    except Exception as e:
        logger.error("Failed to cache prediction result", error=str(e))

def generate_trading_signal(
    sentiment: SentimentResponse,
    prediction: PricePredictionResponse,
    regime: MarketRegimeResponse
) -> Dict[str, Any]:
    """Generate trading signal based on AI analysis."""
    
    # Calculate signal strength based on multiple factors
    sentiment_weight = 0.3
    prediction_weight = 0.4
    regime_weight = 0.3
    
    # Sentiment signal (-1 to 1)
    sentiment_signal = sentiment.overall_sentiment
    
    # Prediction signal (-1 to 1)
    price_change = (prediction.ensemble_prediction - prediction.current_price) / prediction.current_price
    prediction_signal = max(-1, min(1, price_change * 10))  # Scale to -1 to 1
    
    # Regime signal (-1 to 1)
    regime_signals = {
        "bull": 1.0,
        "bear": -1.0,
        "sideways": 0.0,
        "volatile": 0.0
    }
    regime_signal = regime_signals.get(regime.current_regime.lower(), 0.0)
    
    # Combined signal
    combined_signal = (
        sentiment_signal * sentiment_weight +
        prediction_signal * prediction_weight +
        regime_signal * regime_weight
    )
    
    # Determine action
    if combined_signal > 0.3:
        action = "BUY"
        strength = "STRONG" if combined_signal > 0.6 else "MODERATE"
    elif combined_signal < -0.3:
        action = "SELL"
        strength = "STRONG" if combined_signal < -0.6 else "MODERATE"
    else:
        action = "HOLD"
        strength = "NEUTRAL"
    
    return {
        "action": action,
        "strength": strength,
        "signal_score": round(combined_signal, 3),
        "components": {
            "sentiment": round(sentiment_signal, 3),
            "prediction": round(prediction_signal, 3),
            "regime": round(regime_signal, 3)
        },
        "confidence": min(sentiment.confidence, 0.8),  # Conservative confidence
        "timestamp": datetime.utcnow().isoformat()
    }

# MLOps endpoints
@app.mount("/mlops", mlops_service.app)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("AI/ML service starting up...")
    
    # Initialize database connections
    from database import initialize_databases
    await initialize_databases()
    
    # Initialize AI/ML engines
    await sentiment_engine.initialize()
    await prediction_engine.initialize()
    await regime_detector.initialize()
    
    # Start MLOps scheduler
    await model_scheduler.start()
    
    logger.info("AI/ML service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("AI/ML service shutting down...")
    
    # Stop MLOps scheduler
    await model_scheduler.stop()
    
    # Cleanup AI/ML engines
    await sentiment_engine.cleanup()
    await prediction_engine.cleanup()
    await regime_detector.cleanup()
    
    # Close database connections
    from database import cleanup_databases
    await cleanup_databases()
    
    logger.info("AI/ML service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )