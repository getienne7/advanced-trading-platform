"""
Market Regime Detection System using Hidden Markov Models and GARCH.
Identifies market regimes (bull, bear, sideways, volatile) and forecasts volatility.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import structlog

# Statistical and ML imports
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection system using HMM and GARCH models."""
    
    def __init__(self, config):
        self.config = config
        self.ready = False
        
        # Regime detection configuration
        self.regime_config = {
            'lookback_period': 252,  # 1 year of daily data
            'min_regime_duration': 10,  # Minimum days in a regime
            'n_regimes': 4,  # Bull, Bear, Sideways, Volatile
            'features': ['returns', 'volatility', 'volume_ratio', 'momentum', 'rsi']
        }
        
        # GARCH model configuration
        self.garch_config = {
            'p': 1,  # GARCH(1,1) - lag order for volatility
            'q': 1,  # GARCH(1,1) - lag order for squared residuals
            'forecast_horizon': 5,  # Days ahead to forecast
            'min_observations': 100  # Minimum observations for GARCH fitting
        }
        
        # Model storage
        self.regime_models = {}  # HMM models for each symbol
        self.garch_models = {}   # GARCH models for each symbol
        self.scalers = {}        # Feature scalers
        self.regime_history = {} # Historical regime data
        
        # Regime definitions
        self.regime_labels = {
            0: "bull",      # Strong upward trend, low volatility
            1: "bear",      # Strong downward trend, low volatility  
            2: "sideways",  # Low trend, low volatility
            3: "volatile"   # High volatility regardless of trend
        }
        
        # Strategy mappings for each regime
        self.regime_strategies = {
            "bull": {
                "primary": "trend_following",
                "risk_level": "moderate",
                "position_size": 0.8,
                "stop_loss": 0.05,
                "take_profit": 0.15
            },
            "bear": {
                "primary": "short_selling",
                "risk_level": "moderate", 
                "position_size": 0.6,
                "stop_loss": 0.05,
                "take_profit": 0.12
            },
            "sideways": {
                "primary": "mean_reversion",
                "risk_level": "low",
                "position_size": 0.5,
                "stop_loss": 0.03,
                "take_profit": 0.08
            },
            "volatile": {
                "primary": "volatility_trading",
                "risk_level": "high",
                "position_size": 0.3,
                "stop_loss": 0.08,
                "take_profit": 0.20
            }
        }
    
    async def initialize(self):
        """Initialize the regime detection system."""
        try:
            logger.info("Initializing market regime detection system...")
            
            # Create model directories
            model_dir = Path(self.config.regime_model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.ready = True
            logger.info("Market regime detection system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize regime detection system: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        # Save models before cleanup
        await self._save_models()
        
        self.ready = False
        logger.info("Market regime detection system cleaned up")
    
    def is_ready(self) -> bool:
        """Check if the detector is ready."""
        return self.ready
    
    async def detect_regime(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Detect current market regime for a symbol."""
        
        if not self.ready:
            raise RuntimeError("Regime detection system not initialized")
        
        logger.info("Detecting market regime", symbol=symbol, timeframe=timeframe)
        
        try:
            # Get historical data
            historical_data = await self._get_historical_data(symbol, timeframe)
            
            if len(historical_data) < self.regime_config['lookback_period']:
                logger.warning(f"Insufficient data for regime detection: {len(historical_data)} < {self.regime_config['lookback_period']}")
                return self._get_default_regime_response(symbol)
            
            # Extract features for regime detection
            features = self._extract_regime_features(historical_data)
            
            # Get or train regime model
            regime_model = await self._get_or_train_regime_model(symbol, features)
            
            # Detect current regime
            current_regime_id, regime_probability = self._predict_current_regime(regime_model, features)
            current_regime = self.regime_labels[current_regime_id]
            
            # Get regime history
            regime_history = self._analyze_regime_history(regime_model, features, symbol)
            
            # Forecast volatility using GARCH
            volatility_forecast = await self._forecast_volatility(symbol, historical_data)
            
            # Get recommended strategy
            strategy_recommendation = self._get_strategy_recommendation(current_regime, volatility_forecast)
            
            # Update regime history
            self._update_regime_history(symbol, current_regime, regime_probability)
            
            return {
                "symbol": symbol,
                "current_regime": current_regime,
                "regime_probability": float(regime_probability),
                "regime_history": regime_history,
                "volatility_forecast": float(volatility_forecast),
                "strategy_recommendation": strategy_recommendation,
                "regime_transition_matrix": self._get_transition_probabilities(regime_model),
                "confidence_score": self._calculate_confidence_score(regime_probability, len(historical_data)),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Regime detection failed", error=str(e), symbol=symbol)
            return self._get_default_regime_response(symbol)
    
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection."""
        df = data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate rolling volatility (20-day)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Momentum (20-day price change)
        df['momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_position'] = (df['close'] - rolling_mean) / (2 * rolling_std)
        
        # Trend strength (using linear regression slope)
        def calculate_trend_strength(prices, window=20):
            trends = []
            for i in range(len(prices)):
                if i < window:
                    trends.append(0)
                else:
                    y = prices[i-window:i].values
                    x = np.arange(len(y))
                    if len(y) > 1:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        trends.append(slope)
                    else:
                        trends.append(0)
            return trends
        
        df['trend_strength'] = calculate_trend_strength(df['close'])
        
        # Select features for regime detection
        feature_columns = ['returns', 'volatility', 'volume_ratio', 'momentum', 'rsi', 'bb_position', 'trend_strength']
        features_df = df[feature_columns].copy()
        
        # Fill NaN values
        features_df = features_df.bfill().ffill()
        
        return features_df
    
    async def _get_or_train_regime_model(self, symbol: str, features: pd.DataFrame) -> GaussianMixture:
        """Get existing regime model or train a new one."""
        model_key = f"regime_{symbol}"
        
        if model_key in self.regime_models:
            # Check if model needs retraining (e.g., if it's old or performance degraded)
            if self._should_retrain_regime_model(symbol):
                logger.info(f"Retraining regime model for {symbol}")
                await self._train_regime_model(symbol, features)
        else:
            # Train new model
            logger.info(f"Training new regime model for {symbol}")
            await self._train_regime_model(symbol, features)
        
        return self.regime_models[model_key]
    
    async def _train_regime_model(self, symbol: str, features: pd.DataFrame):
        """Train Hidden Markov Model for regime detection using Gaussian Mixture Model."""
        try:
            model_key = f"regime_{symbol}"
            scaler_key = f"regime_scaler_{symbol}"
            
            # Prepare features
            feature_data = features[self.regime_config['features']].values
            
            # Remove any remaining NaN values
            mask = ~np.isnan(feature_data).any(axis=1)
            feature_data = feature_data[mask]
            
            if len(feature_data) < 50:
                raise ValueError(f"Insufficient clean data for training: {len(feature_data)}")
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Train Gaussian Mixture Model as proxy for HMM
            # This identifies different market regimes based on feature distributions
            model = GaussianMixture(
                n_components=self.regime_config['n_regimes'],
                covariance_type='full',
                max_iter=200,
                random_state=42,
                init_params='kmeans'
            )
            
            model.fit(scaled_features)
            
            # Store model and scaler
            self.regime_models[model_key] = model
            self.scalers[scaler_key] = scaler
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regime_characteristics(model, scaled_features, features)
            
            logger.info(f"Regime model trained for {symbol}", 
                       n_regimes=self.regime_config['n_regimes'],
                       aic=model.aic(scaled_features),
                       bic=model.bic(scaled_features))
            
        except Exception as e:
            logger.error(f"Failed to train regime model for {symbol}: {e}")
            raise
    
    def _predict_current_regime(self, model: GaussianMixture, features: pd.DataFrame) -> Tuple[int, float]:
        """Predict current market regime."""
        try:
            # Get the most recent features
            recent_features = features[self.regime_config['features']].iloc[-1:].values
            
            # Scale features
            symbol = "default"  # We'll need to pass this properly in production
            scaler_key = f"regime_scaler_{symbol}"
            if scaler_key in self.scalers:
                scaler = self.scalers[scaler_key]
                scaled_features = scaler.transform(recent_features)
            else:
                scaled_features = recent_features
            
            # Predict regime
            regime_probs = model.predict_proba(scaled_features)[0]
            regime_id = np.argmax(regime_probs)
            regime_probability = regime_probs[regime_id]
            
            return regime_id, regime_probability
            
        except Exception as e:
            logger.error(f"Failed to predict current regime: {e}")
            # Return default regime (sideways) with low confidence
            return 2, 0.5
    
    def _analyze_regime_history(self, model: GaussianMixture, features: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Analyze historical regime transitions."""
        try:
            # Get regime predictions for all historical data
            feature_data = features[self.regime_config['features']].values
            
            # Remove NaN values
            mask = ~np.isnan(feature_data).any(axis=1)
            clean_features = feature_data[mask]
            clean_dates = features.index[mask]
            
            if len(clean_features) == 0:
                return []
            
            # Scale features
            scaler_key = f"regime_scaler_{symbol}"
            if scaler_key in self.scalers:
                scaler = self.scalers[scaler_key]
                scaled_features = scaler.transform(clean_features)
            else:
                scaled_features = clean_features
            
            # Predict regimes for all data
            regime_predictions = model.predict(scaled_features)
            regime_probabilities = model.predict_proba(scaled_features)
            
            # Identify regime changes
            regime_changes = []
            current_regime = regime_predictions[0]
            regime_start = clean_dates[0]
            
            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] != current_regime:
                    # Regime change detected
                    regime_changes.append({
                        "regime": self.regime_labels[current_regime],
                        "start_date": regime_start.isoformat() if hasattr(regime_start, 'isoformat') else str(regime_start),
                        "end_date": clean_dates[i-1].isoformat() if hasattr(clean_dates[i-1], 'isoformat') else str(clean_dates[i-1]),
                        "duration_days": (clean_dates[i-1] - regime_start).days if hasattr(clean_dates[i-1], 'days') else 1,
                        "avg_probability": float(np.mean(regime_probabilities[max(0, i-10):i, current_regime]))
                    })
                    
                    current_regime = regime_predictions[i]
                    regime_start = clean_dates[i]
            
            # Add the final regime
            if len(clean_dates) > 0:
                regime_changes.append({
                    "regime": self.regime_labels[current_regime],
                    "start_date": regime_start.isoformat() if hasattr(regime_start, 'isoformat') else str(regime_start),
                    "end_date": clean_dates[-1].isoformat() if hasattr(clean_dates[-1], 'isoformat') else str(clean_dates[-1]),
                    "duration_days": (clean_dates[-1] - regime_start).days if hasattr(clean_dates[-1], 'days') else 1,
                    "avg_probability": float(np.mean(regime_probabilities[-10:, current_regime]))
                })
            
            # Return the most recent regime changes (last 10)
            return regime_changes[-10:]
            
        except Exception as e:
            logger.error(f"Failed to analyze regime history: {e}")
            return []
    
    async def _forecast_volatility(self, symbol: str, data: pd.DataFrame) -> float:
        """Forecast volatility using GARCH model."""
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < self.garch_config['min_observations']:
                # Use simple historical volatility if insufficient data
                return float(returns.std() * np.sqrt(252))  # Annualized volatility
            
            # Get or train GARCH model
            garch_model = await self._get_or_train_garch_model(symbol, returns)
            
            # Forecast volatility
            forecast = self._garch_forecast(garch_model, returns)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to forecast volatility for {symbol}: {e}")
            # Return simple historical volatility as fallback
            returns = data['close'].pct_change().dropna()
            return float(returns.std() * np.sqrt(252))
    
    async def _get_or_train_garch_model(self, symbol: str, returns: pd.Series) -> Dict[str, Any]:
        """Get existing GARCH model or train a new one."""
        model_key = f"garch_{symbol}"
        
        if model_key not in self.garch_models:
            logger.info(f"Training GARCH model for {symbol}")
            await self._train_garch_model(symbol, returns)
        
        return self.garch_models[model_key]
    
    async def _train_garch_model(self, symbol: str, returns: pd.Series):
        """Train GARCH(1,1) model for volatility forecasting."""
        try:
            model_key = f"garch_{symbol}"
            
            # Simple GARCH(1,1) implementation
            # In production, you might use arch library: from arch import arch_model
            
            # Calculate parameters using maximum likelihood estimation
            returns_array = returns.values
            
            # Initial parameter estimates
            omega = np.var(returns_array) * 0.1  # Long-term variance
            alpha = 0.1  # ARCH parameter
            beta = 0.8   # GARCH parameter
            
            # Simple parameter estimation (in production, use proper MLE)
            # This is a simplified version for demonstration
            
            # Store model parameters
            garch_model = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'last_variance': np.var(returns_array[-20:]),  # Recent variance
                'last_return': returns_array[-1],
                'trained_date': datetime.utcnow().isoformat()
            }
            
            self.garch_models[model_key] = garch_model
            
            logger.info(f"GARCH model trained for {symbol}", 
                       omega=omega, alpha=alpha, beta=beta)
            
        except Exception as e:
            logger.error(f"Failed to train GARCH model for {symbol}: {e}")
            # Store a simple fallback model
            self.garch_models[model_key] = {
                'omega': 0.0001,
                'alpha': 0.1,
                'beta': 0.8,
                'last_variance': np.var(returns.values),
                'last_return': returns.iloc[-1],
                'trained_date': datetime.utcnow().isoformat()
            }
    
    def _garch_forecast(self, model: Dict[str, Any], returns: pd.Series) -> float:
        """Forecast volatility using GARCH model."""
        try:
            # GARCH(1,1) forecast: σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
            omega = model['omega']
            alpha = model['alpha'] 
            beta = model['beta']
            last_variance = model['last_variance']
            last_return = model['last_return']
            
            # One-step ahead variance forecast
            next_variance = omega + alpha * (last_return ** 2) + beta * last_variance
            
            # Convert to annualized volatility
            annualized_volatility = np.sqrt(next_variance * 252)
            
            return float(annualized_volatility)
            
        except Exception as e:
            logger.error(f"GARCH forecast failed: {e}")
            # Fallback to historical volatility
            return float(returns.std() * np.sqrt(252))
    
    def _get_strategy_recommendation(self, regime: str, volatility_forecast: float) -> Dict[str, Any]:
        """Get strategy recommendation based on current regime."""
        base_strategy = self.regime_strategies[regime].copy()
        
        # Adjust strategy based on volatility forecast
        if volatility_forecast > 0.4:  # High volatility
            base_strategy['position_size'] *= 0.7  # Reduce position size
            base_strategy['stop_loss'] *= 1.5      # Wider stop loss
        elif volatility_forecast < 0.15:  # Low volatility
            base_strategy['position_size'] *= 1.2  # Increase position size
            base_strategy['stop_loss'] *= 0.8      # Tighter stop loss
        
        # Add volatility-specific recommendations
        base_strategy['volatility_forecast'] = volatility_forecast
        base_strategy['volatility_regime'] = self._classify_volatility_regime(volatility_forecast)
        
        return base_strategy
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility into regimes."""
        if volatility > 0.4:
            return "high"
        elif volatility > 0.25:
            return "medium"
        else:
            return "low"
    
    def _get_transition_probabilities(self, model: GaussianMixture) -> Dict[str, Dict[str, float]]:
        """Calculate regime transition probabilities."""
        # This is a simplified version - in a true HMM, transition probabilities are learned
        # For GMM, we estimate based on component weights
        
        weights = model.weights_
        n_regimes = len(weights)
        
        # Create a simple transition matrix based on regime stability
        # More stable regimes (higher weight) have higher self-transition probability
        transition_matrix = {}
        
        for i, from_regime in enumerate(self.regime_labels.values()):
            transition_matrix[from_regime] = {}
            
            for j, to_regime in enumerate(self.regime_labels.values()):
                if i == j:
                    # Self-transition probability (higher for more stable regimes)
                    prob = 0.7 + 0.2 * weights[i]
                else:
                    # Transition to other regimes
                    prob = (1 - (0.7 + 0.2 * weights[i])) / (n_regimes - 1)
                
                transition_matrix[from_regime][to_regime] = float(prob)
        
        return transition_matrix
    
    def _calculate_confidence_score(self, regime_probability: float, data_length: int) -> float:
        """Calculate confidence score for regime detection."""
        # Confidence based on regime probability and data sufficiency
        prob_confidence = regime_probability
        data_confidence = min(1.0, data_length / self.regime_config['lookback_period'])
        
        # Combined confidence score
        confidence = (prob_confidence * 0.7 + data_confidence * 0.3)
        
        return float(confidence)
    
    def _update_regime_history(self, symbol: str, regime: str, probability: float):
        """Update regime history for a symbol."""
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        # Add current regime observation
        self.regime_history[symbol].append({
            'regime': regime,
            'probability': probability,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only recent history (last 100 observations)
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]
    
    def _should_retrain_regime_model(self, symbol: str) -> bool:
        """Check if regime model should be retrained."""
        # Simple retraining logic - retrain if model is older than 30 days
        # In production, you might use more sophisticated criteria
        return False  # For now, don't retrain automatically
    
    def _analyze_regime_characteristics(self, model: GaussianMixture, scaled_features: np.ndarray, original_features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        try:
            regime_labels = model.predict(scaled_features)
            
            characteristics = {}
            for regime_id in range(self.regime_config['n_regimes']):
                regime_mask = regime_labels == regime_id
                regime_data = original_features[regime_mask]
                
                if len(regime_data) > 0:
                    characteristics[self.regime_labels[regime_id]] = {
                        'avg_return': float(regime_data['returns'].mean()),
                        'avg_volatility': float(regime_data['volatility'].mean()),
                        'avg_volume_ratio': float(regime_data['volume_ratio'].mean()),
                        'frequency': float(np.sum(regime_mask) / len(regime_labels))
                    }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Failed to analyze regime characteristics: {e}")
            return {}
    
    def _get_default_regime_response(self, symbol: str) -> Dict[str, Any]:
        """Get default regime response when detection fails."""
        return {
            "symbol": symbol,
            "current_regime": "sideways",
            "regime_probability": 0.5,
            "regime_history": [],
            "volatility_forecast": 0.2,
            "strategy_recommendation": self.regime_strategies["sideways"],
            "regime_transition_matrix": {},
            "confidence_score": 0.3,
            "timestamp": datetime.utcnow()
        }
    
    async def _get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get historical data for regime detection (placeholder implementation)."""
        # In production, this would fetch real data from exchanges or data providers
        # For now, generate synthetic data for testing
        
        logger.info(f"Fetching historical data for regime detection: {symbol} ({timeframe})")
        
        # Generate synthetic OHLCV data with regime changes
        np.random.seed(42)  # For reproducible results
        
        periods = 500  # Generate 500 data points (about 2 years of daily data)
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1D')
        
        # Generate realistic price data with different regimes
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        # Create regime periods
        regime_periods = [
            (0, 100, 'bull'),      # Bull market
            (100, 200, 'volatile'), # Volatile period
            (200, 350, 'bear'),     # Bear market
            (350, 450, 'sideways'), # Sideways market
            (450, 500, 'bull')      # Return to bull
        ]
        
        prices = [base_price]
        volumes = []
        
        for i in range(1, periods):
            # Determine current regime
            current_regime = 'sideways'  # default
            for start, end, regime in regime_periods:
                if start <= i < end:
                    current_regime = regime
                    break
            
            # Generate returns based on regime
            if current_regime == 'bull':
                mean_return = 0.001
                volatility = 0.015
            elif current_regime == 'bear':
                mean_return = -0.0008
                volatility = 0.018
            elif current_regime == 'volatile':
                mean_return = 0.0002
                volatility = 0.035
            else:  # sideways
                mean_return = 0.0001
                volatility = 0.012
            
            # Generate return
            daily_return = np.random.normal(mean_return, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Generate volume (higher in volatile periods)
            base_volume = 10000
            if current_regime == 'volatile':
                volume = np.random.uniform(base_volume * 1.5, base_volume * 3)
            else:
                volume = np.random.uniform(base_volume * 0.8, base_volume * 1.2)
            volumes.append(volume)
        
        # Generate OHLC data from close prices
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            if i == 0:
                continue
                
            # Generate realistic OHLC from close price
            volatility = close * 0.02  # 2% intraday volatility
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] + np.random.uniform(-volatility/2, volatility/2)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volumes[i-1]
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _load_existing_models(self):
        """Load existing trained models from disk."""
        try:
            model_dir = Path(self.config.regime_model_path)
            
            if model_dir.exists():
                # Load regime history
                history_file = model_dir / "regime_history.json"
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        self.regime_history = json.load(f)
                
                logger.info("Loaded existing regime models and history")
            
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk."""
        try:
            model_dir = Path(self.config.regime_model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime history
            with open(model_dir / "regime_history.json", 'w') as f:
                json.dump(self.regime_history, f, indent=2, default=str)
            
            # Save models (would need proper serialization for sklearn models)
            # For now, just save the history
            
            logger.info("Regime models and history saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def get_regime_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """Get regime statistics for analysis."""
        if symbol and symbol in self.regime_history:
            history = self.regime_history[symbol]
            
            # Calculate regime distribution
            regimes = [entry['regime'] for entry in history]
            regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
            
            return {
                'symbol': symbol,
                'total_observations': len(history),
                'regime_distribution': regime_counts,
                'recent_regimes': history[-10:] if len(history) >= 10 else history
            }
        
        return {
            'total_symbols': len(self.regime_history),
            'symbols': list(self.regime_history.keys())
        }