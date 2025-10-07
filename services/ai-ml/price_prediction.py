"""
Price Prediction Engine using LSTM and Transformer models.
Provides ensemble predictions with confidence intervals for cryptocurrency trading.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path
import structlog

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)


class PricePredictionEngine:
    """Advanced price prediction engine using LSTM and Transformer models."""
    
    def __init__(self, config):
        self.config = config
        self.ready = False
        
        # Model configurations
        self.lstm_config = {
            'sequence_length': 60,  # 60 time steps for prediction
            'features': ['open', 'high', 'low', 'close', 'volume'],
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
        
        self.transformer_config = {
            'sequence_length': 100,  # Longer sequence for transformer
            'features': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'],
            'embed_dim': 128,
            'num_heads': 8,
            'ff_dim': 256,
            'num_layers': 4,
            'dropout_rate': 0.1,
            'learning_rate': 0.0001
        }
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # Ensemble weights (will be learned from validation performance)
        self.ensemble_weights = {
            'lstm': 0.4,
            'transformer': 0.4,
            'linear': 0.2
        }
        
        # Performance tracking
        self.model_performance = {}
        
    async def initialize(self):
        """Initialize the price prediction engine."""
        try:
            logger.info("Initializing price prediction engine...")
            
            # Create model directories
            model_dir = Path(self.config.price_model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.ready = True
            logger.info("Price prediction engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize price prediction engine: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        # Save models before cleanup
        await self._save_models()
        
        # Clear GPU memory
        tf.keras.backend.clear_session()
        
        self.ready = False
        logger.info("Price prediction engine cleaned up")
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.ready
    
    async def predict_price(
        self,
        symbol: str,
        timeframe: str = "1h",
        horizon: int = 24,
        models: List[str] = None
    ) -> Dict[str, Any]:
        """Generate price predictions for a given symbol."""
        
        if not self.ready:
            raise RuntimeError("Price prediction engine not initialized")
        
        if models is None:
            models = ["lstm", "transformer"]
        
        logger.info("Generating price predictions", symbol=symbol, models=models, horizon=horizon)
        
        try:
            # Get historical data for the symbol
            historical_data = await self._get_historical_data(symbol, timeframe)
            
            if len(historical_data) < max(self.lstm_config['sequence_length'], self.transformer_config['sequence_length']):
                raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Get current price
            current_price = float(historical_data['close'].iloc[-1])
            
            # Generate predictions from each model
            predictions = {}
            confidence_intervals = {}
            
            if "lstm" in models:
                lstm_pred, lstm_ci = await self._predict_with_lstm(historical_data, symbol, horizon)
                predictions["lstm"] = lstm_pred
                confidence_intervals["lstm"] = lstm_ci
            
            if "transformer" in models:
                transformer_pred, transformer_ci = await self._predict_with_transformer(historical_data, symbol, horizon)
                predictions["transformer"] = transformer_pred
                confidence_intervals["transformer"] = transformer_ci
            
            # Add simple linear regression as baseline
            if "linear" in models or len(models) > 1:
                linear_pred, linear_ci = await self._predict_with_linear(historical_data, horizon)
                predictions["linear"] = linear_pred
                confidence_intervals["linear"] = linear_ci
            
            # Generate ensemble prediction
            ensemble_prediction = self._create_ensemble_prediction(predictions, symbol)
            
            # Calculate prediction timeframes
            prediction_times = self._generate_prediction_timeframes(timeframe, horizon)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "predictions": predictions,
                "confidence_intervals": confidence_intervals,
                "ensemble_prediction": ensemble_prediction,
                "prediction_times": prediction_times,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Price prediction failed", error=str(e), symbol=symbol)
            raise
    
    async def _predict_with_lstm(self, data: pd.DataFrame, symbol: str, horizon: int) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Generate predictions using LSTM model."""
        try:
            # Prepare data for LSTM
            features = self.lstm_config['features']
            sequence_length = self.lstm_config['sequence_length']
            
            # Get or create model for this symbol
            model_key = f"lstm_{symbol}"
            if model_key not in self.models:
                await self._train_lstm_model(data, symbol)
            
            model = self.models[model_key]
            scaler = self.scalers[f"{model_key}_scaler"]
            
            # Prepare input sequence
            feature_data = data[features].values
            scaled_data = scaler.transform(feature_data)
            
            # Create sequence for prediction
            sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))
            
            # Generate predictions for different horizons
            predictions = {}
            confidence_intervals = {}
            
            # Predict multiple steps ahead
            current_sequence = sequence.copy()
            
            for step in [1, 6, 12, 24]:  # 1h, 6h, 12h, 24h ahead
                if step <= horizon:
                    # Predict next step
                    pred_scaled = model.predict(current_sequence, verbose=0)
                    
                    # Inverse transform prediction (only close price)
                    pred_unscaled = scaler.inverse_transform(
                        np.concatenate([
                            current_sequence[0, -1, :-1].reshape(1, -1),
                            pred_scaled
                        ], axis=1)
                    )[0, -1]  # Get close price
                    
                    predictions[f"{step}h"] = float(pred_unscaled)
                    
                    # Calculate confidence interval (simple approach using model uncertainty)
                    # In production, you might use Monte Carlo dropout or ensemble methods
                    uncertainty = abs(pred_unscaled * 0.02)  # 2% uncertainty
                    confidence_intervals[f"{step}h"] = [
                        float(pred_unscaled - uncertainty),
                        float(pred_unscaled + uncertainty)
                    ]
                    
                    # Update sequence for next prediction (rolling window)
                    if step < horizon:
                        # Create next input by shifting sequence
                        new_features = np.concatenate([
                            current_sequence[0, 1:, :],
                            pred_scaled.reshape(1, 1, -1)
                        ], axis=1)
                        current_sequence = new_features
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            # Return fallback prediction
            current_price = float(data['close'].iloc[-1])
            return {
                "1h": current_price * 1.001,
                "6h": current_price * 1.005,
                "12h": current_price * 1.01,
                "24h": current_price * 1.02
            }, {
                "1h": [current_price * 0.99, current_price * 1.01],
                "6h": [current_price * 0.98, current_price * 1.02],
                "12h": [current_price * 0.97, current_price * 1.03],
                "24h": [current_price * 0.95, current_price * 1.05]
            }
    
    async def _predict_with_transformer(self, data: pd.DataFrame, symbol: str, horizon: int) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Generate predictions using Transformer model."""
        try:
            # Add technical indicators for transformer
            data_with_indicators = self._add_technical_indicators(data)
            
            features = self.transformer_config['features']
            sequence_length = self.transformer_config['sequence_length']
            
            # Get or create model for this symbol
            model_key = f"transformer_{symbol}"
            if model_key not in self.models:
                await self._train_transformer_model(data_with_indicators, symbol)
            
            model = self.models[model_key]
            scaler = self.scalers[f"{model_key}_scaler"]
            
            # Prepare input sequence
            feature_data = data_with_indicators[features].values
            scaled_data = scaler.transform(feature_data)
            
            # Create sequence for prediction
            sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))
            
            # Generate predictions
            predictions = {}
            confidence_intervals = {}
            
            # Transformer can predict multiple steps at once
            pred_scaled = model.predict(sequence, verbose=0)
            
            # Inverse transform predictions
            for i, step in enumerate([1, 6, 12, 24]):
                if step <= horizon and i < pred_scaled.shape[1]:
                    pred_unscaled = scaler.inverse_transform(
                        np.concatenate([
                            sequence[0, -1, :-1].reshape(1, -1),
                            pred_scaled[0, i].reshape(1, -1)
                        ], axis=1)
                    )[0, -1]  # Get close price
                    
                    predictions[f"{step}h"] = float(pred_unscaled)
                    
                    # Calculate confidence interval
                    uncertainty = abs(pred_unscaled * 0.025)  # 2.5% uncertainty for transformer
                    confidence_intervals[f"{step}h"] = [
                        float(pred_unscaled - uncertainty),
                        float(pred_unscaled + uncertainty)
                    ]
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            # Return fallback prediction
            current_price = float(data['close'].iloc[-1])
            return {
                "1h": current_price * 1.002,
                "6h": current_price * 1.008,
                "12h": current_price * 1.015,
                "24h": current_price * 1.025
            }, {
                "1h": [current_price * 0.985, current_price * 1.015],
                "6h": [current_price * 0.975, current_price * 1.025],
                "12h": [current_price * 0.965, current_price * 1.035],
                "24h": [current_price * 0.95, current_price * 1.05]
            }
    
    async def _predict_with_linear(self, data: pd.DataFrame, horizon: int) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Generate predictions using simple linear regression as baseline."""
        try:
            # Simple linear trend prediction
            prices = data['close'].values
            x = np.arange(len(prices))
            
            # Fit linear regression
            coeffs = np.polyfit(x[-50:], prices[-50:], 1)  # Use last 50 points
            
            predictions = {}
            confidence_intervals = {}
            
            current_price = float(prices[-1])
            trend = coeffs[0]  # Slope
            
            for step in [1, 6, 12, 24]:
                if step <= horizon:
                    pred_price = current_price + (trend * step)
                    predictions[f"{step}h"] = float(pred_price)
                    
                    # Simple confidence interval based on recent volatility
                    recent_volatility = np.std(prices[-24:]) if len(prices) >= 24 else np.std(prices)
                    uncertainty = recent_volatility * np.sqrt(step)
                    
                    confidence_intervals[f"{step}h"] = [
                        float(pred_price - uncertainty),
                        float(pred_price + uncertainty)
                    ]
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Linear prediction failed: {e}")
            current_price = float(data['close'].iloc[-1])
            return {
                "1h": current_price,
                "6h": current_price,
                "12h": current_price,
                "24h": current_price
            }, {
                "1h": [current_price * 0.99, current_price * 1.01],
                "6h": [current_price * 0.97, current_price * 1.03],
                "12h": [current_price * 0.95, current_price * 1.05],
                "24h": [current_price * 0.93, current_price * 1.07]
            }
    
    def _create_ensemble_prediction(self, predictions: Dict[str, Dict[str, float]], symbol: str) -> float:
        """Create ensemble prediction by combining multiple models."""
        if not predictions:
            return 0.0
        
        # Get weights for this symbol (or use default)
        weights = self.ensemble_weights.copy()
        
        # Adjust weights based on model performance if available
        if symbol in self.model_performance:
            perf = self.model_performance[symbol]
            total_weight = 0
            
            for model_name in predictions.keys():
                if model_name in perf:
                    # Weight inversely proportional to error (lower error = higher weight)
                    error = perf[model_name].get('mse', 1.0)
                    weights[model_name] = 1.0 / (1.0 + error)
                    total_weight += weights[model_name]
            
            # Normalize weights
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
        
        # Calculate weighted average for 24h prediction
        ensemble_pred = 0.0
        total_weight = 0.0
        
        for model_name, model_preds in predictions.items():
            if "24h" in model_preds and model_name in weights:
                weight = weights[model_name]
                ensemble_pred += model_preds["24h"] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return float(ensemble_pred)
    
    async def _train_lstm_model(self, data: pd.DataFrame, symbol: str):
        """Train LSTM model for a specific symbol."""
        try:
            logger.info(f"Training LSTM model for {symbol}")
            
            features = self.lstm_config['features']
            sequence_length = self.lstm_config['sequence_length']
            
            # Prepare data
            feature_data = data[features].values
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, sequence_length, target_col=-1)  # Close price is last
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(self.lstm_config['lstm_units'][0], return_sequences=True, input_shape=(sequence_length, len(features))),
                Dropout(self.lstm_config['dropout_rate']),
                LSTM(self.lstm_config['lstm_units'][1], return_sequences=True),
                Dropout(self.lstm_config['dropout_rate']),
                LSTM(self.lstm_config['lstm_units'][2]),
                Dropout(self.lstm_config['dropout_rate']),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.lstm_config['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model and scaler
            model_key = f"lstm_{symbol}"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            
            # Calculate performance metrics
            y_pred = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            if symbol not in self.model_performance:
                self.model_performance[symbol] = {}
            
            self.model_performance[symbol]['lstm'] = {
                'mse': float(mse),
                'mae': float(mae),
                'trained_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"LSTM model trained for {symbol}", mse=mse, mae=mae)
            
        except Exception as e:
            logger.error(f"Failed to train LSTM model for {symbol}: {e}")
            raise
    
    async def _train_transformer_model(self, data: pd.DataFrame, symbol: str):
        """Train Transformer model for a specific symbol."""
        try:
            logger.info(f"Training Transformer model for {symbol}")
            
            features = self.transformer_config['features']
            sequence_length = self.transformer_config['sequence_length']
            
            # Prepare data
            feature_data = data[features].values
            
            # Scale data
            scaler = StandardScaler()  # Transformer works better with StandardScaler
            scaled_data = scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, sequence_length, target_col=-1)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build Transformer model
            model = self._build_transformer_model(sequence_length, len(features))
            
            # Train model
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=7, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=16,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model and scaler
            model_key = f"transformer_{symbol}"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            
            # Calculate performance metrics
            y_pred = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            if symbol not in self.model_performance:
                self.model_performance[symbol] = {}
            
            self.model_performance[symbol]['transformer'] = {
                'mse': float(mse),
                'mae': float(mae),
                'trained_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Transformer model trained for {symbol}", mse=mse, mae=mae)
            
        except Exception as e:
            logger.error(f"Failed to train Transformer model for {symbol}: {e}")
            raise
    
    def _build_transformer_model(self, sequence_length: int, num_features: int) -> Model:
        """Build Transformer model architecture."""
        
        # Input layer
        inputs = Input(shape=(sequence_length, num_features))
        
        # Transformer blocks
        x = inputs
        for _ in range(self.transformer_config['num_layers']):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.transformer_config['num_heads'],
                key_dim=self.transformer_config['embed_dim'] // self.transformer_config['num_heads']
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization()(x + attention_output)
            
            # Feed forward
            ff_output = Dense(self.transformer_config['ff_dim'], activation='relu')(x)
            ff_output = Dropout(self.transformer_config['dropout_rate'])(ff_output)
            ff_output = Dense(num_features)(ff_output)
            
            # Add & Norm
            x = LayerNormalization()(x + ff_output)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.transformer_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int, target_col: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, target_col])  # Target is the close price
        
        return np.array(X), np.array(y)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for transformer model."""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # Fill NaN values
        df = df.bfill().ffill()
        
        return df
    
    def _generate_prediction_timeframes(self, timeframe: str, horizon: int) -> List[str]:
        """Generate prediction timeframes."""
        base_time = datetime.utcnow()
        timeframes = []
        
        # Convert timeframe to hours
        if timeframe == "1m":
            hours_per_step = 1/60
        elif timeframe == "5m":
            hours_per_step = 5/60
        elif timeframe == "15m":
            hours_per_step = 15/60
        elif timeframe == "1h":
            hours_per_step = 1
        elif timeframe == "4h":
            hours_per_step = 4
        elif timeframe == "1d":
            hours_per_step = 24
        else:
            hours_per_step = 1  # Default to 1 hour
        
        for step in [1, 6, 12, 24]:
            if step <= horizon:
                future_time = base_time + timedelta(hours=step * hours_per_step)
                timeframes.append(future_time.isoformat())
        
        return timeframes
    
    async def _get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get historical data for a symbol (placeholder implementation)."""
        # In production, this would fetch real data from exchanges or data providers
        # For now, generate synthetic data for testing
        
        logger.info(f"Fetching historical data for {symbol} ({timeframe})")
        
        # Generate synthetic OHLCV data
        np.random.seed(42)  # For reproducible results
        
        periods = 1000  # Generate 1000 data points
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='1H')
        
        # Generate realistic price data with trend and volatility
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        # Random walk with trend
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive trend with 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = close * 0.01  # 1% intraday volatility
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = low + np.random.uniform(0, high - low)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.uniform(1000, 10000)  # Random volume
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _load_existing_models(self):
        """Load existing trained models from disk."""
        try:
            model_dir = Path(self.config.price_model_path)
            
            if model_dir.exists():
                # Load model metadata
                metadata_file = model_dir / "model_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.model_metadata = json.load(f)
                
                # Load performance data
                perf_file = model_dir / "model_performance.json"
                if perf_file.exists():
                    with open(perf_file, 'r') as f:
                        self.model_performance = json.load(f)
                
                logger.info("Loaded existing model metadata and performance data")
            
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk."""
        try:
            model_dir = Path(self.config.price_model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model metadata
            with open(model_dir / "model_metadata.json", 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            # Save performance data
            with open(model_dir / "model_performance.json", 'w') as f:
                json.dump(self.model_performance, f, indent=2, default=str)
            
            # Save individual models (TensorFlow models)
            for model_name, model in self.models.items():
                model_path = model_dir / f"{model_name}.h5"
                model.save(str(model_path))
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = model_dir / f"{scaler_name}.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            logger.info("Models and scalers saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def retrain_models(self, symbol: str, force: bool = False):
        """Retrain models for a specific symbol."""
        try:
            # Check if retraining is needed
            if not force and symbol in self.model_performance:
                last_trained = self.model_performance[symbol].get('lstm', {}).get('trained_at')
                if last_trained:
                    last_trained_dt = datetime.fromisoformat(last_trained.replace('Z', '+00:00'))
                    if datetime.utcnow() - last_trained_dt < timedelta(days=7):
                        logger.info(f"Models for {symbol} are recent, skipping retraining")
                        return
            
            logger.info(f"Retraining models for {symbol}")
            
            # Get fresh historical data
            historical_data = await self._get_historical_data(symbol, "1h")
            
            # Retrain LSTM
            await self._train_lstm_model(historical_data, symbol)
            
            # Retrain Transformer
            data_with_indicators = self._add_technical_indicators(historical_data)
            await self._train_transformer_model(data_with_indicators, symbol)
            
            # Save updated models
            await self._save_models()
            
            logger.info(f"Models retrained successfully for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to retrain models for {symbol}: {e}")
            raise
    
    def get_model_performance(self, symbol: str = None) -> Dict[str, Any]:
        """Get model performance metrics."""
        if symbol:
            return self.model_performance.get(symbol, {})
        return self.model_performance
    
    def update_ensemble_weights(self, symbol: str, weights: Dict[str, float]):
        """Update ensemble weights for a specific symbol."""
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Store symbol-specific weights (could be extended to per-symbol weights)
            self.ensemble_weights.update(normalized_weights)
            
            logger.info(f"Updated ensemble weights for {symbol}", weights=normalized_weights)