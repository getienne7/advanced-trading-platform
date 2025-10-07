"""
MLOps Configuration
Configuration settings for MLOps pipeline components.
"""
import os
from typing import Dict, Any
from pathlib import Path


class MLOpsConfig:
    """MLOps configuration settings."""
    
    def __init__(self):
        # Base paths
        self.base_path = Path(os.getenv("MLOPS_BASE_PATH", "models"))
        self.model_registry_path = self.base_path / "registry"
        self.experiment_path = self.base_path / "experiments"
        self.monitoring_path = self.base_path / "monitoring"
        
        # MLflow configuration
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{self.base_path}/mlruns")
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "trading_models")
        
        # Model training configuration
        self.training_config = {
            'max_concurrent_jobs': int(os.getenv("MAX_CONCURRENT_TRAINING_JOBS", "3")),
            'job_timeout_hours': int(os.getenv("TRAINING_JOB_TIMEOUT_HOURS", "6")),
            'retry_attempts': int(os.getenv("TRAINING_RETRY_ATTEMPTS", "2")),
            'retry_delay_minutes': int(os.getenv("TRAINING_RETRY_DELAY_MINUTES", "30")),
            'min_training_data_points': int(os.getenv("MIN_TRAINING_DATA_POINTS", "1000")),
            'validation_split': float(os.getenv("VALIDATION_SPLIT", "0.2"))
        }
        
        # Model validation thresholds
        self.validation_thresholds = {
            'min_accuracy': float(os.getenv("MIN_MODEL_ACCURACY", "0.6")),
            'max_mse': float(os.getenv("MAX_MODEL_MSE", "1000")),
            'min_r2_score': float(os.getenv("MIN_R2_SCORE", "0.5")),
            'min_sharpe_ratio': float(os.getenv("MIN_SHARPE_RATIO", "0.5")),
            'max_drawdown': float(os.getenv("MAX_DRAWDOWN", "0.2"))
        }
        
        # Drift detection configuration
        self.drift_config = {
            'drift_threshold': float(os.getenv("DRIFT_THRESHOLD", "0.1")),
            'performance_threshold': float(os.getenv("PERFORMANCE_THRESHOLD", "0.15")),
            'monitoring_window_hours': int(os.getenv("MONITORING_WINDOW_HOURS", "24")),
            'min_samples_for_drift': int(os.getenv("MIN_SAMPLES_FOR_DRIFT", "100")),
            'feature_drift_methods': ['ks_test', 'psi', 'wasserstein'],
            'performance_metrics': ['mse', 'mae', 'accuracy', 'sharpe_ratio']
        }
        
        # A/B testing configuration
        self.ab_test_config = {
            'min_test_duration_hours': int(os.getenv("AB_TEST_MIN_DURATION_HOURS", "24")),
            'max_test_duration_hours': int(os.getenv("AB_TEST_MAX_DURATION_HOURS", "168")),  # 1 week
            'min_samples_per_variant': int(os.getenv("AB_TEST_MIN_SAMPLES", "1000")),
            'confidence_level': float(os.getenv("AB_TEST_CONFIDENCE_LEVEL", "0.95")),
            'statistical_significance_threshold': float(os.getenv("AB_TEST_SIGNIFICANCE_THRESHOLD", "0.05")),
            'minimum_effect_size': float(os.getenv("AB_TEST_MIN_EFFECT_SIZE", "0.02"))
        }
        
        # Deployment configuration
        self.deployment_config = {
            'strategies': ['blue_green', 'canary', 'rolling'],
            'default_strategy': os.getenv("DEFAULT_DEPLOYMENT_STRATEGY", "blue_green"),
            'canary_traffic_percentage': float(os.getenv("CANARY_TRAFFIC_PERCENTAGE", "0.1")),
            'rollback_threshold': float(os.getenv("ROLLBACK_THRESHOLD", "0.05")),
            'health_check_timeout_seconds': int(os.getenv("HEALTH_CHECK_TIMEOUT", "30")),
            'deployment_timeout_minutes': int(os.getenv("DEPLOYMENT_TIMEOUT_MINUTES", "10"))
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'metrics_retention_days': int(os.getenv("METRICS_RETENTION_DAYS", "90")),
            'alert_thresholds': {
                'error_rate': float(os.getenv("ALERT_ERROR_RATE", "0.05")),
                'latency_p95_ms': int(os.getenv("ALERT_LATENCY_P95", "1000")),
                'drift_score': float(os.getenv("ALERT_DRIFT_SCORE", "0.1")),
                'performance_degradation': float(os.getenv("ALERT_PERFORMANCE_DEGRADATION", "0.1"))
            },
            'notification_channels': {
                'email': os.getenv("ALERT_EMAIL", ""),
                'slack_webhook': os.getenv("SLACK_WEBHOOK_URL", ""),
                'teams_webhook': os.getenv("TEAMS_WEBHOOK_URL", "")
            }
        }
        
        # Scheduler configuration
        self.scheduler_config = {
            'default_schedules': {
                'lstm_daily': {
                    'schedule_type': 'daily',
                    'hour': 2,
                    'minute': 0
                },
                'transformer_weekly': {
                    'schedule_type': 'weekly',
                    'weekday': 0,  # Monday
                    'hour': 3,
                    'minute': 0
                },
                'ensemble_monthly': {
                    'schedule_type': 'monthly',
                    'day': 1,
                    'hour': 4,
                    'minute': 0
                }
            },
            'auto_retrain_triggers': {
                'performance_degradation': True,
                'drift_detection': True,
                'schedule_based': True
            }
        }
        
        # Model storage configuration
        self.storage_config = {
            'model_format': os.getenv("MODEL_FORMAT", "pickle"),  # pickle, joblib, tensorflow
            'compression': os.getenv("MODEL_COMPRESSION", "gzip"),
            'versioning': True,
            'max_versions_per_model': int(os.getenv("MAX_MODEL_VERSIONS", "10")),
            'cleanup_deprecated_after_days': int(os.getenv("CLEANUP_DEPRECATED_DAYS", "30"))
        }
        
        # Feature store configuration
        self.feature_store_config = {
            'enabled': os.getenv("FEATURE_STORE_ENABLED", "true").lower() == "true",
            'backend': os.getenv("FEATURE_STORE_BACKEND", "redis"),  # redis, postgres, s3
            'feature_ttl_hours': int(os.getenv("FEATURE_TTL_HOURS", "24")),
            'batch_size': int(os.getenv("FEATURE_BATCH_SIZE", "1000"))
        }
        
        # Security configuration
        self.security_config = {
            'model_encryption': os.getenv("MODEL_ENCRYPTION_ENABLED", "false").lower() == "true",
            'encryption_key': os.getenv("MODEL_ENCRYPTION_KEY", ""),
            'audit_logging': True,
            'access_control': {
                'require_approval_for_production': True,
                'allowed_deployment_users': os.getenv("ALLOWED_DEPLOYMENT_USERS", "").split(","),
                'model_access_roles': ["data_scientist", "ml_engineer", "admin"]
            }
        }
        
        # Performance optimization
        self.performance_config = {
            'model_caching': True,
            'prediction_caching_ttl_seconds': int(os.getenv("PREDICTION_CACHE_TTL", "300")),
            'batch_prediction_size': int(os.getenv("BATCH_PREDICTION_SIZE", "1000")),
            'async_training': True,
            'gpu_enabled': os.getenv("GPU_ENABLED", "false").lower() == "true",
            'max_memory_gb': int(os.getenv("MAX_MEMORY_GB", "8"))
        }
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type."""
        model_configs = {
            'lstm': {
                'sequence_length': 60,
                'features': ['open', 'high', 'low', 'close', 'volume'],
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'early_stopping_patience': 10
            },
            'transformer': {
                'sequence_length': 100,
                'features': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower'],
                'embed_dim': 128,
                'num_heads': 8,
                'ff_dim': 256,
                'num_layers': 4,
                'dropout_rate': 0.1,
                'learning_rate': 0.0001,
                'batch_size': 16,
                'epochs': 150,
                'early_stopping_patience': 15
            },
            'ensemble': {
                'models': ['lstm', 'transformer', 'linear'],
                'weights': [0.4, 0.4, 0.2],
                'voting_method': 'weighted_average',
                'meta_learner': 'linear_regression'
            }
        }
        
        return model_configs.get(model_type, {})
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for specific trading symbol."""
        symbol_configs = {
            'BTC/USDT': {
                'data_frequency': '1h',
                'lookback_days': 30,
                'volatility_window': 24,
                'min_price_change': 0.001,
                'max_position_size': 1.0
            },
            'ETH/USDT': {
                'data_frequency': '1h',
                'lookback_days': 30,
                'volatility_window': 24,
                'min_price_change': 0.002,
                'max_position_size': 10.0
            }
        }
        
        # Default configuration
        default_config = {
            'data_frequency': '1h',
            'lookback_days': 30,
            'volatility_window': 24,
            'min_price_change': 0.001,
            'max_position_size': 1.0
        }
        
        return symbol_configs.get(symbol, default_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'training_config': self.training_config,
            'validation_thresholds': self.validation_thresholds,
            'drift_config': self.drift_config,
            'ab_test_config': self.ab_test_config,
            'deployment_config': self.deployment_config,
            'monitoring_config': self.monitoring_config,
            'scheduler_config': self.scheduler_config,
            'storage_config': self.storage_config,
            'feature_store_config': self.feature_store_config,
            'security_config': self.security_config,
            'performance_config': self.performance_config
        }


# Global configuration instance
mlops_config = MLOpsConfig()