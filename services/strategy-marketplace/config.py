"""
Configuration settings for Strategy Marketplace Service
"""

import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "STRATEGY_MARKETPLACE_DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/strategy_marketplace"
    )
    
    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # Service URLs
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
    TRADING_ENGINE_URL: str = os.getenv("TRADING_ENGINE_URL", "http://localhost:8002")
    STRATEGY_ENGINE_URL: str = os.getenv("STRATEGY_ENGINE_URL", "http://localhost:8003")
    USER_SERVICE_URL: str = os.getenv("USER_SERVICE_URL", "http://localhost:8001")
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Platform settings
    PLATFORM_FEE_RATE: float = float(os.getenv("PLATFORM_FEE_RATE", "0.3"))  # 30%
    MAX_PERFORMANCE_FEE: float = float(os.getenv("MAX_PERFORMANCE_FEE", "0.5"))  # 50%
    MAX_SUBSCRIPTION_FEE: float = float(os.getenv("MAX_SUBSCRIPTION_FEE", "1000.0"))
    
    # Performance calculation settings
    PERFORMANCE_CALCULATION_INTERVAL: int = int(os.getenv("PERFORMANCE_CALCULATION_INTERVAL", "3600"))  # 1 hour
    LEADERBOARD_UPDATE_INTERVAL: int = int(os.getenv("LEADERBOARD_UPDATE_INTERVAL", "86400"))  # 24 hours
    
    # Copy trading settings
    MAX_COPY_TRADE_DELAY: int = int(os.getenv("MAX_COPY_TRADE_DELAY", "5"))  # 5 seconds
    DEFAULT_SLIPPAGE_TOLERANCE: float = float(os.getenv("DEFAULT_SLIPPAGE_TOLERANCE", "0.001"))  # 0.1%
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SQL_DEBUG: bool = os.getenv("SQL_DEBUG", "false").lower() == "true"
    
    # API settings
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "100"))  # requests per minute
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))  # seconds
    
    # Security settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    ALLOWED_HOSTS: list = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    
    # Monitoring settings
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds

# Create global settings instance
settings = Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    SQL_DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    """Production environment settings"""
    SQL_DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    
    # Override with production database
    DATABASE_URL: str = os.getenv(
        "STRATEGY_MARKETPLACE_DATABASE_URL",
        "postgresql://user:password@prod-db:5432/strategy_marketplace"
    )

class TestSettings(Settings):
    """Test environment settings"""
    DATABASE_URL: str = "sqlite:///./test_strategy_marketplace.db"
    JWT_SECRET_KEY: str = "test-secret-key"
    SQL_DEBUG: bool = True

def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()