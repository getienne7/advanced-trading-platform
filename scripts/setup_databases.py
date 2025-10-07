#!/usr/bin/env python3
"""
Database setup script for the Advanced Trading Platform.
This script initializes all database systems and creates necessary schemas.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

from database import DatabaseManager, DatabaseConfig
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def setup_postgresql():
    """Set up PostgreSQL database with initial data."""
    logger.info("Setting up PostgreSQL...")
    
    db_manager = DatabaseManager()
    await db_manager.postgres.initialize()
    
    # Create sample data
    async with db_manager.postgres.get_session() as session:
        from database import Strategy, Trade, Position
        from sqlalchemy import select
        
        # Check if strategies already exist
        result = await session.execute(select(Strategy))
        existing_strategies = result.scalars().all()
        
        if not existing_strategies:
            # Create default strategies
            strategies = [
                Strategy(
                    id="arbitrage-btc-usdt",
                    name="BTC/USDT Arbitrage",
                    description="Arbitrage strategy for BTC/USDT across multiple exchanges",
                    parameters='{"min_profit_threshold": 0.001, "max_position_size": 1.0, "exchanges": ["binance", "coinbase"]}'
                ),
                Strategy(
                    id="momentum-eth-usdt",
                    name="ETH/USDT Momentum",
                    description="Momentum trading strategy for ETH/USDT",
                    parameters='{"lookback_period": 20, "momentum_threshold": 0.02, "stop_loss": 0.05}'
                ),
                Strategy(
                    id="mean-reversion-ada-usdt",
                    name="ADA/USDT Mean Reversion",
                    description="Mean reversion strategy for ADA/USDT",
                    parameters='{"bollinger_period": 20, "bollinger_std": 2, "rsi_period": 14}'
                )
            ]
            
            for strategy in strategies:
                session.add(strategy)
            
            await session.commit()
            logger.info(f"Created {len(strategies)} default strategies")
        else:
            logger.info(f"Found {len(existing_strategies)} existing strategies")
    
    await db_manager.postgres.close()
    logger.info("PostgreSQL setup completed")


async def setup_redis():
    """Set up Redis with initial cache data."""
    logger.info("Setting up Redis...")
    
    db_manager = DatabaseManager()
    await db_manager.redis.initialize()
    
    # Set up initial cache data
    cache_data = {
        "system:status": "initialized",
        "system:version": "1.0.0",
        "config:max_positions": "10",
        "config:risk_limit": "0.02"
    }
    
    for key, value in cache_data.items():
        await db_manager.redis.set(key, value)
    
    # Set up hash for exchange configurations
    exchange_config = {
        "binance": '{"api_key": "", "secret": "", "sandbox": true}',
        "coinbase": '{"api_key": "", "secret": "", "passphrase": "", "sandbox": true}',
        "bitunix": '{"api_key": "", "secret": "", "sandbox": true}'
    }
    
    await db_manager.redis.hset("exchanges:config", exchange_config)
    
    await db_manager.redis.close()
    logger.info("Redis setup completed")


async def setup_influxdb():
    """Set up InfluxDB with initial buckets and data."""
    logger.info("Setting up InfluxDB...")
    
    db_manager = DatabaseManager()
    await db_manager.influxdb.initialize()
    
    # Write some sample market data
    sample_data = [
        {"symbol": "BTC/USDT", "price": 50000.0, "volume": 1.5, "exchange": "binance"},
        {"symbol": "ETH/USDT", "price": 3000.0, "volume": 10.0, "exchange": "binance"},
        {"symbol": "ADA/USDT", "price": 0.5, "volume": 1000.0, "exchange": "coinbase"},
    ]
    
    for data in sample_data:
        await db_manager.influxdb.write_market_data(**data)
    
    await db_manager.influxdb.close()
    logger.info("InfluxDB setup completed")


async def verify_setup():
    """Verify that all databases are properly set up."""
    logger.info("Verifying database setup...")
    
    db_manager = DatabaseManager()
    await db_manager.initialize_all()
    
    # Verify PostgreSQL
    async with db_manager.postgres.get_session() as session:
        from database import Strategy
        from sqlalchemy import select, func
        
        result = await session.execute(select(func.count(Strategy.id)))
        strategy_count = result.scalar()
        logger.info(f"PostgreSQL: {strategy_count} strategies found")
    
    # Verify Redis
    system_status = await db_manager.redis.get("system:status")
    exchange_count = len(await db_manager.redis.hgetall("exchanges:config"))
    logger.info(f"Redis: System status = {system_status}, {exchange_count} exchanges configured")
    
    # Verify InfluxDB
    market_data = await db_manager.influxdb.query_market_data("BTC/USDT", "binance", "-1d")
    logger.info(f"InfluxDB: {len(market_data)} market data points found")
    
    await db_manager.close_all()
    logger.info("Database setup verification completed successfully!")


async def main():
    """Main setup function."""
    try:
        logger.info("Starting database setup for Advanced Trading Platform...")
        
        # Set up each database system
        await setup_postgresql()
        await setup_redis()
        await setup_influxdb()
        
        # Verify the setup
        await verify_setup()
        
        logger.info("Database setup completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)