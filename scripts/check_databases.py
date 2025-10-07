#!/usr/bin/env python3
"""
Database health check utility for the Advanced Trading Platform.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

from database import DatabaseManager
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


async def check_database_health():
    """Check the health of all database systems."""
    db_manager = DatabaseManager()
    
    try:
        logger.info("Starting database health check...")
        
        # Initialize all databases
        await db_manager.initialize_all()
        
        # Perform health checks
        health_status = await db_manager.health_check()
        
        # Report results
        all_healthy = True
        for db_name, is_healthy in health_status.items():
            status = "✓ HEALTHY" if is_healthy else "✗ UNHEALTHY"
            logger.info(f"{db_name.upper()}: {status}")
            if not is_healthy:
                all_healthy = False
        
        if all_healthy:
            logger.info("All database systems are healthy!")
            return 0
        else:
            logger.error("Some database systems are unhealthy!")
            return 1
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return 1
    finally:
        await db_manager.close_all()


async def test_database_operations():
    """Test basic database operations."""
    db_manager = DatabaseManager()
    
    try:
        logger.info("Testing database operations...")
        
        await db_manager.initialize_all()
        
        # Test PostgreSQL
        logger.info("Testing PostgreSQL operations...")
        async with db_manager.postgres.get_session() as session:
            from database import Strategy
            from sqlalchemy import select
            
            # Test query
            result = await session.execute(select(Strategy))
            strategies = result.scalars().all()
            logger.info(f"Found {len(strategies)} strategies in PostgreSQL")
        
        # Test Redis
        logger.info("Testing Redis operations...")
        await db_manager.redis.set("test_key", "test_value", expire=60)
        value = await db_manager.redis.get("test_key")
        assert value == "test_value", "Redis test failed"
        await db_manager.redis.delete("test_key")
        logger.info("Redis operations successful")
        
        # Test InfluxDB
        logger.info("Testing InfluxDB operations...")
        success = await db_manager.influxdb.write_market_data(
            symbol="BTC/USDT",
            price=50000.0,
            volume=1.5,
            exchange="test"
        )
        assert success, "InfluxDB write test failed"
        logger.info("InfluxDB operations successful")
        
        logger.info("All database operations completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Database operations test failed: {e}")
        return 1
    finally:
        await db_manager.close_all()


async def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        return await test_database_operations()
    else:
        return await check_database_health()


if __name__ == "__main__":
    exit_code = asyncio.run(main())