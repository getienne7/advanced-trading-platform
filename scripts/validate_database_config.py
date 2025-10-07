#!/usr/bin/env python3
"""
Database configuration validation script.
This script validates the database configuration without requiring running services.
"""
import sys
import os
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

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


def validate_database_imports():
    """Validate that all required database packages are available."""
    logger.info("Validating database imports...")
    
    try:
        import asyncpg
        logger.info("✓ asyncpg available")
    except ImportError as e:
        logger.error(f"✗ asyncpg not available: {e}")
        return False
    
    try:
        import aioredis
        logger.info("✓ aioredis available")
    except ImportError as e:
        logger.error(f"✗ aioredis not available: {e}")
        return False
    
    try:
        from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
        logger.info("✓ InfluxDB client available")
    except ImportError as e:
        logger.error(f"✗ InfluxDB client not available: {e}")
        return False
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        logger.info("✓ SQLAlchemy async available")
    except ImportError as e:
        logger.error(f"✗ SQLAlchemy async not available: {e}")
        return False
    
    return True


def validate_database_config():
    """Validate database configuration."""
    logger.info("Validating database configuration...")
    
    try:
        from database import DatabaseConfig
        
        config = DatabaseConfig()
        
        # Validate PostgreSQL URL
        if config.postgres_url:
            logger.info(f"✓ PostgreSQL URL configured: {config.postgres_url}")
        else:
            logger.error("✗ PostgreSQL URL not configured")
            return False
        
        # Validate Redis URL
        if config.redis_url:
            logger.info(f"✓ Redis URL configured: {config.redis_url}")
        else:
            logger.error("✗ Redis URL not configured")
            return False
        
        # Validate InfluxDB configuration
        if config.influxdb_url:
            logger.info(f"✓ InfluxDB URL configured: {config.influxdb_url}")
        else:
            logger.error("✗ InfluxDB URL not configured")
            return False
        
        if config.influxdb_org:
            logger.info(f"✓ InfluxDB organization configured: {config.influxdb_org}")
        else:
            logger.error("✗ InfluxDB organization not configured")
            return False
        
        if config.influxdb_bucket:
            logger.info(f"✓ InfluxDB bucket configured: {config.influxdb_bucket}")
        else:
            logger.error("✗ InfluxDB bucket not configured")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Database configuration validation failed: {e}")
        return False


def validate_database_models():
    """Validate database models."""
    logger.info("Validating database models...")
    
    try:
        from database import Base, Trade, Position, Strategy
        
        # Check that models are properly defined
        models = [Trade, Position, Strategy]
        for model in models:
            if hasattr(model, '__tablename__'):
                logger.info(f"✓ Model {model.__name__} properly defined with table '{model.__tablename__}'")
            else:
                logger.error(f"✗ Model {model.__name__} missing __tablename__")
                return False
        
        # Check that Base metadata contains tables
        table_names = list(Base.metadata.tables.keys())
        expected_tables = ['trades', 'positions', 'strategies']
        
        for table_name in expected_tables:
            if table_name in table_names:
                logger.info(f"✓ Table '{table_name}' defined in metadata")
            else:
                logger.error(f"✗ Table '{table_name}' missing from metadata")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Database models validation failed: {e}")
        return False


def validate_database_managers():
    """Validate database manager classes."""
    logger.info("Validating database managers...")
    
    try:
        from database import PostgreSQLManager, RedisManager, InfluxDBManager, DatabaseManager, DatabaseConfig
        
        config = DatabaseConfig()
        
        # Test manager instantiation
        postgres_manager = PostgreSQLManager(config)
        logger.info("✓ PostgreSQLManager instantiated successfully")
        
        redis_manager = RedisManager(config)
        logger.info("✓ RedisManager instantiated successfully")
        
        influxdb_manager = InfluxDBManager(config)
        logger.info("✓ InfluxDBManager instantiated successfully")
        
        db_manager = DatabaseManager()
        logger.info("✓ DatabaseManager instantiated successfully")
        
        # Check that managers have required methods
        required_methods = {
            'PostgreSQLManager': ['initialize', 'get_session', 'close'],
            'RedisManager': ['initialize', 'get', 'set', 'close'],
            'InfluxDBManager': ['initialize', 'write_market_data', 'query_market_data', 'close'],
            'DatabaseManager': ['initialize_all', 'health_check', 'close_all']
        }
        
        managers = {
            'PostgreSQLManager': postgres_manager,
            'RedisManager': redis_manager,
            'InfluxDBManager': influxdb_manager,
            'DatabaseManager': db_manager
        }
        
        for manager_name, manager in managers.items():
            for method_name in required_methods[manager_name]:
                if hasattr(manager, method_name):
                    logger.info(f"✓ {manager_name}.{method_name} method available")
                else:
                    logger.error(f"✗ {manager_name}.{method_name} method missing")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Database managers validation failed: {e}")
        return False


def validate_docker_configuration():
    """Validate Docker configuration."""
    logger.info("Validating Docker configuration...")
    
    try:
        # Check if docker-compose.yml exists
        docker_compose_path = Path("docker-compose.yml")
        if docker_compose_path.exists():
            logger.info("✓ docker-compose.yml file exists")
        else:
            logger.error("✗ docker-compose.yml file missing")
            return False
        
        # Check if init script exists
        init_script_path = Path("scripts/init-db.sql")
        if init_script_path.exists():
            logger.info("✓ Database initialization script exists")
        else:
            logger.error("✗ Database initialization script missing")
            return False
        
        # Check if monitoring configuration exists
        prometheus_config = Path("monitoring/prometheus.yml")
        if prometheus_config.exists():
            logger.info("✓ Prometheus configuration exists")
        else:
            logger.error("✗ Prometheus configuration missing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Docker configuration validation failed: {e}")
        return False


def main():
    """Main validation function."""
    logger.info("Starting database infrastructure validation...")
    
    validations = [
        ("Database Imports", validate_database_imports),
        ("Database Configuration", validate_database_config),
        ("Database Models", validate_database_models),
        ("Database Managers", validate_database_managers),
        ("Docker Configuration", validate_docker_configuration)
    ]
    
    all_passed = True
    
    for validation_name, validation_func in validations:
        logger.info(f"\n--- {validation_name} ---")
        if validation_func():
            logger.info(f"✓ {validation_name} validation passed")
        else:
            logger.error(f"✗ {validation_name} validation failed")
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All database infrastructure validations passed!")
        logger.info("The database infrastructure is properly configured.")
        logger.info("To start the services, run: docker-compose up -d postgres redis influxdb")
        return 0
    else:
        logger.error("\n❌ Some validations failed!")
        logger.error("Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)