"""
Database infrastructure setup and management.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from decimal import Decimal
import asyncpg
import aioredis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Boolean, Text, ForeignKey
import structlog

logger = structlog.get_logger(__name__)

# Database Configuration
class DatabaseConfig:
    """Database configuration settings."""
    
    def __init__(self):
        self.postgres_url = os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://trading_user:trading_password@localhost:5432/trading_platform"
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.influxdb_token = os.getenv("INFLUXDB_TOKEN", "")
        self.influxdb_org = os.getenv("INFLUXDB_ORG", "trading-platform")
        self.influxdb_bucket = os.getenv("INFLUXDB_BUCKET", "market-data")


# SQLAlchemy Base and Models
Base = declarative_base()


class Trade(Base):
    """Trade execution records."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    exchange = Column(String(50), nullable=False)
    strategy_id = Column(String(100), nullable=False, index=True)
    order_id = Column(String(100), nullable=False, unique=True)
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    fees = Column(Numeric(20, 8), default=0)
    status = Column(String(20), default="FILLED")


class Position(Base):
    """Current positions."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    avg_price = Column(Numeric(20, 8), nullable=False)
    unrealized_pnl = Column(Numeric(20, 8), default=0)
    strategy_id = Column(String(100), nullable=False, index=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Strategy(Base):
    """Trading strategies."""
    __tablename__ = "strategies"
    
    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    parameters = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# PostgreSQL Connection Manager
class PostgreSQLManager:
    """PostgreSQL database manager for relational data."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        
    async def initialize(self):
        """Initialize PostgreSQL connection."""
        try:
            self.engine = create_async_engine(
                self.config.postgres_url,
                echo=False,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info("PostgreSQL initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        if not self.session_factory:
            await self.initialize()
        return self.session_factory()
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgreSQL connections closed")


# Redis Connection Manager
class RedisManager:
    """Redis manager for high-performance caching."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.redis = None
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Redis initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.redis:
            await self.initialize()
        return await self.redis.get(key)
    
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self.redis:
            await self.initialize()
        return await self.redis.set(key, value, ex=expire)
    
    async def delete(self, key: str) -> int:
        """Delete key from Redis."""
        if not self.redis:
            await self.initialize()
        return await self.redis.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.redis:
            await self.initialize()
        return await self.redis.exists(key)
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        if not self.redis:
            await self.initialize()
        return await self.redis.hset(name, mapping=mapping)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        if not self.redis:
            await self.initialize()
        return await self.redis.hget(name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        if not self.redis:
            await self.initialize()
        return await self.redis.hgetall(name)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")


# InfluxDB Connection Manager
class InfluxDBManager:
    """InfluxDB manager for time-series market data."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self.write_api = None
        self.query_api = None
        
    async def initialize(self):
        """Initialize InfluxDB connection."""
        try:
            self.client = InfluxDBClientAsync(
                url=self.config.influxdb_url,
                token=self.config.influxdb_token,
                org=self.config.influxdb_org
            )
            
            self.write_api = self.client.write_api()
            self.query_api = self.client.query_api()
            
            # Test connection
            await self.client.ping()
            logger.info("InfluxDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {e}")
            raise
    
    async def write_market_data(self, 
                              symbol: str, 
                              price: float, 
                              volume: float, 
                              exchange: str,
                              timestamp: Optional[datetime] = None) -> bool:
        """Write market data point."""
        if not self.write_api:
            await self.initialize()
            
        try:
            point = Point("market_data") \
                .tag("symbol", symbol) \
                .tag("exchange", exchange) \
                .field("price", price) \
                .field("volume", volume)
            
            if timestamp:
                point = point.time(timestamp)
                
            await self.write_api.write(
                bucket=self.config.influxdb_bucket,
                record=point
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to write market data: {e}")
            return False
    
    async def write_trade_data(self,
                             symbol: str,
                             side: str,
                             quantity: float,
                             price: float,
                             exchange: str,
                             strategy_id: str,
                             timestamp: Optional[datetime] = None) -> bool:
        """Write trade execution data."""
        if not self.write_api:
            await self.initialize()
            
        try:
            point = Point("trades") \
                .tag("symbol", symbol) \
                .tag("side", side) \
                .tag("exchange", exchange) \
                .tag("strategy_id", strategy_id) \
                .field("quantity", quantity) \
                .field("price", price) \
                .field("value", quantity * price)
            
            if timestamp:
                point = point.time(timestamp)
                
            await self.write_api.write(
                bucket=self.config.influxdb_bucket,
                record=point
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to write trade data: {e}")
            return False
    
    async def query_market_data(self, 
                              symbol: str, 
                              exchange: str,
                              start_time: str = "-1h") -> List[Dict[str, Any]]:
        """Query market data."""
        if not self.query_api:
            await self.initialize()
            
        try:
            query = f'''
                from(bucket: "{self.config.influxdb_bucket}")
                |> range(start: {start_time})
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            result = await self.query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'time': record.get_time(),
                        'symbol': record.values.get('symbol'),
                        'exchange': record.values.get('exchange'),
                        'price': record.values.get('price'),
                        'volume': record.values.get('volume')
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to query market data: {e}")
            return []
    
    async def close(self):
        """Close InfluxDB connection."""
        if self.client:
            await self.client.close()
            logger.info("InfluxDB connection closed")


# Database Manager - Central coordinator
class DatabaseManager:
    """Central database manager coordinating all database systems."""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.postgres = PostgreSQLManager(self.config)
        self.redis = RedisManager(self.config)
        self.influxdb = InfluxDBManager(self.config)
        
    async def initialize_all(self):
        """Initialize all database connections."""
        try:
            await asyncio.gather(
                self.postgres.initialize(),
                self.redis.initialize(),
                self.influxdb.initialize()
            )
            logger.info("All database systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database systems: {e}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database systems."""
        health = {}
        
        # PostgreSQL health check
        try:
            async with self.postgres.get_session() as session:
                await session.execute("SELECT 1")
            health['postgresql'] = True
        except Exception:
            health['postgresql'] = False
        
        # Redis health check
        try:
            await self.redis.redis.ping()
            health['redis'] = True
        except Exception:
            health['redis'] = False
        
        # InfluxDB health check
        try:
            await self.influxdb.client.ping()
            health['influxdb'] = True
        except Exception:
            health['influxdb'] = False
        
        return health
    
    async def close_all(self):
        """Close all database connections."""
        await asyncio.gather(
            self.postgres.close(),
            self.redis.close(),
            self.influxdb.close()
        )
        logger.info("All database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions for easy access
async def get_postgres_session() -> AsyncSession:
    """Get PostgreSQL session."""
    return await db_manager.postgres.get_session()


async def get_redis() -> aioredis.Redis:
    """Get Redis client."""
    if not db_manager.redis.redis:
        await db_manager.redis.initialize()
    return db_manager.redis.redis


async def get_influxdb() -> InfluxDBClientAsync:
    """Get InfluxDB client."""
    if not db_manager.influxdb.client:
        await db_manager.influxdb.initialize()
    return db_manager.influxdb.client


# Database initialization function
async def initialize_databases():
    """Initialize all database systems."""
    await db_manager.initialize_all()


# Database cleanup function
async def cleanup_databases():
    """Cleanup all database connections."""
    await db_manager.close_all()