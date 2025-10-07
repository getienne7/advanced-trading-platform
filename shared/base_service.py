"""
Base microservice class with common functionality for all services.
"""
import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, start_http_server


class ServiceConfig(BaseSettings):
    """Base configuration for all services"""
    service_name: str
    service_version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Database
    database_url: str
    redis_url: str
    
    # Message Queue
    rabbitmq_url: str
    
    # Monitoring
    metrics_port: int = 8080
    health_check_port: int = 8081
    
    # Service Discovery
    consul_host: str = "localhost"
    consul_port: int = 8500
    
    class Config:
        env_file = ".env"


class BaseService(ABC):
    """Base class for all microservices"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False
        
        # Database connections
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None
        
        # Metrics
        self.setup_metrics()
        
        # Graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> structlog.BoundLogger:
        """Setup structured logging"""
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
        
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, self.config.log_level.upper())
        )
        
        return structlog.get_logger(service=self.config.service_name)
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        self.request_count = Counter(
            'service_requests_total',
            'Total service requests',
            ['service', 'method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'service_request_duration_seconds',
            'Service request duration',
            ['service', 'method', 'endpoint']
        )
        
        self.active_connections = Gauge(
            'service_active_connections',
            'Active connections',
            ['service']
        )
        
        self.health_status = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service']
        )
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize service resources"""
        self.logger.info("Initializing service", service=self.config.service_name)
        
        # Initialize database connections
        await self._init_database()
        await self._init_redis()
        
        # Start metrics server
        start_http_server(self.config.metrics_port)
        
        # Service-specific initialization
        await self.init_service()
        
        self.health_status.labels(service=self.config.service_name).set(1)
        self.logger.info("Service initialized successfully")
    
    async def _init_database(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            self.logger.info("Database connection pool created")
        except Exception as e:
            self.logger.error("Failed to create database pool", error=str(e))
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis = aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def run(self):
        """Main service run loop"""
        try:
            await self.initialize()
            self.running = True
            
            self.logger.info("Service started", service=self.config.service_name)
            
            # Start service-specific tasks
            await self.start_service()
            
            # Keep service running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error("Service error", error=str(e))
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        if not self.running:
            return
            
        self.logger.info("Shutting down service")
        self.running = False
        
        # Service-specific cleanup
        await self.cleanup_service()
        
        # Close connections
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis:
            await self.redis.close()
        
        self.health_status.labels(service=self.config.service_name).set(0)
        self.logger.info("Service shutdown complete")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Check database
            db_healthy = False
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                db_healthy = True
            
            # Check Redis
            redis_healthy = False
            if self.redis:
                await self.redis.ping()
                redis_healthy = True
            
            # Service-specific health checks
            service_healthy = await self.service_health_check()
            
            healthy = db_healthy and redis_healthy and service_healthy
            
            return {
                "service": self.config.service_name,
                "version": self.config.service_version,
                "healthy": healthy,
                "checks": {
                    "database": db_healthy,
                    "redis": redis_healthy,
                    "service": service_healthy
                },
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "service": self.config.service_name,
                "healthy": False,
                "error": str(e)
            }
    
    # Abstract methods to be implemented by concrete services
    @abstractmethod
    async def init_service(self):
        """Service-specific initialization"""
        pass
    
    @abstractmethod
    async def start_service(self):
        """Start service-specific tasks"""
        pass
    
    @abstractmethod
    async def cleanup_service(self):
        """Service-specific cleanup"""
        pass
    
    @abstractmethod
    async def service_health_check(self) -> bool:
        """Service-specific health check"""
        pass


class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self, consul_host: str, consul_port: int):
        self.consul_host = consul_host
        self.consul_port = consul_port
    
    async def register_service(self, service_name: str, service_id: str, 
                             host: str, port: int, health_check_url: str):
        """Register service with Consul"""
        # Implementation for service registration
        pass
    
    async def deregister_service(self, service_id: str):
        """Deregister service from Consul"""
        # Implementation for service deregistration
        pass
    
    async def discover_service(self, service_name: str) -> Dict[str, Any]:
        """Discover service instances"""
        # Implementation for service discovery
        pass


class MessageBus:
    """Inter-service communication via message queue"""
    
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None
    
    async def connect(self):
        """Connect to message queue"""
        # Implementation for RabbitMQ connection
        pass
    
    async def publish(self, exchange: str, routing_key: str, message: Dict[str, Any]):
        """Publish message to exchange"""
        # Implementation for message publishing
        pass
    
    async def subscribe(self, queue: str, callback):
        """Subscribe to queue messages"""
        # Implementation for message subscription
        pass
    
    async def disconnect(self):
        """Disconnect from message queue"""
        # Implementation for disconnection
        pass