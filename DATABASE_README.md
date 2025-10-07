# Database Infrastructure

This document describes the database infrastructure for the Advanced Trading Platform, which uses a multi-database architecture to handle different types of data efficiently.

## Architecture Overview

The platform uses three database systems:

1. **PostgreSQL** - Relational data (trades, positions, strategies, users)
2. **Redis** - High-performance caching and session storage
3. **InfluxDB** - Time-series market data and metrics

## Database Systems

### PostgreSQL (Relational Data)

**Purpose**: Stores structured relational data including trades, positions, strategies, and user information.

**Configuration**:

- Host: `postgres:5432` (Docker) / `localhost:5432` (Local)
- Database: `trading_platform`
- User: `trading_user`
- Password: `trading_password`

**Key Tables**:

- `trades` - Trade execution records
- `positions` - Current positions
- `strategies` - Trading strategy definitions
- `users` - User accounts (future implementation)

**Features**:

- Automatic indexing for performance
- Triggers for timestamp updates
- Views for common queries
- Connection pooling (20 connections, 30 max overflow)

### Redis (Caching & Sessions)

**Purpose**: High-performance caching, session storage, and real-time data.

**Configuration**:

- Host: `redis:6379` (Docker) / `localhost:6379` (Local)
- No authentication (development)
- Max connections: 20

**Usage Patterns**:

- Configuration caching (`config:*`)
- Exchange settings (`exchanges:config`)
- Session data (`session:*`)
- Real-time price caching (`price:*`)
- System status (`system:*`)

### InfluxDB (Time-Series Data)

**Purpose**: Stores time-series market data, trading metrics, and performance data.

**Configuration**:

- Host: `influxdb:8086` (Docker) / `localhost:8086` (Local)
- Organization: `trading-platform`
- Bucket: `market-data`
- User: `admin`
- Password: `adminpassword`

**Measurements**:

- `market_data` - Price and volume data
- `trades` - Trade execution metrics
- `performance` - Strategy performance metrics
- `system_metrics` - System health metrics

## Setup Instructions

### 1. Using Docker Compose (Recommended)

```bash
# Start all database services
cd advanced_trading_platform
docker-compose up -d postgres redis influxdb

# Wait for services to be ready (30-60 seconds)
docker-compose logs -f postgres redis influxdb

# Run database setup script
python scripts/setup_databases.py

# Verify setup
python scripts/check_databases.py
```

### 2. Manual Setup

#### PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE trading_platform;
CREATE USER trading_user WITH PASSWORD 'trading_password';
GRANT ALL PRIVILEGES ON DATABASE trading_platform TO trading_user;
\q

# Run initialization script
psql -U trading_user -d trading_platform -f scripts/init-db.sql
```

#### Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### InfluxDB Setup

```bash
# Install InfluxDB 2.x
wget -qO- https://repos.influxdata.com/influxdb.key | sudo apt-key add -
echo "deb https://repos.influxdata.com/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/influxdb.list
sudo apt-get update && sudo apt-get install influxdb2

# Start InfluxDB
sudo systemctl start influxdb
sudo systemctl enable influxdb

# Setup initial configuration
influx setup --org trading-platform --bucket market-data --username admin --password adminpassword
```

## Database Management

### Health Checks

```bash
# Check all databases
python scripts/check_databases.py

# Test database operations
python scripts/check_databases.py test
```

### Migrations

```bash
# Initialize Alembic (first time only)
alembic init migrations

# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Backup and Restore

#### PostgreSQL

```bash
# Backup
pg_dump -U trading_user -h localhost trading_platform > backup.sql

# Restore
psql -U trading_user -h localhost trading_platform < backup.sql
```

#### Redis

```bash
# Backup (Redis saves automatically to dump.rdb)
redis-cli BGSAVE

# Restore (copy dump.rdb to Redis data directory and restart)
```

#### InfluxDB

```bash
# Backup
influx backup /path/to/backup --org trading-platform

# Restore
influx restore /path/to/backup --org trading-platform
```

## Performance Optimization

### PostgreSQL

- Indexes on frequently queried columns
- Connection pooling (20 connections)
- Query optimization with EXPLAIN ANALYZE
- Regular VACUUM and ANALYZE

### Redis

- Memory optimization with appropriate data types
- TTL for temporary data
- Pipeline operations for bulk operations
- Monitor memory usage

### InfluxDB

- Appropriate retention policies
- Batch writes for better performance
- Proper tag vs field selection
- Downsampling for long-term storage

## Monitoring

### Prometheus Metrics

- Database connection counts
- Query performance
- Error rates
- Resource usage

### Grafana Dashboards

- Database performance metrics
- Query execution times
- Connection pool status
- Storage usage

### Health Endpoints

- `/health/postgres` - PostgreSQL status
- `/health/redis` - Redis status
- `/health/influxdb` - InfluxDB status

## Security Considerations

### Development Environment

- Default passwords (change in production)
- No SSL/TLS (enable in production)
- Local network access only

### Production Recommendations

- Strong passwords and secrets management
- SSL/TLS encryption
- Network security (VPC, firewalls)
- Regular security updates
- Backup encryption
- Access logging and monitoring

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if services are running: `docker-compose ps`
   - Verify network connectivity
   - Check firewall settings

2. **Authentication Failed**
   - Verify credentials in environment variables
   - Check user permissions
   - Ensure database exists

3. **Performance Issues**
   - Monitor connection pool usage
   - Check for slow queries
   - Verify adequate resources (CPU, memory, disk)

4. **Data Inconsistency**
   - Check transaction isolation levels
   - Verify foreign key constraints
   - Review concurrent access patterns

### Logs and Debugging

```bash
# Docker logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs influxdb

# Application logs
tail -f logs/database.log

# Database-specific logs
# PostgreSQL: /var/log/postgresql/
# Redis: /var/log/redis/
# InfluxDB: /var/log/influxdb/
```

## Environment Variables

```bash
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://trading_user:trading_password@localhost:5432/trading_platform

# Redis
REDIS_URL=redis://localhost:6379

# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token-here
INFLUXDB_ORG=trading-platform
INFLUXDB_BUCKET=market-data
```

## API Usage Examples

### PostgreSQL (via SQLAlchemy)

```python
from shared.database import get_postgres_session, Trade

async with get_postgres_session() as session:
    # Create a new trade
    trade = Trade(
        symbol="BTC/USDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        exchange="binance",
        strategy_id="arbitrage-btc-usdt"
    )
    session.add(trade)
    await session.commit()
```

### Redis

```python
from shared.database import get_redis

redis = await get_redis()
await redis.set("price:BTC/USDT", "50000.0", ex=60)
price = await redis.get("price:BTC/USDT")
```

### InfluxDB

```python
from shared.database import db_manager

await db_manager.influxdb.write_market_data(
    symbol="BTC/USDT",
    price=50000.0,
    volume=1.5,
    exchange="binance"
)

data = await db_manager.influxdb.query_market_data(
    symbol="BTC/USDT",
    exchange="binance",
    start_time="-1h"
)
```
