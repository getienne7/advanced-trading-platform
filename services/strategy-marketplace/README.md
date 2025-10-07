# Strategy Marketplace Service

The Strategy Marketplace Service is a core component of the Advanced Trading Platform that handles strategy publication, subscription, performance tracking, and monetization.

## Features

### Strategy Publication

- Publish trading strategies to the marketplace
- Set subscription fees and performance fees
- Categorize strategies by type and risk level
- Code integrity verification with hash validation

### Strategy Subscription

- Subscribe to published strategies
- Configure allocation percentages and risk multipliers
- Automatic copy trading with position scaling
- Real-time trade replication

### Performance Tracking

- Comprehensive performance metrics calculation
- Sharpe ratio, Sortino ratio, and other risk-adjusted returns
- Historical performance tracking and analysis
- Leaderboard rankings and comparisons

### Monetization System

- Subscription-based revenue model
- Performance-based fee sharing
- Automated earnings calculation and distribution
- Platform fee management (30% default)

### Rating and Review System

- User ratings and reviews for strategies
- Verified subscriber ratings
- Performance-based rating categories
- Community feedback integration

## API Endpoints

### Strategy Management

- `POST /strategies` - Publish a new strategy
- `GET /strategies` - List published strategies with filtering
- `GET /strategies/{strategy_id}` - Get strategy details
- `GET /my-strategies` - Get user's published strategies

### Subscription Management

- `POST /strategies/{strategy_id}/subscribe` - Subscribe to a strategy
- `DELETE /strategies/{strategy_id}/subscribe` - Unsubscribe from a strategy
- `GET /my-subscriptions` - Get user's active subscriptions

### Performance and Analytics

- `GET /strategies/{strategy_id}/performance` - Get strategy performance metrics
- `GET /marketplace/stats` - Get marketplace statistics

### Rating System

- `POST /strategies/{strategy_id}/rate` - Rate and review a strategy

### Monetization

- `POST /strategies/{strategy_id}/earnings` - Calculate strategy earnings

## Database Schema

### Core Tables

- `strategies` - Published trading strategies
- `strategy_subscriptions` - User subscriptions to strategies
- `strategy_performance` - Performance metrics and tracking
- `strategy_ratings` - User ratings and reviews
- `strategy_earnings` - Creator earnings and revenue tracking
- `strategy_leaderboard` - Strategy rankings and leaderboards

## Configuration

### Environment Variables

- `STRATEGY_MARKETPLACE_DATABASE_URL` - PostgreSQL database connection
- `JWT_SECRET_KEY` - JWT token secret for authentication
- `REDIS_URL` - Redis connection for caching
- `AUTH_SERVICE_URL` - Authentication service endpoint
- `TRADING_ENGINE_URL` - Trading engine service endpoint
- `STRATEGY_ENGINE_URL` - Strategy engine service endpoint
- `PLATFORM_FEE_RATE` - Platform fee percentage (default: 0.3)

### Service Dependencies

- PostgreSQL database for data persistence
- Redis for caching and session management
- Authentication service for user verification
- Trading engine for copy trading execution
- Strategy engine for strategy management

## Copy Trading System

The service implements a sophisticated copy trading system that:

1. **Signal Processing**: Receives trading signals from strategy creators
2. **Position Scaling**: Adjusts position sizes based on subscriber allocation and risk settings
3. **Risk Management**: Applies position limits and risk multipliers
4. **Execution**: Forwards scaled trades to the trading engine
5. **Tracking**: Monitors copy trading performance and slippage

### Copy Trading Flow

```
Strategy Signal → Position Scaling → Risk Checks → Trade Execution → Performance Tracking
```

## Performance Metrics

The service calculates comprehensive performance metrics including:

- **Return Metrics**: Total return, annualized return, excess return
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR
- **Trading Metrics**: Win rate, profit factor, average trade duration
- **Market Metrics**: Beta, alpha, correlation with market

## Revenue Model

### For Strategy Creators

- **Subscription Fees**: Monthly recurring revenue from subscribers
- **Performance Fees**: Percentage of profits generated (max 50%)
- **Platform Fee**: 30% of total revenue goes to platform

### For Subscribers

- **Subscription Cost**: Monthly fee to access strategy signals
- **Performance Sharing**: Share of profits with strategy creator
- **Trading Costs**: Standard exchange fees and slippage

## Security Features

- JWT-based authentication and authorization
- Input validation and sanitization
- Rate limiting and DDoS protection
- Audit logging for all transactions
- Encrypted sensitive data storage

## Monitoring and Observability

- Health check endpoints for service monitoring
- Performance metrics collection
- Error tracking and alerting
- Database query optimization
- Redis caching for improved performance

## Development

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up database
export STRATEGY_MARKETPLACE_DATABASE_URL="postgresql://user:pass@localhost/db"

# Run migrations
alembic upgrade head

# Start the service
uvicorn app:app --host 0.0.0.0 --port 8007
```

### Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test file
pytest tests/test_strategy_service.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f strategy-marketplace

# Scale the service
docker-compose up -d --scale strategy-marketplace=3
```

## Integration with Other Services

### Authentication Service

- User authentication and authorization
- JWT token validation
- User profile information

### Trading Engine

- Copy trading execution
- Position management
- Trade history and performance data

### Strategy Engine

- Strategy registration and management
- Signal generation and distribution
- Strategy performance calculation

### Analytics Service

- Advanced performance analytics
- Risk analysis and reporting
- Market data integration

## Future Enhancements

- Machine learning-based strategy recommendations
- Social trading features and community building
- Advanced risk management tools
- Mobile app integration
- Institutional client features
- Multi-asset class support
