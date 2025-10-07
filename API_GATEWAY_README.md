# API Gateway

The API Gateway serves as the central entry point for all client requests to the Advanced Trading Platform. It handles authentication, authorization, rate limiting, request routing, and monitoring.

## Features

### 🔐 Authentication & Authorization

- JWT-based authentication
- Role-based access control (RBAC)
- Token expiration and refresh
- Secure password hashing

### 🚦 Rate Limiting

- Redis-based distributed rate limiting
- Configurable limits per client IP
- Automatic rate limit enforcement
- Metrics collection for rate limit violations

### 🔀 Request Routing

- Intelligent service discovery
- Load balancing across service instances
- Automatic failover handling
- Request/response transformation

### 📊 Monitoring & Metrics

- Prometheus metrics collection
- Request duration tracking
- Error rate monitoring
- Health check endpoints

### 🛡️ Security

- CORS protection
- Trusted host validation
- Request validation
- Input sanitization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile App     │    │  External API   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │                 │
                    │ • Authentication│
                    │ • Rate Limiting │
                    │ • Routing       │
                    │ • Monitoring    │
                    └─────────┬───────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────▼───────┐ ┌─────────▼───────┐ ┌─────────▼───────┐
│ Trading Engine  │ │  Market Data    │ │   Analytics     │
│   Service       │ │    Service      │ │    Service      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## API Endpoints

### Authentication

#### POST /auth/login

Login with username and password.

**Request:**

```json
{
  "username": "admin",
  "password": "password"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### GET /auth/me

Get current user information.

**Headers:**

```
Authorization: Bearer <token>
```

**Response:**

```json
{
  "user_id": "1",
  "username": "admin",
  "roles": ["admin"],
  "permissions": ["read", "write", "admin"]
}
```

#### POST /auth/logout

Logout current user.

**Headers:**

```
Authorization: Bearer <token>
```

### Service Routing

All service endpoints are prefixed with `/api/` and require authentication.

#### Trading Engine

- `GET /api/trading/positions` - Get current positions
- `POST /api/trading/orders` - Place new order
- `GET /api/trading/orders/{order_id}` - Get order details
- `DELETE /api/trading/orders/{order_id}` - Cancel order

#### Market Data

- `GET /api/market-data/ticker/{symbol}` - Get ticker data
- `GET /api/market-data/orderbook/{symbol}` - Get order book
- `GET /api/market-data/trades/{symbol}` - Get recent trades
- `GET /api/market-data/candles/{symbol}` - Get candlestick data

#### Analytics

- `GET /api/analytics/performance` - Get performance metrics
- `GET /api/analytics/reports` - Get trading reports
- `POST /api/analytics/backtest` - Run backtest

#### AI/ML

- `GET /api/ai-ml/predictions/{symbol}` - Get price predictions
- `GET /api/ai-ml/sentiment/{symbol}` - Get sentiment analysis
- `POST /api/ai-ml/train` - Train ML models

### System Endpoints

#### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /metrics

Prometheus metrics endpoint.

**Response:**

```
# HELP api_gateway_requests_total Total API requests
# TYPE api_gateway_requests_total counter
api_gateway_requests_total{method="GET",endpoint="/health",status="200"} 42
```

## Configuration

### Environment Variables

| Variable               | Description                    | Default                                |
| ---------------------- | ------------------------------ | -------------------------------------- |
| `JWT_SECRET`           | JWT signing secret             | `your-secret-key-change-in-production` |
| `JWT_EXPIRATION_HOURS` | JWT token expiration           | `24`                                   |
| `RATE_LIMIT_REQUESTS`  | Rate limit requests per window | `100`                                  |
| `RATE_LIMIT_WINDOW`    | Rate limit window in seconds   | `60`                                   |
| `CORS_ORIGINS`         | Allowed CORS origins           | `*`                                    |
| `TRUSTED_HOSTS`        | Trusted host names             | `*`                                    |

### Service URLs

| Service         | Environment Variable  | Default                       |
| --------------- | --------------------- | ----------------------------- |
| Trading Engine  | `TRADING_ENGINE_URL`  | `http://trading-engine:8001`  |
| Market Data     | `MARKET_DATA_URL`     | `http://market-data:8002`     |
| Risk Management | `RISK_MANAGEMENT_URL` | `http://risk-management:8003` |
| Analytics       | `ANALYTICS_URL`       | `http://analytics:8004`       |
| AI/ML           | `AI_ML_URL`           | `http://ai-ml:8005`           |

## Usage

### Development

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Dependencies**

   ```bash
   docker-compose up -d postgres redis influxdb
   ```

4. **Run API Gateway**
   ```bash
   cd services/api-gateway
   python main.py
   ```

### Docker

1. **Build Image**

   ```bash
   docker build -f services/api-gateway/Dockerfile -t api-gateway .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 --env-file .env api-gateway
   ```

### Docker Compose

```bash
docker-compose up -d api-gateway
```

## Testing

### Manual Testing

1. **Health Check**

   ```bash
   curl http://localhost:8000/health
   ```

2. **Login**

   ```bash
   curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "password"}'
   ```

3. **Authenticated Request**
   ```bash
   curl http://localhost:8000/auth/me \
     -H "Authorization: Bearer <token>"
   ```

### Automated Testing

```bash
python test_api_gateway.py
```

## Monitoring

### Metrics

The API Gateway exposes the following Prometheus metrics:

- `api_gateway_requests_total` - Total number of requests
- `api_gateway_request_duration_seconds` - Request duration histogram
- `api_gateway_rate_limit_exceeded_total` - Rate limit violations

### Health Checks

- **Endpoint:** `GET /health`
- **Docker:** Built-in health check every 30 seconds
- **Kubernetes:** Readiness and liveness probes

### Logging

Structured JSON logging with the following fields:

- `timestamp` - ISO 8601 timestamp
- `level` - Log level (INFO, ERROR, etc.)
- `logger` - Logger name
- `event` - Event description
- `request_id` - Unique request identifier
- Additional context fields

## Security Considerations

### Production Deployment

1. **Change Default Secrets**
   - Generate strong JWT secret
   - Use secure database passwords
   - Rotate API keys regularly

2. **Enable HTTPS**
   - Use TLS certificates
   - Redirect HTTP to HTTPS
   - Enable HSTS headers

3. **Network Security**
   - Use private networks
   - Configure firewalls
   - Implement VPN access

4. **Monitoring**
   - Set up alerting
   - Monitor for suspicious activity
   - Regular security audits

### Rate Limiting

- Default: 100 requests per 60 seconds per IP
- Configurable per endpoint
- Redis-based for distributed systems
- Automatic cleanup of expired entries

### Authentication

- JWT tokens with configurable expiration
- Secure password hashing with bcrypt
- Role-based access control
- Token blacklisting support (TODO)

## Troubleshooting

### Common Issues

1. **Connection Refused**

   ```
   Error: Connection refused to service
   ```

   - Check if target service is running
   - Verify service URLs in configuration
   - Check network connectivity

2. **Authentication Failed**

   ```
   Error: 401 Unauthorized
   ```

   - Verify JWT token is valid
   - Check token expiration
   - Ensure correct Authorization header format

3. **Rate Limit Exceeded**

   ```
   Error: 429 Too Many Requests
   ```

   - Wait for rate limit window to reset
   - Reduce request frequency
   - Check rate limit configuration

4. **Service Unavailable**
   ```
   Error: 503 Service Unavailable
   ```

   - Check target service health
   - Verify service discovery
   - Check load balancer configuration

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Health Checks

Check service health:

```bash
curl http://localhost:8000/health
```

Check individual service health through gateway:

```bash
curl http://localhost:8000/api/trading/health \
  -H "Authorization: Bearer <token>"
```

## Development

### Adding New Routes

1. **Define Route Handler**

   ```python
   @app.api_route("/api/new-service/{path:path}", methods=["GET", "POST"])
   async def route_new_service(request: Request, path: str, current_user: UserInfo = Depends(get_current_user)):
       return await route_to_service("new-service", f"/api/{path}", request, current_user)
   ```

2. **Add Service Configuration**

   ```python
   self.services = {
       # ... existing services
       "new-service": os.getenv("NEW_SERVICE_URL", "http://new-service:8006")
   }
   ```

3. **Update Documentation**
   - Add endpoint documentation
   - Update service list
   - Add configuration variables

### Custom Middleware

```python
@app.middleware("http")
async def custom_middleware(request: Request, call_next):
    # Pre-processing
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Post-processing
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response
```

## API Reference

For complete API documentation, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json
