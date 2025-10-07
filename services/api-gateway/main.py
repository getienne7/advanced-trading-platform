"""
API Gateway Service for Advanced Trading Platform.
Handles request routing, authentication, rate limiting, and validation.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import httpx
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import sys
from pathlib import Path

# Add shared directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from database import get_redis
from utils import setup_logging

# Configure logging
logger = setup_logging("api-gateway")

# Metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration')
RATE_LIMIT_EXCEEDED = Counter('api_gateway_rate_limit_exceeded_total', 'Rate limit exceeded count', ['client_ip'])

# Configuration
class GatewayConfig:
    """API Gateway configuration."""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        self.trusted_hosts = os.getenv("TRUSTED_HOSTS", "*").split(",")
        
        # Service endpoints
        self.services = {
            "trading-engine": os.getenv("TRADING_ENGINE_URL", "http://trading-engine:8001"),
            "market-data": os.getenv("MARKET_DATA_URL", "http://market-data:8002"),
            "risk-management": os.getenv("RISK_MANAGEMENT_URL", "http://risk-management:8003"),
            "analytics": os.getenv("ANALYTICS_URL", "http://analytics:8004"),
            "ai-ml": os.getenv("AI_ML_URL", "http://ai-ml:8005")
        }

config = GatewayConfig()

# Pydantic models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]

# Authentication
security = HTTPBearer()

class AuthenticationService:
    """JWT-based authentication service."""
    
    def __init__(self):
        self.config = config
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours)
        to_encode = user_data.copy()
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.config.jwt_secret, 
            algorithm=self.config.jwt_algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

auth_service = AuthenticationService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Get current authenticated user."""
    try:
        payload = auth_service.verify_token(credentials.credentials)
        user_info = UserInfo(
            user_id=payload.get("user_id"),
            username=payload.get("username"),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )
        return user_info
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Rate limiting
class RateLimiter:
    """Redis-based rate limiter."""
    
    def __init__(self):
        self.config = config
    
    async def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed based on rate limit."""
        try:
            redis = await get_redis()
            key = f"rate_limit:{client_ip}"
            
            # Get current count
            current = await redis.get(key)
            if current is None:
                # First request in window
                await redis.setex(key, self.config.rate_limit_window, 1)
                return True
            
            current_count = int(current)
            if current_count >= self.config.rate_limit_requests:
                RATE_LIMIT_EXCEEDED.labels(client_ip=client_ip).inc()
                return False
            
            # Increment counter
            await redis.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return True

rate_limiter = RateLimiter()

# Request routing
class ServiceRouter:
    """Service request router."""
    
    def __init__(self):
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def route_request(self, 
                          service_name: str, 
                          path: str, 
                          method: str,
                          headers: Dict[str, str],
                          params: Optional[Dict[str, Any]] = None,
                          json_data: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Route request to appropriate service."""
        
        if service_name not in self.config.services:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")
        
        service_url = self.config.services[service_name]
        full_url = f"{service_url}{path}"
        
        try:
            response = await self.client.request(
                method=method,
                url=full_url,
                headers=headers,
                params=params,
                json=json_data
            )
            return response
            
        except httpx.RequestError as e:
            logger.error(f"Service request failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service '{service_name}' unavailable")
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

router = ServiceRouter()

# FastAPI app
app = FastAPI(
    title="Advanced Trading Platform API Gateway",
    description="Centralized API gateway for the trading platform",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.trusted_hosts
)

# Middleware for metrics and rate limiting
@app.middleware("http")
async def metrics_and_rate_limit_middleware(request: Request, call_next):
    """Middleware for metrics collection and rate limiting."""
    start_time = datetime.utcnow()
    client_ip = request.client.host
    
    # Rate limiting
    if not await rate_limiter.is_allowed(client_ip):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status="429"
        ).inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Process request
    response = await call_next(request)
    
    # Collect metrics
    duration = (datetime.utcnow() - start_time).total_seconds()
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code)
    ).inc()
    
    return response

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """User login endpoint."""
    # TODO: Implement actual user authentication against database
    # For now, using dummy authentication
    
    if login_request.username == "admin" and login_request.password == "password":
        user_data = {
            "user_id": "1",
            "username": login_request.username,
            "roles": ["admin"],
            "permissions": ["read", "write", "admin"]
        }
        
        access_token = auth_service.create_access_token(user_data)
        
        return LoginResponse(
            access_token=access_token,
            expires_in=config.jwt_expiration_hours * 3600
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/me", response_model=UserInfo)
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information."""
    return current_user

@app.post("/auth/logout")
async def logout(current_user: UserInfo = Depends(get_current_user)):
    """User logout endpoint."""
    # TODO: Implement token blacklisting
    return {"message": "Logged out successfully"}

# Service routing endpoints
@app.api_route("/api/trading/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_trading_service(
    request: Request,
    path: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Route requests to trading engine service."""
    return await route_to_service("trading-engine", f"/api/{path}", request, current_user)

@app.api_route("/api/market-data/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_market_data_service(
    request: Request,
    path: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Route requests to market data service."""
    return await route_to_service("market-data", f"/api/{path}", request, current_user)

@app.api_route("/api/analytics/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_analytics_service(
    request: Request,
    path: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Route requests to analytics service."""
    return await route_to_service("analytics", f"/api/{path}", request, current_user)

@app.api_route("/api/ai-ml/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_ai_ml_service(
    request: Request,
    path: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Route requests to AI/ML service."""
    return await route_to_service("ai-ml", f"/api/{path}", request, current_user)

async def route_to_service(service_name: str, path: str, request: Request, current_user: UserInfo):
    """Generic service routing function."""
    try:
        # Prepare headers
        headers = dict(request.headers)
        headers["X-User-ID"] = current_user.user_id
        headers["X-Username"] = current_user.username
        headers["X-User-Roles"] = ",".join(current_user.roles)
        
        # Get request data
        json_data = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                json_data = await request.json()
            except:
                pass
        
        # Route request
        response = await router.route_request(
            service_name=service_name,
            path=path,
            method=request.method,
            headers=headers,
            params=dict(request.query_params),
            json_data=json_data
        )
        
        # Return response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service routing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("API Gateway starting up...")
    
    # Initialize database connections
    from database import initialize_databases
    await initialize_databases()
    
    logger.info("API Gateway started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("API Gateway shutting down...")
    
    # Close HTTP client
    await router.close()
    
    # Close database connections
    from database import cleanup_databases
    await cleanup_databases()
    
    logger.info("API Gateway shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )