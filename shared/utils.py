"""
Shared utility functions across all microservices.
"""
import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Union
import aiohttp
import structlog
from cryptography.fernet import Fernet
from passlib.context import CryptContext


logger = structlog.get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def generate_api_signature(secret: str, method: str, endpoint: str, 
                          params: Dict = None, body: str = "") -> str:
    """Generate HMAC signature for API authentication"""
    timestamp = str(int(time.time() * 1000))
    nonce = str(int(time.time() * 1000000))
    
    if params:
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        if query_string:
            endpoint += f"?{query_string}"
    
    message = f"{method}{endpoint}{timestamp}{nonce}{body}"
    signature = hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature, timestamp, nonce


def encrypt_sensitive_data(data: str, key: bytes) -> str:
    """Encrypt sensitive data using Fernet"""
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data.decode()


def decrypt_sensitive_data(encrypted_data: str, key: bytes) -> str:
    """Decrypt sensitive data using Fernet"""
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data.encode())
    return decrypted_data.decode()


def normalize_decimal(value: Union[str, float, Decimal], precision: int = 8) -> Decimal:
    """Normalize decimal values with proper precision"""
    if isinstance(value, str):
        decimal_value = Decimal(value)
    elif isinstance(value, float):
        decimal_value = Decimal(str(value))
    else:
        decimal_value = value
    
    # Round down to avoid precision issues
    quantize_value = Decimal('0.1') ** precision
    return decimal_value.quantize(quantize_value, rounding=ROUND_DOWN)


def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return float((new_value - old_value) / old_value * 100)


def calculate_compound_return(returns: List[float]) -> float:
    """Calculate compound return from a list of returns"""
    if not returns:
        return 0.0
    
    compound = 1.0
    for ret in returns:
        compound *= (1 + ret / 100)
    
    return (compound - 1) * 100


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside risk
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252))


def calculate_max_drawdown(values: List[Decimal]) -> Dict[str, Any]:
    """Calculate maximum drawdown"""
    if not values or len(values) < 2:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}
    
    peak = values[0]
    max_drawdown = Decimal('0')
    max_drawdown_pct = 0.0
    
    for value in values:
        if value > peak:
            peak = value
        
        drawdown = peak - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = float(drawdown / peak * 100) if peak > 0 else 0.0
    
    return {
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": max_drawdown_pct
    }


def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk (VaR)"""
    if not returns:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    return float(np.percentile(returns_array, (1 - confidence_level) * 100))


def calculate_expected_shortfall(returns: List[float], confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional VaR)"""
    if not returns:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    var = np.percentile(returns_array, (1 - confidence_level) * 100)
    
    # Average of returns below VaR
    tail_returns = returns_array[returns_array <= var]
    
    if len(tail_returns) == 0:
        return var
    
    return float(np.mean(tail_returns))


def calculate_correlation_matrix(price_series: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate correlation matrix for multiple price series"""
    import numpy as np
    
    symbols = list(price_series.keys())
    correlation_matrix = {}
    
    for symbol1 in symbols:
        correlation_matrix[symbol1] = {}
        for symbol2 in symbols:
            if symbol1 == symbol2:
                correlation_matrix[symbol1][symbol2] = 1.0
            else:
                series1 = np.array(price_series[symbol1])
                series2 = np.array(price_series[symbol2])
                
                if len(series1) != len(series2) or len(series1) < 2:
                    correlation_matrix[symbol1][symbol2] = 0.0
                else:
                    correlation = np.corrcoef(series1, series2)[0, 1]
                    correlation_matrix[symbol1][symbol2] = float(correlation) if not np.isnan(correlation) else 0.0
    
    return correlation_matrix


def format_currency(amount: Decimal, currency: str = "USD", precision: int = 2) -> str:
    """Format currency amount for display"""
    if currency == "USD":
        return f"${amount:.{precision}f}"
    elif currency == "BTC":
        return f"{amount:.8f} BTC"
    elif currency == "ETH":
        return f"{amount:.6f} ETH"
    else:
        return f"{amount:.{precision}f} {currency}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage for display"""
    return f"{value:.{precision}f}%"


def get_utc_timestamp() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Convert timestamp to datetime"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to timestamp"""
    return int(dt.timestamp())


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> bool:
        """Acquire rate limit token"""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    async def wait_for_token(self):
        """Wait until a token is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class RetryHandler:
    """Retry handler with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(
                    "Function call failed, retrying",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)
        
        raise last_exception


async def make_http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """Make HTTP request with proper error handling"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        try:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data
            ) as response:
                response.raise_for_status()
                return await response.json()
        
        except aiohttp.ClientError as e:
            logger.error("HTTP request failed", url=url, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error in HTTP request", url=url, error=str(e))
            raise


def validate_trading_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or len(symbol) < 3:
        return False
    
    # Basic validation - should contain only uppercase letters and numbers
    return symbol.replace('/', '').replace('-', '').isalnum()


def calculate_position_size(
    account_balance: Decimal,
    risk_percentage: float,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    leverage: int = 1
) -> Decimal:
    """Calculate optimal position size based on risk management"""
    if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0:
        return Decimal('0')
    
    risk_amount = account_balance * Decimal(str(risk_percentage))
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff == 0:
        return Decimal('0')
    
    position_size = risk_amount / price_diff / Decimal(str(leverage))
    return normalize_decimal(position_size, 8)


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly Criterion for optimal position sizing"""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    win_loss_ratio = avg_win / abs(avg_loss)
    kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Cap at 25% for safety
    return max(0.0, min(kelly_percentage, 0.25))


class PerformanceTimer:
    """Context manager for measuring execution time"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        logger.info(
            "Operation completed",
            operation=self.operation_name,
            execution_time_ms=round(execution_time * 1000, 2)
        )


def setup_logging(service_name: str) -> structlog.BoundLogger:
    """Setup structured logging for a service."""
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
    
    return structlog.get_logger(service_name)