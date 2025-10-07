# Risk Management Service

Advanced risk calculation and monitoring service for the trading platform, providing comprehensive Value at Risk (VaR) calculations, portfolio risk analysis, and real-time risk monitoring.

## Features

### 🎯 **Dynamic Risk Calculation Engine**

- **Monte Carlo Simulation**: Advanced simulation-based VaR calculation with fat-tail distributions
- **Historical Simulation**: Non-parametric VaR using actual historical returns
- **Parametric VaR**: Fast analytical VaR calculation with normal and adjusted distributions
- **Hybrid Approach**: Intelligent combination of all methods based on data characteristics

### 📊 **Comprehensive Risk Metrics**

- Value at Risk (VaR) at 95% and 99% confidence levels
- Expected Shortfall (Conditional VaR)
- Volatility analysis with annualized metrics
- Skewness and kurtosis for distribution analysis
- Maximum drawdown calculation
- Sharpe, Sortino, and Calmar ratios
- Beta and correlation analysis

### 🏦 **Portfolio Risk Management**

- Multi-asset portfolio VaR calculation
- Component VaR and marginal VaR analysis
- Risk budget allocation and attribution
- Diversification ratio calculation
- Concentration risk monitoring
- Correlation matrix analysis
- Systematic vs. idiosyncratic risk decomposition

### 🧪 **Stress Testing**

- Predefined stress scenarios (market crash, crypto winter, etc.)
- Custom scenario analysis
- Historical stress testing
- Monte Carlo stress testing
- Tail risk and black swan protection

### 💧 **Liquidity Risk Assessment**

- Participation rate analysis
- Time to liquidate calculations
- Liquidity cost estimation
- Market impact assessment
- Liquidity risk scoring

## API Endpoints

### Risk Calculation

- `POST /risk/var/calculate` - Calculate VaR for individual positions
- `POST /risk/portfolio/calculate` - Calculate comprehensive portfolio risk
- `POST /risk/stress-test` - Run stress tests on portfolio
- `POST /risk/liquidity/assess` - Assess liquidity risk

### Risk Monitoring

- `POST /risk/limits/check` - Check portfolio against risk limits
- `GET /risk/scenarios/predefined` - Get predefined stress scenarios

### Configuration

- `GET /risk/config` - Get current risk configuration
- `POST /risk/config/update` - Update risk engine configuration

### Health & Monitoring

- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics

## Usage Examples

### Calculate VaR for a Position

```python
import requests

# Calculate VaR for a BTC position
var_request = {
    "returns": [0.02, -0.01, 0.015, -0.008, 0.025],  # Historical returns
    "position_value": 100000,  # $100k position
    "asset_class": "cryptocurrency",
    "model": "hybrid",
    "symbol": "BTC"
}

response = requests.post("http://localhost:8008/risk/var/calculate", json=var_request)
result = response.json()

print(f"VaR 95%: ${result['risk_metrics']['var_95']:,.2f}")
print(f"VaR 99%: ${result['risk_metrics']['var_99']:,.2f}")
```

### Calculate Portfolio Risk

```python
# Calculate risk for a multi-asset portfolio
portfolio_request = {
    "positions": {
        "BTC": 50000,
        "ETH": 30000,
        "ADA": 20000
    },
    "returns_data": {
        "BTC": [0.02, -0.01, 0.015, -0.008, 0.025],
        "ETH": [0.018, -0.012, 0.020, -0.005, 0.022],
        "ADA": [0.025, -0.015, 0.018, -0.010, 0.030]
    },
    "portfolio_id": "portfolio_001"
}

response = requests.post("http://localhost:8008/risk/portfolio/calculate", json=portfolio_request)
result = response.json()

print(f"Portfolio VaR 95%: ${result['portfolio_risk']['total_var_95']:,.2f}")
print(f"Diversification Ratio: {result['portfolio_risk']['diversification_ratio']:.2f}")
```

### Run Stress Tests

```python
# Run stress tests on portfolio
stress_request = {
    "positions": {"BTC": 50000, "ETH": 30000},
    "scenarios": [
        {"BTC": -0.3, "ETH": -0.35},  # Market crash
        {"BTC": -0.5, "ETH": -0.55}   # Crypto winter
    ],
    "scenario_names": ["Market Crash", "Crypto Winter"]
}

response = requests.post("http://localhost:8008/risk/stress-test", json=stress_request)
result = response.json()

for scenario, data in result['stress_test_results'].items():
    print(f"{scenario}: ${data['pnl']:,.2f} ({data['pnl_percentage']:.1f}%)")
```

## Configuration

### Risk Configuration Parameters

```python
{
    "confidence_levels": [0.95, 0.99],
    "time_horizon_days": 1,
    "lookback_days": 252,
    "monte_carlo_simulations": 10000,
    "rebalancing_frequency": "daily",
    "risk_free_rate": 0.02,
    "market_benchmark": "BTC",
    "correlation_threshold": 0.7,
    "concentration_limit": 0.25
}
```

### Asset Classes Supported

- `cryptocurrency` - Bitcoin, Ethereum, altcoins
- `forex` - Currency pairs
- `equity` - Stocks and indices
- `commodity` - Gold, oil, etc.
- `bond` - Fixed income securities

### Risk Models Available

- `monte_carlo` - Simulation-based approach
- `historical_simulation` - Non-parametric historical approach
- `parametric` - Analytical normal distribution approach
- `hybrid` - Intelligent combination of all methods

## Predefined Stress Scenarios

The service includes several predefined stress scenarios:

1. **Market Crash (-30%)** - Severe market downturn
2. **Crypto Winter (-50%)** - Extended bear market
3. **Flash Crash (-20%)** - Sudden market crash
4. **Altcoin Collapse** - Major altcoins lose value
5. **Regulatory Shock** - Negative regulatory impact

## Risk Metrics Explained

### Value at Risk (VaR)

- **VaR 95%**: Maximum expected loss over the time horizon with 95% confidence
- **VaR 99%**: Maximum expected loss over the time horizon with 99% confidence

### Expected Shortfall (ES)

- Average loss beyond the VaR threshold
- Also known as Conditional VaR (CVaR)
- Provides information about tail risk

### Risk-Adjusted Returns

- **Sharpe Ratio**: Excess return per unit of total risk
- **Sortino Ratio**: Excess return per unit of downside risk
- **Calmar Ratio**: Annual return divided by maximum drawdown

### Portfolio Risk Metrics

- **Component VaR**: Each asset's contribution to portfolio VaR
- **Marginal VaR**: Change in portfolio VaR from a small position change
- **Diversification Ratio**: Benefit from diversification
- **Concentration Risk**: Herfindahl index of position concentration

## Installation & Setup

### Prerequisites

- Python 3.10+
- NumPy, SciPy, Pandas
- FastAPI, Uvicorn
- PostgreSQL, Redis

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Docker Deployment

```bash
# Build image
docker build -t risk-management-service .

# Run container
docker run -p 8008:8008 risk-management-service
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_risk_management.py -v

# Run specific test categories
pytest test_risk_management.py::TestDynamicRiskEngine -v
pytest test_risk_management.py::TestRiskManagementAPI -v

# Run simple integration test
python test_risk_management.py
```

## Performance Considerations

### Optimization Tips

- Use appropriate sample sizes for Monte Carlo simulations
- Cache correlation matrices for frequently accessed portfolios
- Implement parallel processing for large portfolios
- Use historical simulation for real-time calculations

### Scalability

- Service supports concurrent risk calculations
- Stateless design allows horizontal scaling
- Redis caching for improved performance
- Prometheus metrics for monitoring

## Monitoring & Alerting

### Key Metrics to Monitor

- `risk_calculations_total` - Total risk calculations performed
- `var_calculation_seconds` - VaR calculation latency
- `portfolio_var_95` - Current portfolio VaR levels
- `risk_limit_breaches_total` - Risk limit violations

### Health Checks

- Service health endpoint: `/health`
- Database connectivity checks
- Risk engine initialization status

## Integration with Trading Platform

The Risk Management Service integrates with:

- **Portfolio Management**: Real-time position monitoring
- **Order Management**: Pre-trade risk checks
- **Alert System**: Risk limit breach notifications
- **Reporting**: Daily risk reports and analytics
- **Compliance**: Regulatory risk reporting

## Mathematical Models

### Monte Carlo VaR

Uses simulation to generate potential future scenarios based on:

- Historical return distribution fitting
- Skewed t-distribution for fat tails
- Bootstrap sampling for scenario generation

### Historical Simulation

- Uses actual historical returns
- No distributional assumptions
- Captures actual market behavior
- Limited by historical data availability

### Parametric VaR

- Assumes normal distribution of returns
- Fast analytical calculation
- Adjustments for asset class characteristics
- Good for real-time applications

## Risk Model Validation

### Backtesting

- Compare predicted VaR with actual losses
- Calculate violation ratios
- Perform statistical tests (Kupiec, Christoffersen)
- Monitor model performance over time

### Model Validation Metrics

- Coverage ratio (should be close to confidence level)
- Independence of violations
- Magnitude of violations (Expected Shortfall)
- Stability across different market conditions

## Compliance & Regulatory

### Regulatory Standards

- Basel III market risk framework
- FRTB (Fundamental Review of Trading Book)
- CFTC risk management requirements
- MiFID II risk reporting

### Audit Trail

- Complete calculation history
- Model parameters and assumptions
- Data sources and quality checks
- Risk limit monitoring and breaches

## Future Enhancements

### Planned Features

- Machine learning-based risk models
- Real-time streaming risk calculations
- Advanced correlation modeling (DCC-GARCH)
- Climate risk integration
- ESG risk factors

### Model Improvements

- Regime-switching models
- Copula-based dependency modeling
- High-frequency risk calculations
- Cross-asset volatility spillovers

## Support & Documentation

For technical support or questions:

- Check the API documentation at `/docs`
- Review test cases for usage examples
- Monitor service logs for debugging
- Use health check endpoints for status

## License

This Risk Management Service is part of the Advanced Trading Platform and follows the same licensing terms.
