# Advanced Trading Platform Dashboard

A modern, real-time dashboard for the Advanced Trading Platform's AI/ML market intelligence system.

## 🚀 Features

### AI-Powered Market Intelligence

- **Sentiment Analysis** - Multi-source sentiment analysis from news, Twitter, and Reddit
- **Price Predictions** - LSTM and Transformer-based price forecasting with ensemble models
- **Market Regime Detection** - Bull/Bear/Sideways/Volatile market classification
- **Trading Signals** - AI-generated buy/sell/hold recommendations with confidence scores

### Real-Time Dashboard

- **Live Data Updates** - Auto-refreshing every 30 seconds
- **Interactive Charts** - Recharts-powered visualizations
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Modern UI** - Clean, professional interface with Tailwind CSS

### System Monitoring

- **Health Checks** - Real-time system status monitoring
- **Model Status** - AI/ML model availability indicators
- **Performance Metrics** - API response times and success rates

## 🛠️ Technology Stack

- **Frontend**: React 18, Tailwind CSS, Recharts
- **Backend**: FastAPI, TensorFlow, scikit-learn
- **AI/ML**: LSTM, Transformer, GARCH, Gaussian Mixture Models
- **Data**: Real-time market data with synthetic fallbacks

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

```bash
# Start both backend and frontend automatically
python start_dashboard.py
```

### Option 2: Manual Startup

#### Start Backend (Terminal 1)

```bash
cd services/ai-ml
python main.py
```

#### Start Frontend (Terminal 2)

```bash
cd frontend
npm install
npm start
```

## 📊 Dashboard Sections

### 1. System Status

- Overall system health
- AI/ML model status
- Last update timestamp

### 2. Key Metrics

- Current price
- 24h prediction
- Market regime
- Sentiment score

### 3. Sentiment Analysis

- Overall sentiment score
- Source breakdown (News, Twitter, Reddit)
- Confidence levels
- Historical sentiment trends

### 4. Price Predictions

- Multi-model predictions (LSTM, Transformer, Linear)
- Ensemble forecasting
- Confidence intervals
- Prediction timeline charts

### 5. Market Regime

- Current regime classification
- Volatility forecasting
- Strategy recommendations
- Regime transition probabilities

### 6. Trading Signals

- Buy/Sell/Hold recommendations
- Signal strength and confidence
- Component analysis
- Risk assessment

## 🔧 Configuration

### Environment Variables

```bash
REACT_APP_API_URL=http://localhost:8005  # Backend API URL
```

### API Endpoints

- `GET /health` - System health check
- `GET /api/analysis/{symbol}` - Comprehensive analysis
- `POST /api/sentiment/analyze` - Sentiment analysis
- `POST /api/predictions/price` - Price predictions
- `GET /api/regime/{symbol}` - Market regime detection

## 📱 Supported Symbols

- BTC/USDT - Bitcoin
- ETH/USDT - Ethereum
- ADA/USDT - Cardano
- DOT/USDT - Polkadot
- LINK/USDT - Chainlink

## 🎨 UI Components

### Cards

- Sentiment Analysis Card
- Price Prediction Card
- Market Regime Card
- Trading Signals Card
- System Status Card

### Charts

- Line charts for price predictions
- Bar charts for sentiment analysis
- Pie charts for regime distribution
- Progress bars for confidence scores

## 🔄 Data Flow

1. **Frontend** requests data from backend API
2. **Backend** processes requests using AI/ML models
3. **Models** analyze market data and generate insights
4. **Results** are returned to frontend for visualization
5. **Dashboard** updates in real-time with new data

## 🚨 Error Handling

- Graceful degradation when API is unavailable
- Loading states for all components
- Error messages with actionable information
- Fallback data when models are training

## 📈 Performance

- Optimized React components with proper memoization
- Efficient API calls with caching
- Responsive design for all screen sizes
- Fast chart rendering with Recharts

## 🔐 Security

- CORS configuration for API access
- Input validation on all API endpoints
- Rate limiting on backend services
- Secure data transmission

## 🧪 Testing

```bash
# Run frontend tests
npm test

# Run backend tests
cd ../services/ai-ml
python -m pytest
```

## 📦 Deployment

### Development

```bash
npm start  # Runs on http://localhost:3000
```

### Production Build

```bash
npm run build  # Creates optimized production build
```

### Docker Deployment

```bash
docker-compose up --build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Advanced Trading Platform and is proprietary software.

## 🆘 Support

For support and questions:

- Check the API documentation at `http://localhost:8005/docs`
- Review the backend logs for error details
- Ensure all dependencies are properly installed

---

**Built with ❤️ for professional traders and quantitative analysts**
