import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8005';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Unable to connect to AI/ML service. Please ensure the service is running on port 8005.');
    }
    
    if (error.response?.status === 500) {
      throw new Error('Internal server error. Please try again later.');
    }
    
    if (error.response?.status === 404) {
      throw new Error('API endpoint not found. Please check the service configuration.');
    }
    
    throw new Error(error.response?.data?.detail || error.message || 'An unexpected error occurred');
  }
);

// API Functions
export const fetchSystemHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return {
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      models_loaded: {
        sentiment: false,
        prediction: false,
        regime: false
      }
    };
  }
};

export const fetchSentimentAnalysis = async (symbol, sources = ['news', 'twitter', 'reddit'], timeframe = '24h') => {
  const response = await api.post('/api/sentiment/analyze', {
    symbol,
    sources,
    timeframe
  });
  return response.data;
};

export const fetchPricePrediction = async (symbol, timeframe = '1h', horizon = 24, models = ['lstm', 'transformer', 'linear']) => {
  const response = await api.post('/api/predictions/price', {
    symbol,
    timeframe,
    horizon,
    models
  });
  return response.data;
};

export const fetchMarketRegime = async (symbol) => {
  const response = await api.get(`/api/regime/${symbol}`);
  return response.data;
};

export const fetchComprehensiveAnalysis = async (symbol) => {
  const response = await api.get(`/api/analysis/${symbol}`);
  return response.data;
};

export const fetchMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

// Utility functions
export const formatCurrency = (value, currency = 'USD') => {
  if (typeof value !== 'number') return 'N/A';
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

export const formatPercentage = (value, decimals = 2) => {
  if (typeof value !== 'number') return 'N/A';
  
  return `${(value * 100).toFixed(decimals)}%`;
};

export const formatTimestamp = (timestamp) => {
  if (!timestamp) return 'N/A';
  
  const date = new Date(timestamp);
  return date.toLocaleString();
};

export const getSentimentColor = (sentiment) => {
  if (sentiment > 0.2) return 'text-success-600';
  if (sentiment < -0.2) return 'text-danger-600';
  return 'text-gray-600';
};

export const getRegimeColor = (regime) => {
  switch (regime?.toLowerCase()) {
    case 'bull':
      return 'text-success-600';
    case 'bear':
      return 'text-danger-600';
    case 'volatile':
      return 'text-warning-600';
    case 'sideways':
    default:
      return 'text-gray-600';
  }
};

export default api;