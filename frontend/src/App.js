import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp, Activity, Target, RefreshCw, AlertCircle } from 'lucide-react';
import SentimentAnalysis from './components/SentimentAnalysis';
import PricePrediction from './components/PricePrediction';
import MarketRegime from './components/MarketRegime';
import TradingSignals from './components/TradingSignals';
import SystemStatus from './components/SystemStatus';
import { fetchComprehensiveAnalysis, fetchSystemHealth } from './services/api';

function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [analysisData, setAnalysisData] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT'];

  const fetchData = async (symbol = selectedSymbol) => {
    setLoading(true);
    setError(null);
    
    try {
      const [analysis, health] = await Promise.all([
        fetchComprehensiveAnalysis(symbol),
        fetchSystemHealth()
      ]);
      
      setAnalysisData(analysis);
      setSystemHealth(health);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchData();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
  };

  const handleRefresh = () => {
    fetchData();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Advanced Trading Platform</h1>
                <p className="text-sm text-gray-500">AI-Powered Market Intelligence</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Symbol Selector */}
              <select
                value={selectedSymbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                className="rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
              
              {/* Refresh Button */}
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="btn-primary flex items-center space-x-2"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
              
              {/* Last Update */}
              {lastUpdate && (
                <div className="text-sm text-gray-500">
                  Last updated: {lastUpdate.toLocaleTimeString()}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Status */}
        <div className="mb-8">
          <SystemStatus health={systemHealth} />
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 bg-danger-50 border border-danger-200 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-danger-600" />
              <span className="text-danger-700 font-medium">Error loading data</span>
            </div>
            <p className="text-danger-600 mt-1">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && !analysisData && (
          <div className="flex items-center justify-center py-12">
            <div className="flex items-center space-x-3">
              <RefreshCw className="w-6 h-6 animate-spin text-primary-600" />
              <span className="text-lg text-gray-600">Loading AI analysis...</span>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {analysisData && (
          <div className="space-y-8">
            {/* Key Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="metric-card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Current Price</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${analysisData.price_prediction?.current_price?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-primary-600" />
                </div>
              </div>
              
              <div className="metric-card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">24h Prediction</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${analysisData.price_prediction?.ensemble_prediction?.toLocaleString() || 'N/A'}
                    </p>
                    {analysisData.price_prediction && (
                      <p className={`text-sm ${
                        analysisData.price_prediction.ensemble_prediction > analysisData.price_prediction.current_price
                          ? 'text-success-600' : 'text-danger-600'
                      }`}>
                        {((analysisData.price_prediction.ensemble_prediction - analysisData.price_prediction.current_price) / analysisData.price_prediction.current_price * 100).toFixed(2)}%
                      </p>
                    )}
                  </div>
                  <Target className="w-8 h-8 text-success-600" />
                </div>
              </div>
              
              <div className="metric-card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Market Regime</p>
                    <p className="text-lg font-bold text-gray-900 capitalize">
                      {analysisData.market_regime?.current_regime || 'N/A'}
                    </p>
                    {analysisData.market_regime && (
                      <span className={`status-indicator status-${analysisData.market_regime.current_regime}`}>
                        {(analysisData.market_regime.regime_probability * 100).toFixed(0)}% confidence
                      </span>
                    )}
                  </div>
                  <Activity className="w-8 h-8 text-warning-600" />
                </div>
              </div>
              
              <div className="metric-card">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Sentiment</p>
                    <p className="text-lg font-bold text-gray-900 capitalize">
                      {analysisData.sentiment?.sentiment_label || 'N/A'}
                    </p>
                    {analysisData.sentiment && (
                      <p className="text-sm text-gray-600">
                        Score: {analysisData.sentiment.overall_sentiment.toFixed(3)}
                      </p>
                    )}
                  </div>
                  <Brain className="w-8 h-8 text-primary-600" />
                </div>
              </div>
            </div>

            {/* Main Analysis Sections */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Sentiment Analysis */}
              <SentimentAnalysis 
                data={analysisData.sentiment} 
                loading={loading}
              />
              
              {/* Price Prediction */}
              <PricePrediction 
                data={analysisData.price_prediction} 
                loading={loading}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Market Regime */}
              <MarketRegime 
                data={analysisData.market_regime} 
                loading={loading}
              />
              
              {/* Trading Signals */}
              <TradingSignals 
                data={analysisData.trading_signal} 
                loading={loading}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;