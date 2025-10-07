import React from 'react';
import { Activity, TrendingUp, TrendingDown, Minus, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const MarketRegime = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-gray-900">Market Regime</h3>
          <div className="animate-pulse w-4 h-4 bg-gray-300 rounded"></div>
        </div>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-gray-900">Market Regime</h3>
        </div>
        <div className="text-center py-8 text-gray-500">
          No regime data available
        </div>
      </div>
    );
  }

  const getRegimeIcon = (regime) => {
    switch (regime?.toLowerCase()) {
      case 'bull':
        return <TrendingUp className="w-5 h-5 text-success-600" />;
      case 'bear':
        return <TrendingDown className="w-5 h-5 text-danger-600" />;
      case 'volatile':
        return <Zap className="w-5 h-5 text-warning-600" />;
      case 'sideways':
      default:
        return <Minus className="w-5 h-5 text-gray-600" />;
    }
  };

  const getRegimeColor = (regime) => {
    switch (regime?.toLowerCase()) {
      case 'bull':
        return 'text-success-600 bg-success-50 border-success-200';
      case 'bear':
        return 'text-danger-600 bg-danger-50 border-danger-200';
      case 'volatile':
        return 'text-warning-600 bg-warning-50 border-warning-200';
      case 'sideways':
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getVolatilityColor = (volatility) => {
    if (volatility > 0.4) return 'text-danger-600';
    if (volatility > 0.25) return 'text-warning-600';
    return 'text-success-600';
  };

  const getVolatilityLabel = (volatility) => {
    if (volatility > 0.4) return 'High';
    if (volatility > 0.25) return 'Medium';
    return 'Low';
  };

  // Prepare transition matrix data for visualization
  const transitionData = data.regime_transition_matrix ? 
    Object.entries(data.regime_transition_matrix).map(([from, transitions]) => ({
      from,
      ...transitions
    })) : [];

  // Regime history data for chart
  const historyData = data.regime_history?.slice(-10).map((entry, index) => ({
    period: `P${index + 1}`,
    regime: entry.regime,
    duration: entry.duration_days || 1,
    probability: entry.avg_probability || 0,
  })) || [];

  const REGIME_COLORS = {
    bull: '#10b981',
    bear: '#ef4444',
    sideways: '#6b7280',
    volatile: '#f59e0b'
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-gray-900">Market Regime Analysis</h3>
        <div className="flex items-center space-x-2">
          {getRegimeIcon(data.current_regime)}
          <span className={`font-medium capitalize ${getRegimeColor(data.current_regime).split(' ')[0]}`}>
            {data.current_regime}
          </span>
        </div>
      </div>

      {/* Current Regime Status */}
      <div className={`p-4 rounded-lg border mb-6 ${getRegimeColor(data.current_regime)}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            {getRegimeIcon(data.current_regime)}
            <span className="font-semibold text-lg capitalize">{data.current_regime} Market</span>
          </div>
          <span className="text-sm font-medium">
            {(data.regime_probability * 100).toFixed(1)}% confidence
          </span>
        </div>
        
        {/* Confidence Bar */}
        <div className="w-full bg-white bg-opacity-50 rounded-full h-2 mb-2">
          <div
            className="h-2 rounded-full bg-current opacity-70 transition-all duration-500"
            style={{ width: `${data.regime_probability * 100}%` }}
          ></div>
        </div>
        
        <div className="text-sm opacity-80">
          Confidence Score: {data.confidence_score?.toFixed(3) || 'N/A'}
        </div>
      </div>

      {/* Volatility Forecast */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Volatility Forecast</span>
          <div className="text-right">
            <span className={`text-lg font-bold ${getVolatilityColor(data.volatility_forecast)}`}>
              {(data.volatility_forecast * 100).toFixed(1)}%
            </span>
            <div className={`text-xs font-medium ${getVolatilityColor(data.volatility_forecast)}`}>
              {getVolatilityLabel(data.volatility_forecast)} Risk
            </div>
          </div>
        </div>
        
        {/* Volatility Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${
              data.volatility_forecast > 0.4 ? 'bg-danger-500' :
              data.volatility_forecast > 0.25 ? 'bg-warning-500' : 'bg-success-500'
            }`}
            style={{ width: `${Math.min(data.volatility_forecast * 200, 100)}%` }}
          ></div>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Low (0%)</span>
          <span>Medium (25%)</span>
          <span>High (50%+)</span>
        </div>
      </div>

      {/* Strategy Recommendation */}
      {data.strategy_recommendation && (
        <div className="mb-6 p-4 bg-primary-50 rounded-lg border border-primary-200">
          <h4 className="text-sm font-semibold text-primary-900 mb-3">Recommended Strategy</h4>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-gray-600">Strategy</div>
              <div className="font-medium text-primary-900 capitalize">
                {data.strategy_recommendation.primary?.replace('_', ' ')}
              </div>
            </div>
            
            <div>
              <div className="text-gray-600">Risk Level</div>
              <div className="font-medium text-primary-900 capitalize">
                {data.strategy_recommendation.risk_level}
              </div>
            </div>
            
            <div>
              <div className="text-gray-600">Position Size</div>
              <div className="font-medium text-primary-900">
                {(data.strategy_recommendation.position_size * 100).toFixed(0)}%
              </div>
            </div>
            
            <div>
              <div className="text-gray-600">Stop Loss</div>
              <div className="font-medium text-primary-900">
                {(data.strategy_recommendation.stop_loss * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Regime History */}
      {historyData.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Recent Regime History</h4>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'duration' ? `${value} days` : `${(value * 100).toFixed(0)}%`,
                    name === 'duration' ? 'Duration' : 'Probability'
                  ]}
                  labelFormatter={(label) => `Period: ${label}`}
                />
                <Bar 
                  dataKey="duration" 
                  fill="#3b82f6"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Transition Probabilities */}
      {data.regime_transition_matrix && Object.keys(data.regime_transition_matrix).length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Regime Transition Probabilities</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(data.regime_transition_matrix).map(([from, transitions]) => (
              <div key={from} className="space-y-1">
                <div className="font-medium text-gray-700 capitalize">From {from}:</div>
                {Object.entries(transitions).map(([to, prob]) => (
                  <div key={to} className="flex justify-between pl-2">
                    <span className="text-gray-600 capitalize">→ {to}</span>
                    <span className="font-medium">{(prob * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          Last updated: {new Date(data.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default MarketRegime;