import React from 'react';
import { TrendingUp, Target, Brain, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

const PricePrediction = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-gray-900">Price Predictions</h3>
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
          <h3 className="text-lg font-semibold text-gray-900">Price Predictions</h3>
        </div>
        <div className="text-center py-8 text-gray-500">
          No prediction data available
        </div>
      </div>
    );
  }

  const formatCurrency = (value) => {
    if (typeof value !== 'number') return 'N/A';
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercentage = (current, predicted) => {
    if (typeof current !== 'number' || typeof predicted !== 'number') return 'N/A';
    const change = ((predicted - current) / current) * 100;
    return `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
  };

  const getChangeColor = (current, predicted) => {
    if (typeof current !== 'number' || typeof predicted !== 'number') return 'text-gray-600';
    return predicted >= current ? 'text-success-600' : 'text-danger-600';
  };

  const getModelIcon = (model) => {
    switch (model.toLowerCase()) {
      case 'lstm':
        return <Brain className="w-4 h-4" />;
      case 'transformer':
        return <Target className="w-4 h-4" />;
      case 'linear':
        return <TrendingUp className="w-4 h-4" />;
      default:
        return <BarChart3 className="w-4 h-4" />;
    }
  };

  // Prepare data for charts
  const timeHorizons = ['1h', '6h', '12h', '24h'];
  const chartData = timeHorizons.map(horizon => {
    const dataPoint = {
      time: horizon,
      current: data.current_price,
    };

    // Add predictions from each model
    Object.entries(data.predictions || {}).forEach(([model, predictions]) => {
      if (predictions[horizon]) {
        dataPoint[model] = predictions[horizon];
      }
    });

    // Add ensemble prediction
    if (horizon === '24h') {
      dataPoint.ensemble = data.ensemble_prediction;
    }

    return dataPoint;
  });

  // Model comparison data
  const modelData = Object.entries(data.predictions || {}).map(([model, predictions]) => ({
    model: model.toUpperCase(),
    prediction: predictions['24h'] || 0,
    change: predictions['24h'] ? ((predictions['24h'] - data.current_price) / data.current_price) * 100 : 0,
  }));

  // Add ensemble to model data
  if (data.ensemble_prediction) {
    modelData.push({
      model: 'ENSEMBLE',
      prediction: data.ensemble_prediction,
      change: ((data.ensemble_prediction - data.current_price) / data.current_price) * 100,
    });
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-gray-900">Price Predictions</h3>
        <div className="flex items-center space-x-2">
          <Target className="w-5 h-5 text-primary-600" />
          <span className="text-sm font-medium text-gray-600">24h Forecast</span>
        </div>
      </div>

      {/* Current Price & Ensemble Prediction */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-sm font-medium text-gray-600 mb-1">Current Price</div>
          <div className="text-2xl font-bold text-gray-900">
            {formatCurrency(data.current_price)}
          </div>
        </div>
        
        <div className="text-center p-4 bg-primary-50 rounded-lg">
          <div className="text-sm font-medium text-primary-600 mb-1">24h Ensemble</div>
          <div className="text-2xl font-bold text-primary-900">
            {formatCurrency(data.ensemble_prediction)}
          </div>
          <div className={`text-sm font-medium ${getChangeColor(data.current_price, data.ensemble_prediction)}`}>
            {formatPercentage(data.current_price, data.ensemble_prediction)}
          </div>
        </div>
      </div>

      {/* Model Predictions */}
      <div className="space-y-3 mb-6">
        <h4 className="text-sm font-semibold text-gray-900">Model Predictions (24h)</h4>
        
        {Object.entries(data.predictions || {}).map(([model, predictions]) => (
          <div key={model} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              {getModelIcon(model)}
              <div>
                <div className="font-medium text-gray-900">{model.toUpperCase()}</div>
                <div className="text-xs text-gray-500">
                  {model === 'lstm' ? 'Neural Network' : 
                   model === 'transformer' ? 'Attention Model' : 
                   'Linear Regression'}
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className="font-medium text-gray-900">
                {formatCurrency(predictions['24h'])}
              </div>
              <div className={`text-xs font-medium ${getChangeColor(data.current_price, predictions['24h'])}`}>
                {formatPercentage(data.current_price, predictions['24h'])}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Prediction Timeline Chart */}
      {chartData.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Prediction Timeline</h4>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  domain={['dataMin - 100', 'dataMax + 100']}
                  tickFormatter={(value) => `$${value.toLocaleString()}`}
                />
                <Tooltip 
                  formatter={(value, name) => [formatCurrency(value), name.toUpperCase()]}
                  labelFormatter={(label) => `Time: ${label}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="current" 
                  stroke="#6b7280" 
                  strokeDasharray="5 5"
                  name="current"
                />
                <Line 
                  type="monotone" 
                  dataKey="lstm" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="lstm"
                />
                <Line 
                  type="monotone" 
                  dataKey="transformer" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="transformer"
                />
                <Line 
                  type="monotone" 
                  dataKey="linear" 
                  stroke="#f59e0b" 
                  strokeWidth={2}
                  name="linear"
                />
                <Line 
                  type="monotone" 
                  dataKey="ensemble" 
                  stroke="#ef4444" 
                  strokeWidth={3}
                  name="ensemble"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Model Comparison Chart */}
      {modelData.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Model Comparison (24h Change %)</h4>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(value) => `${value.toFixed(1)}%`} />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'change' ? `${value.toFixed(2)}%` : formatCurrency(value),
                    name === 'change' ? 'Change' : 'Price'
                  ]}
                />
                <Bar 
                  dataKey="change" 
                  fill="#3b82f6"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Confidence Intervals */}
      {data.confidence_intervals && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Confidence Intervals (24h)</h4>
          <div className="space-y-2">
            {Object.entries(data.confidence_intervals).map(([model, intervals]) => {
              const interval24h = intervals['24h'];
              if (!interval24h || !Array.isArray(interval24h) || interval24h.length !== 2) return null;
              
              return (
                <div key={model} className="flex items-center justify-between text-sm">
                  <span className="font-medium text-gray-700">{model.toUpperCase()}</span>
                  <span className="text-gray-600">
                    {formatCurrency(interval24h[0])} - {formatCurrency(interval24h[1])}
                  </span>
                </div>
              );
            })}
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

export default PricePrediction;