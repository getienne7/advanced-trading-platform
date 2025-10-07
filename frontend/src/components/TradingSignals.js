import React from 'react';
import { TrendingUp, TrendingDown, Minus, Shield, Target, AlertTriangle } from 'lucide-react';

const TradingSignals = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-gray-900">Trading Signals</h3>
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
          <h3 className="text-lg font-semibold text-gray-900">Trading Signals</h3>
        </div>
        <div className="text-center py-8 text-gray-500">
          No trading signal data available
        </div>
      </div>
    );
  }

  const getActionIcon = (action) => {
    switch (action?.toLowerCase()) {
      case 'buy':
        return <TrendingUp className="w-5 h-5 text-success-600" />;
      case 'sell':
        return <TrendingDown className="w-5 h-5 text-danger-600" />;
      case 'hold':
      default:
        return <Minus className="w-5 h-5 text-gray-600" />;
    }
  };

  const getActionColor = (action) => {
    switch (action?.toLowerCase()) {
      case 'buy':
        return 'text-success-600 bg-success-50 border-success-200';
      case 'sell':
        return 'text-danger-600 bg-danger-50 border-danger-200';
      case 'hold':
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStrengthColor = (strength) => {
    switch (strength?.toLowerCase()) {
      case 'strong':
        return 'text-primary-600 bg-primary-50';
      case 'moderate':
        return 'text-warning-600 bg-warning-50';
      case 'weak':
        return 'text-gray-600 bg-gray-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-success-600';
    if (confidence >= 0.6) return 'text-warning-600';
    return 'text-danger-600';
  };

  const getSignalStrengthBar = (score) => {
    const normalizedScore = (score + 1) / 2; // Convert from [-1, 1] to [0, 1]
    const percentage = normalizedScore * 100;
    
    let color = 'bg-gray-400';
    if (score > 0.3) color = 'bg-success-500';
    else if (score < -0.3) color = 'bg-danger-500';
    
    return { percentage, color };
  };

  const signalStrength = getSignalStrengthBar(data.signal_score || 0);

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-gray-900">Trading Signals</h3>
        <div className="flex items-center space-x-2">
          {getActionIcon(data.action)}
          <span className={`font-medium ${getActionColor(data.action).split(' ')[0]}`}>
            {data.action?.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Main Signal */}
      <div className={`p-4 rounded-lg border mb-6 ${getActionColor(data.action)}`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            {getActionIcon(data.action)}
            <div>
              <div className="font-bold text-xl">{data.action?.toUpperCase()} Signal</div>
              <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStrengthColor(data.strength)}`}>
                {data.strength?.toUpperCase()} Strength
              </div>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm font-medium opacity-80">Signal Score</div>
            <div className="text-2xl font-bold">
              {data.signal_score?.toFixed(3) || 'N/A'}
            </div>
          </div>
        </div>

        {/* Signal Strength Bar */}
        <div className="mb-3">
          <div className="flex justify-between text-xs opacity-80 mb-1">
            <span>Strong Sell (-1.0)</span>
            <span>Neutral (0.0)</span>
            <span>Strong Buy (+1.0)</span>
          </div>
          <div className="w-full bg-white bg-opacity-50 rounded-full h-3">
            <div
              className={`h-3 rounded-full ${signalStrength.color} transition-all duration-500`}
              style={{ 
                width: `${Math.abs(data.signal_score || 0) * 50}%`,
                marginLeft: (data.signal_score || 0) < 0 ? `${50 - Math.abs(data.signal_score || 0) * 50}%` : '50%'
              }}
            ></div>
          </div>
        </div>

        {/* Confidence */}
        <div className="flex items-center justify-between">
          <span className="text-sm opacity-80">Confidence</span>
          <span className={`font-bold ${getConfidenceColor(data.confidence || 0)}`}>
            {((data.confidence || 0) * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Signal Components */}
      {data.components && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Signal Components</h4>
          <div className="space-y-3">
            {Object.entries(data.components).map(([component, value]) => (
              <div key={component} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {component === 'sentiment' && <Brain className="w-4 h-4 text-primary-600" />}
                  {component === 'prediction' && <Target className="w-4 h-4 text-success-600" />}
                  {component === 'regime' && <Activity className="w-4 h-4 text-warning-600" />}
                  <div>
                    <div className="font-medium text-gray-900 capitalize">{component}</div>
                    <div className="text-xs text-gray-500">
                      {component === 'sentiment' ? 'Market sentiment analysis' :
                       component === 'prediction' ? 'Price prediction models' :
                       component === 'regime' ? 'Market regime detection' : 'Analysis component'}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`font-medium ${value >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                    {value >= 0 ? '+' : ''}{value?.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {Math.abs(value || 0) > 0.5 ? 'Strong' : Math.abs(value || 0) > 0.2 ? 'Moderate' : 'Weak'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk Assessment */}
      <div className="mb-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
        <div className="flex items-center space-x-2 mb-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600" />
          <h4 className="text-sm font-semibold text-yellow-900">Risk Assessment</h4>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-yellow-700">Signal Confidence</div>
            <div className={`font-medium ${getConfidenceColor(data.confidence || 0)}`}>
              {data.confidence >= 0.8 ? 'High' : data.confidence >= 0.6 ? 'Medium' : 'Low'}
            </div>
          </div>
          
          <div>
            <div className="text-yellow-700">Recommendation</div>
            <div className="font-medium text-yellow-900">
              {data.confidence >= 0.7 ? 'Execute with caution' : 'Wait for stronger signal'}
            </div>
          </div>
        </div>
        
        <div className="mt-3 text-xs text-yellow-700">
          <strong>Note:</strong> Always consider your risk tolerance and portfolio allocation before executing trades.
        </div>
      </div>

      {/* Action Items */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Recommended Actions</h4>
        <div className="space-y-2 text-sm">
          {data.action === 'BUY' && (
            <>
              <div className="flex items-center space-x-2 text-success-700">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span>Consider opening a long position</span>
              </div>
              <div className="flex items-center space-x-2 text-success-700">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span>Set stop-loss below recent support levels</span>
              </div>
              <div className="flex items-center space-x-2 text-success-700">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span>Monitor for trend continuation signals</span>
              </div>
            </>
          )}
          
          {data.action === 'SELL' && (
            <>
              <div className="flex items-center space-x-2 text-danger-700">
                <div className="w-2 h-2 bg-danger-500 rounded-full"></div>
                <span>Consider reducing long positions</span>
              </div>
              <div className="flex items-center space-x-2 text-danger-700">
                <div className="w-2 h-2 bg-danger-500 rounded-full"></div>
                <span>Evaluate short position opportunities</span>
              </div>
              <div className="flex items-center space-x-2 text-danger-700">
                <div className="w-2 h-2 bg-danger-500 rounded-full"></div>
                <span>Tighten stop-losses on existing positions</span>
              </div>
            </>
          )}
          
          {data.action === 'HOLD' && (
            <>
              <div className="flex items-center space-x-2 text-gray-700">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span>Maintain current positions</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-700">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span>Wait for clearer directional signals</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-700">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span>Monitor market conditions closely</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Last Updated */}
      <div className="pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          Signal generated: {new Date(data.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default TradingSignals;