import React from 'react';
import { CheckCircle, XCircle, AlertCircle, Activity, Brain, Target, TrendingUp } from 'lucide-react';

const SystemStatus = ({ health }) => {
  if (!health) {
    return (
      <div className="bg-gray-100 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <div className="animate-pulse w-4 h-4 bg-gray-300 rounded-full"></div>
          <span className="text-gray-600">Checking system status...</span>
        </div>
      </div>
    );
  }

  const isHealthy = health.status === 'healthy';
  const modelsLoaded = health.models_loaded || {};

  const getStatusIcon = (status) => {
    if (status === 'healthy' || status === true) {
      return <CheckCircle className="w-4 h-4 text-success-600" />;
    } else if (status === 'degraded') {
      return <AlertCircle className="w-4 h-4 text-warning-600" />;
    } else {
      return <XCircle className="w-4 h-4 text-danger-600" />;
    }
  };

  const getStatusColor = (status) => {
    if (status === 'healthy' || status === true) {
      return 'text-success-600 bg-success-50 border-success-200';
    } else if (status === 'degraded') {
      return 'text-warning-600 bg-warning-50 border-warning-200';
    } else {
      return 'text-danger-600 bg-danger-50 border-danger-200';
    }
  };

  const getModelIcon = (modelType) => {
    switch (modelType) {
      case 'sentiment':
        return <Brain className="w-4 h-4" />;
      case 'prediction':
        return <Target className="w-4 h-4" />;
      case 'regime':
        return <Activity className="w-4 h-4" />;
      default:
        return <TrendingUp className="w-4 h-4" />;
    }
  };

  return (
    <div className={`rounded-lg border p-4 ${getStatusColor(health.status)}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon(health.status)}
          <div>
            <h3 className="font-semibold">
              AI/ML System Status: {health.status?.charAt(0).toUpperCase() + health.status?.slice(1)}
            </h3>
            <p className="text-sm opacity-80">
              Last checked: {new Date(health.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>

        {/* Model Status Indicators */}
        <div className="flex items-center space-x-4">
          {Object.entries(modelsLoaded).map(([model, status]) => (
            <div key={model} className="flex items-center space-x-1">
              {getModelIcon(model)}
              {getStatusIcon(status)}
              <span className="text-xs font-medium capitalize">{model}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Additional Status Information */}
      {!isHealthy && (
        <div className="mt-3 pt-3 border-t border-current border-opacity-20">
          <div className="text-sm">
            <strong>Issues detected:</strong> Some AI/ML models may not be functioning properly. 
            Please check the service logs or contact support if issues persist.
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;