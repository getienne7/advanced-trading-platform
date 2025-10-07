import React from 'react';
import { TrendingUp, TrendingDown, Minus, MessageSquare, Twitter, Globe } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const SentimentAnalysis = ({ data, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-gray-900">Sentiment Analysis</h3>
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
          <h3 className="text-lg font-semibold text-gray-900">Sentiment Analysis</h3>
        </div>
        <div className="text-center py-8 text-gray-500">
          No sentiment data available
        </div>
      </div>
    );
  }

  const getSentimentIcon = (sentiment) => {
    if (sentiment > 0.1) return <TrendingUp className="w-5 h-5 text-success-600" />;
    if (sentiment < -0.1) return <TrendingDown className="w-5 h-5 text-danger-600" />;
    return <Minus className="w-5 h-5 text-gray-600" />;
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment > 0.1) return 'text-success-600';
    if (sentiment < -0.1) return 'text-danger-600';
    return 'text-gray-600';
  };

  const getSourceIcon = (source) => {
    switch (source.toLowerCase()) {
      case 'twitter':
        return <Twitter className="w-4 h-4" />;
      case 'reddit':
        return <MessageSquare className="w-4 h-4" />;
      case 'news':
      default:
        return <Globe className="w-4 h-4" />;
    }
  };

  // Prepare data for charts
  const pieData = Object.entries(data.sources || {}).map(([source, sourceData]) => ({
    name: source.charAt(0).toUpperCase() + source.slice(1),
    value: Math.abs(sourceData.sentiment || 0),
    sentiment: sourceData.sentiment || 0,
    confidence: sourceData.confidence || 0,
  }));

  const barData = Object.entries(data.sources || {}).map(([source, sourceData]) => ({
    source: source.charAt(0).toUpperCase() + source.slice(1),
    sentiment: sourceData.sentiment || 0,
    confidence: sourceData.confidence || 0,
  }));

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-semibold text-gray-900">Sentiment Analysis</h3>
        <div className="flex items-center space-x-2">
          {getSentimentIcon(data.overall_sentiment)}
          <span className={`font-medium ${getSentimentColor(data.overall_sentiment)}`}>
            {data.sentiment_label?.charAt(0).toUpperCase() + data.sentiment_label?.slice(1)}
          </span>
        </div>
      </div>

      {/* Overall Sentiment */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Overall Sentiment</span>
          <span className={`text-lg font-bold ${getSentimentColor(data.overall_sentiment)}`}>
            {data.overall_sentiment?.toFixed(3)}
          </span>
        </div>
        
        {/* Sentiment Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${
              data.overall_sentiment > 0 ? 'bg-success-500' : 
              data.overall_sentiment < 0 ? 'bg-danger-500' : 'bg-gray-400'
            }`}
            style={{
              width: `${Math.min(Math.abs(data.overall_sentiment) * 100, 100)}%`,
              marginLeft: data.overall_sentiment < 0 ? `${100 - Math.min(Math.abs(data.overall_sentiment) * 100, 100)}%` : '0'
            }}
          ></div>
        </div>
        
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Bearish (-1.0)</span>
          <span>Neutral (0.0)</span>
          <span>Bullish (+1.0)</span>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Confidence</span>
          <span className="text-sm font-bold text-gray-900">
            {(data.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
          <div
            className="bg-primary-500 h-1 rounded-full transition-all duration-500"
            style={{ width: `${data.confidence * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Source Breakdown */}
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-gray-900">Source Breakdown</h4>
        
        {Object.entries(data.sources || {}).map(([source, sourceData]) => (
          <div key={source} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              {getSourceIcon(source)}
              <div>
                <div className="font-medium text-gray-900 capitalize">{source}</div>
                <div className="text-xs text-gray-500">
                  {sourceData.count || 0} items analyzed
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className={`font-medium ${getSentimentColor(sourceData.sentiment)}`}>
                {sourceData.sentiment?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-xs text-gray-500">
                {((sourceData.confidence || 0) * 100).toFixed(0)}% confidence
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Sentiment Distribution Chart */}
      {barData.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Sentiment by Source</h4>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" tick={{ fontSize: 12 }} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'sentiment' ? value.toFixed(3) : `${(value * 100).toFixed(0)}%`,
                    name === 'sentiment' ? 'Sentiment' : 'Confidence'
                  ]}
                />
                <Bar 
                  dataKey="sentiment" 
                  fill="#3b82f6"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          Last updated: {new Date(data.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalysis;