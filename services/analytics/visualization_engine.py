"""
Visualization Engine - Advanced interactive charts and dashboard widgets
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Engine for creating interactive charts and dashboard widgets using Plotly
    """
    
    def __init__(self):
        self.default_theme = {
            'background_color': '#1e1e1e',
            'grid_color': '#333333',
            'text_color': '#ffffff',
            'positive_color': '#00ff88',
            'negative_color': '#ff4444',
            'neutral_color': '#888888',
            'accent_color': '#00aaff'
        }
        
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True
        }
    
    async def create_pnl_chart(
        self,
        pnl_data: Dict[str, Any],
        timeframe: str = '1D'
    ) -> Dict[str, Any]:
        """Create P&L attribution chart"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('P&L Over Time', 'P&L by Strategy', 'P&L by Asset', 'Cumulative P&L'),
                specs=[[{"secondary_y": True}, {"type": "pie"}],
                       [{"type": "bar"}, {"secondary_y": True}]]
            )
            
            # Mock time series data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            cumulative_pnl = np.cumsum(np.random.normal(50, 200, len(dates)))
            daily_pnl = np.diff(np.concatenate([[0], cumulative_pnl]))
            
            # P&L Over Time (top left)
            fig.add_trace(
                go.Scatter(
                    x=dates[1:],
                    y=daily_pnl,
                    mode='lines+markers',
                    name='Daily P&L',
                    line=dict(color=self.default_theme['accent_color']),
                    hovertemplate='Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # P&L by Strategy (top right)
            strategy_data = pnl_data.get('strategy_attribution', {})
            if strategy_data:
                strategies = list(strategy_data.keys())
                strategy_pnl = [strategy_data[s].get('total_pnl', 0) for s in strategies]
                colors = [self.default_theme['positive_color'] if pnl >= 0 else self.default_theme['negative_color'] 
                         for pnl in strategy_pnl]
                
                fig.add_trace(
                    go.Pie(
                        labels=strategies,
                        values=[abs(pnl) for pnl in strategy_pnl],
                        marker=dict(colors=colors),
                        hovertemplate='Strategy: %{label}<br>P&L: $%{value:,.2f}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # P&L by Asset (bottom left)
            asset_data = pnl_data.get('asset_attribution', {})
            if asset_data:
                assets = list(asset_data.keys())
                asset_pnl = [asset_data[a].get('total_pnl', 0) for a in assets]
                colors = [self.default_theme['positive_color'] if pnl >= 0 else self.default_theme['negative_color'] 
                         for pnl in asset_pnl]
                
                fig.add_trace(
                    go.Bar(
                        x=assets,
                        y=asset_pnl,
                        marker=dict(color=colors),
                        name='Asset P&L',
                        hovertemplate='Asset: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Cumulative P&L (bottom right)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color=self.default_theme['positive_color'], width=3),
                    fill='tonexty',
                    hovertemplate='Date: %{x}<br>Cumulative P&L: $%{y:,.2f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='P&L Attribution Dashboard',
                template='plotly_dark',
                paper_bgcolor=self.default_theme['background_color'],
                plot_bgcolor=self.default_theme['background_color'],
                font=dict(color=self.default_theme['text_color']),
                height=800,
                showlegend=True
            )
            
            return {
                'chart_data': fig.to_dict(),
                'chart_config': self.chart_config,
                'chart_type': 'pnl_attribution'
            }
            
        except Exception as e:
            logger.error(f"Error creating P&L chart: {e}")
            return self._create_error_chart("Error creating P&L chart")
    
    async def create_performance_chart(
        self,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive performance metrics chart"""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Returns Distribution', 'Risk Metrics', 'Drawdown Analysis',
                    'Win/Loss Ratio', 'Monthly Returns', 'Performance Metrics'
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "bar"}, {"secondary_y": True}],
                    [{"type": "pie"}, {"type": "heatmap"}, {"type": "indicator"}]
                ]
            )
            
            # Mock performance data
            daily_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            # Returns Distribution (top left)
            fig.add_trace(
                go.Histogram(
                    x=daily_returns * 100,
                    nbinsx=30,
                    name='Daily Returns',
                    marker=dict(color=self.default_theme['accent_color']),
                    hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Risk Metrics (top middle)
            risk_metrics = ['Sharpe', 'Sortino', 'Calmar', 'Max DD']
            risk_values = [1.2, 1.5, 0.8, -0.15]
            colors = [self.default_theme['positive_color'] if v >= 0 else self.default_theme['negative_color'] 
                     for v in risk_values]
            
            fig.add_trace(
                go.Bar(
                    x=risk_metrics,
                    y=risk_values,
                    marker=dict(color=colors),
                    name='Risk Metrics',
                    hovertemplate='Metric: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Drawdown Analysis (top right)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(drawdown))),
                    y=drawdown,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color=self.default_theme['negative_color']),
                    fill='tonexty',
                    hovertemplate='Day: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=3
            )
            
            # Win/Loss Ratio (bottom left)
            wins = len(daily_returns[daily_returns > 0])
            losses = len(daily_returns[daily_returns < 0])
            
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    marker=dict(colors=[self.default_theme['positive_color'], self.default_theme['negative_color']]),
                    hovertemplate='%{label}: %{value} days<br>%{percent}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Monthly Returns Heatmap (bottom middle)
            monthly_returns = self._generate_monthly_returns_data()
            
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns['returns'],
                    x=monthly_returns['months'],
                    y=monthly_returns['years'],
                    colorscale='RdYlGn',
                    hovertemplate='Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Performance Indicator (bottom right)
            total_return = (cumulative_returns[-1]) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=total_return,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Total Return %"},
                    delta={'reference': 10},
                    gauge={
                        'axis': {'range': [-50, 100]},
                        'bar': {'color': self.default_theme['accent_color']},
                        'steps': [
                            {'range': [-50, 0], 'color': self.default_theme['negative_color']},
                            {'range': [0, 50], 'color': self.default_theme['neutral_color']},
                            {'range': [50, 100], 'color': self.default_theme['positive_color']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                title='Performance Analytics Dashboard',
                template='plotly_dark',
                paper_bgcolor=self.default_theme['background_color'],
                plot_bgcolor=self.default_theme['background_color'],
                font=dict(color=self.default_theme['text_color']),
                height=800,
                showlegend=True
            )
            
            return {
                'chart_data': fig.to_dict(),
                'chart_config': self.chart_config,
                'chart_type': 'performance_analytics'
            }
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return self._create_error_chart("Error creating performance chart")
    
    async def create_risk_dashboard(
        self,
        risk_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create risk metrics dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'VaR Analysis', 'Correlation Matrix', 'Position Concentration',
                    'Risk Score Gauge', 'Stress Test Results', 'Leverage Analysis'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "heatmap"}, {"type": "pie"}],
                    [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # VaR Analysis (top left)
            var_methods = ['Parametric', 'Historical', 'Monte Carlo']
            var_values = [2500, 2800, 2650]
            
            fig.add_trace(
                go.Bar(
                    x=var_methods,
                    y=var_values,
                    marker=dict(color=self.default_theme['negative_color']),
                    name='VaR 95%',
                    hovertemplate='Method: %{x}<br>VaR: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Correlation Matrix (top middle)
            assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
            correlation_matrix = np.random.uniform(0.3, 0.9, (len(assets), len(assets)))
            np.fill_diagonal(correlation_matrix, 1.0)
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=assets,
                    y=assets,
                    colorscale='RdYlBu_r',
                    zmin=-1,
                    zmax=1,
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Position Concentration (top right)
            concentration_data = risk_data.get('concentration_metrics', {})
            position_weights = concentration_data.get('position_weights', {
                'BTC/USDT': 0.4,
                'ETH/USDT': 0.3,
                'Others': 0.3
            })
            
            fig.add_trace(
                go.Pie(
                    labels=list(position_weights.keys()),
                    values=list(position_weights.values()),
                    hovertemplate='Asset: %{label}<br>Weight: %{percent}<extra></extra>'
                ),
                row=1, col=3
            )
            
            # Risk Score Gauge (bottom left)
            risk_score = risk_data.get('risk_score', 6.5)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': self._get_risk_color(risk_score)},
                        'steps': [
                            {'range': [0, 3], 'color': self.default_theme['positive_color']},
                            {'range': [3, 7], 'color': self.default_theme['neutral_color']},
                            {'range': [7, 10], 'color': self.default_theme['negative_color']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 8
                        }
                    }
                ),
                row=2, col=1
            )
            
            # Stress Test Results (bottom middle)
            stress_scenarios = ['Market Crash 10%', 'Market Crash 20%', 'Vol Spike 2x', 'Corr Spike 90%']
            stress_pnl = [-5000, -12000, -8000, -6000]
            
            fig.add_trace(
                go.Bar(
                    x=stress_scenarios,
                    y=stress_pnl,
                    marker=dict(color=self.default_theme['negative_color']),
                    name='Stress P&L',
                    hovertemplate='Scenario: %{x}<br>P&L Impact: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Leverage Analysis (bottom right)
            leverage_data = risk_data.get('leverage_metrics', {})
            leveraged_positions = leverage_data.get('leveraged_positions', [])
            
            if leveraged_positions:
                symbols = [pos['symbol'] for pos in leveraged_positions]
                leverages = [pos['leverage'] for pos in leveraged_positions]
                notional_values = [pos['notional_value'] for pos in leveraged_positions]
                
                fig.add_trace(
                    go.Scatter(
                        x=leverages,
                        y=notional_values,
                        mode='markers+text',
                        text=symbols,
                        textposition='top center',
                        marker=dict(
                            size=[lev * 5 for lev in leverages],
                            color=leverages,
                            colorscale='Reds',
                            showscale=True
                        ),
                        name='Leveraged Positions',
                        hovertemplate='Symbol: %{text}<br>Leverage: %{x}x<br>Notional: $%{y:,.0f}<extra></extra>'
                    ),
                    row=2, col=3
                )
            
            # Update layout
            fig.update_layout(
                title='Risk Management Dashboard',
                template='plotly_dark',
                paper_bgcolor=self.default_theme['background_color'],
                plot_bgcolor=self.default_theme['background_color'],
                font=dict(color=self.default_theme['text_color']),
                height=800,
                showlegend=True
            )
            
            return {
                'chart_data': fig.to_dict(),
                'chart_config': self.chart_config,
                'chart_type': 'risk_dashboard'
            }
            
        except Exception as e:
            logger.error(f"Error creating risk dashboard: {e}")
            return self._create_error_chart("Error creating risk dashboard")
    
    async def create_portfolio_heatmap(
        self,
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create portfolio performance heatmap"""
        try:
            # Generate mock portfolio performance data
            assets = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT', 'UNI/USDT']
            timeframes = ['1H', '4H', '1D', '1W', '1M']
            
            # Create performance matrix
            performance_matrix = []
            for asset in assets:
                asset_performance = []
                for timeframe in timeframes:
                    # Mock performance data
                    performance = np.random.normal(0, 5)  # Random performance between -15% and +15%
                    asset_performance.append(performance)
                performance_matrix.append(asset_performance)
            
            fig = go.Figure(data=go.Heatmap(
                z=performance_matrix,
                x=timeframes,
                y=assets,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='Asset: %{y}<br>Timeframe: %{x}<br>Performance: %{z:.2f}%<extra></extra>',
                colorbar=dict(title="Performance %")
            ))
            
            fig.update_layout(
                title='Portfolio Performance Heatmap',
                template='plotly_dark',
                paper_bgcolor=self.default_theme['background_color'],
                plot_bgcolor=self.default_theme['background_color'],
                font=dict(color=self.default_theme['text_color']),
                height=500,
                xaxis_title='Timeframe',
                yaxis_title='Assets'
            )
            
            return {
                'chart_data': fig.to_dict(),
                'chart_config': self.chart_config,
                'chart_type': 'portfolio_heatmap'
            }
            
        except Exception as e:
            logger.error(f"Error creating portfolio heatmap: {e}")
            return self._create_error_chart("Error creating portfolio heatmap")
    
    async def create_custom_widget(
        self,
        widget_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create custom dashboard widget based on configuration"""
        try:
            widget_type = widget_config.get('type', 'line')
            data = widget_config.get('data', {})
            title = widget_config.get('title', 'Custom Widget')
            
            if widget_type == 'line':
                fig = go.Figure()
                for series in data.get('series', []):
                    fig.add_trace(go.Scatter(
                        x=series.get('x', []),
                        y=series.get('y', []),
                        mode='lines+markers',
                        name=series.get('name', 'Series'),
                        line=dict(color=series.get('color', self.default_theme['accent_color']))
                    ))
            
            elif widget_type == 'bar':
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    marker=dict(color=data.get('colors', self.default_theme['accent_color']))
                ))
            
            elif widget_type == 'pie':
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=data.get('labels', []),
                    values=data.get('values', []),
                    marker=dict(colors=data.get('colors', []))
                ))
            
            elif widget_type == 'gauge':
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=data.get('value', 0),
                    title={'text': title},
                    gauge={
                        'axis': {'range': data.get('range', [0, 100])},
                        'bar': {'color': data.get('color', self.default_theme['accent_color'])}
                    }
                ))
            
            else:
                return self._create_error_chart(f"Unknown widget type: {widget_type}")
            
            fig.update_layout(
                title=title,
                template='plotly_dark',
                paper_bgcolor=self.default_theme['background_color'],
                plot_bgcolor=self.default_theme['background_color'],
                font=dict(color=self.default_theme['text_color']),
                height=widget_config.get('height', 400)
            )
            
            return {
                'chart_data': fig.to_dict(),
                'chart_config': self.chart_config,
                'chart_type': 'custom_widget'
            }
            
        except Exception as e:
            logger.error(f"Error creating custom widget: {e}")
            return self._create_error_chart("Error creating custom widget")
    
    def _generate_monthly_returns_data(self) -> Dict[str, Any]:
        """Generate mock monthly returns data for heatmap"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = ['2022', '2023', '2024']
        
        returns = []
        for year in years:
            year_returns = []
            for month in months:
                # Mock monthly return
                monthly_return = np.random.normal(1, 5)  # 1% average with 5% std
                year_returns.append(monthly_return)
            returns.append(year_returns)
        
        return {
            'months': months,
            'years': years,
            'returns': returns
        }
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score <= 3:
            return self.default_theme['positive_color']
        elif risk_score <= 7:
            return self.default_theme['neutral_color']
        else:
            return self.default_theme['negative_color']
    
    def _create_error_chart(self, error_message: str) -> Dict[str, Any]:
        """Create error chart when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=self.default_theme['negative_color'])
        )
        
        fig.update_layout(
            title='Visualization Error',
            template='plotly_dark',
            paper_bgcolor=self.default_theme['background_color'],
            plot_bgcolor=self.default_theme['background_color'],
            font=dict(color=self.default_theme['text_color']),
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return {
            'chart_data': fig.to_dict(),
            'chart_config': self.chart_config,
            'chart_type': 'error',
            'error': error_message
        }