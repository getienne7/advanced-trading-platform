"""
Dashboard Widgets - Customizable and responsive dashboard components
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)


class DashboardWidgets:
    """
    Manager for customizable dashboard widgets with responsive design
    """
    
    def __init__(self):
        self.widget_templates = {
            'pnl_summary': self._pnl_summary_template,
            'performance_metrics': self._performance_metrics_template,
            'risk_gauge': self._risk_gauge_template,
            'position_table': self._position_table_template,
            'trade_history': self._trade_history_template,
            'market_overview': self._market_overview_template,
            'alerts_panel': self._alerts_panel_template,
            'strategy_performance': self._strategy_performance_template
        }
        
        self.responsive_breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1440
        }
    
    async def create_widget(
        self,
        widget_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a dashboard widget"""
        try:
            if widget_type not in self.widget_templates:
                raise ValueError(f"Unknown widget type: {widget_type}")
            
            config = config or {}
            template_func = self.widget_templates[widget_type]
            
            widget = await template_func(data, config)
            
            # Add responsive configuration
            widget['responsive'] = self._get_responsive_config(widget_type, config)
            
            # Add widget metadata
            widget['metadata'] = {
                'widget_type': widget_type,
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
                'refresh_interval': config.get('refresh_interval', 30)  # seconds
            }
            
            return widget
            
        except Exception as e:
            logger.error(f"Error creating widget {widget_type}: {e}")
            return self._create_error_widget(f"Error creating {widget_type} widget")
    
    async def create_dashboard_layout(
        self,
        layout_config: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create complete dashboard layout with widgets"""
        try:
            layout = {
                'layout_id': layout_config.get('layout_id', 'default'),
                'title': layout_config.get('title', 'Trading Dashboard'),
                'grid_config': layout_config.get('grid_config', {
                    'columns': 12,
                    'row_height': 60,
                    'margin': [10, 10]
                }),
                'widgets': [],
                'responsive': True,
                'theme': user_preferences.get('theme', 'dark') if user_preferences else 'dark'
            }
            
            # Create widgets based on layout configuration
            for widget_config in layout_config.get('widgets', []):
                widget_data = widget_config.get('data', {})
                widget = await self.create_widget(
                    widget_config['type'],
                    widget_data,
                    widget_config.get('config', {})
                )
                
                # Add layout properties
                widget['layout'] = {
                    'x': widget_config.get('x', 0),
                    'y': widget_config.get('y', 0),
                    'w': widget_config.get('w', 6),
                    'h': widget_config.get('h', 4),
                    'minW': widget_config.get('minW', 2),
                    'minH': widget_config.get('minH', 2),
                    'maxW': widget_config.get('maxW', 12),
                    'maxH': widget_config.get('maxH', 12),
                    'isDraggable': widget_config.get('isDraggable', True),
                    'isResizable': widget_config.get('isResizable', True)
                }
                
                layout['widgets'].append(widget)
            
            return layout
            
        except Exception as e:
            logger.error(f"Error creating dashboard layout: {e}")
            return self._create_error_layout()
    
    async def _pnl_summary_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """P&L summary widget template"""
        total_pnl = data.get('total_pnl', 0)
        unrealized_pnl = data.get('unrealized_pnl', 0)
        realized_pnl = data.get('realized_pnl', 0)
        
        return {
            'widget_type': 'pnl_summary',
            'title': config.get('title', 'P&L Summary'),
            'content': {
                'metrics': [
                    {
                        'label': 'Total P&L',
                        'value': total_pnl,
                        'format': 'currency',
                        'color': 'positive' if total_pnl >= 0 else 'negative',
                        'change': data.get('pnl_change_24h', 0),
                        'change_format': 'percentage'
                    },
                    {
                        'label': 'Unrealized P&L',
                        'value': unrealized_pnl,
                        'format': 'currency',
                        'color': 'positive' if unrealized_pnl >= 0 else 'negative'
                    },
                    {
                        'label': 'Realized P&L',
                        'value': realized_pnl,
                        'format': 'currency',
                        'color': 'positive' if realized_pnl >= 0 else 'negative'
                    }
                ],
                'chart_data': {
                    'type': 'sparkline',
                    'data': data.get('pnl_history', []),
                    'color': 'positive' if total_pnl >= 0 else 'negative'
                }
            },
            'size': config.get('size', 'medium'),
            'refresh_interval': config.get('refresh_interval', 5)
        }
    
    async def _performance_metrics_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Performance metrics widget template"""
        return {
            'widget_type': 'performance_metrics',
            'title': config.get('title', 'Performance Metrics'),
            'content': {
                'metrics': [
                    {
                        'label': 'Total Return',
                        'value': data.get('total_return_pct', 0),
                        'format': 'percentage',
                        'color': 'positive' if data.get('total_return_pct', 0) >= 0 else 'negative'
                    },
                    {
                        'label': 'Sharpe Ratio',
                        'value': data.get('sharpe_ratio', 0),
                        'format': 'decimal',
                        'color': 'positive' if data.get('sharpe_ratio', 0) >= 1 else 'neutral'
                    },
                    {
                        'label': 'Win Rate',
                        'value': data.get('win_rate', 0),
                        'format': 'percentage',
                        'color': 'positive' if data.get('win_rate', 0) >= 0.5 else 'negative'
                    },
                    {
                        'label': 'Max Drawdown',
                        'value': data.get('max_drawdown_pct', 0),
                        'format': 'percentage',
                        'color': 'negative'
                    },
                    {
                        'label': 'Profit Factor',
                        'value': data.get('profit_factor', 0),
                        'format': 'decimal',
                        'color': 'positive' if data.get('profit_factor', 0) >= 1 else 'negative'
                    },
                    {
                        'label': 'Volatility',
                        'value': data.get('volatility', 0),
                        'format': 'percentage',
                        'color': 'neutral'
                    }
                ]
            },
            'size': config.get('size', 'large')
        }
    
    async def _risk_gauge_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Risk gauge widget template"""
        risk_score = data.get('risk_score', 0)
        
        return {
            'widget_type': 'risk_gauge',
            'title': config.get('title', 'Risk Score'),
            'content': {
                'gauge': {
                    'value': risk_score,
                    'min': 0,
                    'max': 10,
                    'thresholds': [
                        {'value': 3, 'color': 'positive', 'label': 'Low Risk'},
                        {'value': 7, 'color': 'neutral', 'label': 'Medium Risk'},
                        {'value': 10, 'color': 'negative', 'label': 'High Risk'}
                    ],
                    'current_level': self._get_risk_level(risk_score)
                },
                'details': [
                    {
                        'label': 'VaR 95%',
                        'value': data.get('var_95', 0),
                        'format': 'currency'
                    },
                    {
                        'label': 'Portfolio Correlation',
                        'value': data.get('avg_correlation', 0),
                        'format': 'decimal'
                    },
                    {
                        'label': 'Concentration Risk',
                        'value': data.get('concentration_risk', 0),
                        'format': 'percentage'
                    }
                ]
            },
            'size': config.get('size', 'medium')
        }
    
    async def _position_table_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Position table widget template"""
        positions = data.get('positions', [])
        
        # Format positions for table display
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                'symbol': pos.get('symbol', ''),
                'side': pos.get('side', ''),
                'size': {
                    'value': pos.get('quantity', 0),
                    'format': 'decimal'
                },
                'entry_price': {
                    'value': pos.get('entry_price', 0),
                    'format': 'currency'
                },
                'current_price': {
                    'value': pos.get('current_price', 0),
                    'format': 'currency'
                },
                'pnl': {
                    'value': pos.get('unrealized_pnl', 0),
                    'format': 'currency',
                    'color': 'positive' if pos.get('unrealized_pnl', 0) >= 0 else 'negative'
                },
                'pnl_pct': {
                    'value': pos.get('pnl_percentage', 0),
                    'format': 'percentage',
                    'color': 'positive' if pos.get('pnl_percentage', 0) >= 0 else 'negative'
                }
            })
        
        return {
            'widget_type': 'position_table',
            'title': config.get('title', 'Open Positions'),
            'content': {
                'columns': [
                    {'key': 'symbol', 'label': 'Symbol', 'sortable': True},
                    {'key': 'side', 'label': 'Side', 'sortable': True},
                    {'key': 'size', 'label': 'Size', 'sortable': True},
                    {'key': 'entry_price', 'label': 'Entry Price', 'sortable': True},
                    {'key': 'current_price', 'label': 'Current Price', 'sortable': True},
                    {'key': 'pnl', 'label': 'P&L', 'sortable': True},
                    {'key': 'pnl_pct', 'label': 'P&L %', 'sortable': True}
                ],
                'rows': formatted_positions,
                'pagination': {
                    'enabled': len(formatted_positions) > 10,
                    'page_size': config.get('page_size', 10)
                },
                'actions': config.get('actions', ['close_position', 'modify_position'])
            },
            'size': config.get('size', 'large')
        }
    
    async def _trade_history_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trade history widget template"""
        trades = data.get('trades', [])
        
        # Format trades for display
        formatted_trades = []
        for trade in trades[-20:]:  # Show last 20 trades
            formatted_trades.append({
                'timestamp': {
                    'value': trade.get('executed_at', ''),
                    'format': 'datetime'
                },
                'symbol': trade.get('symbol', ''),
                'side': {
                    'value': trade.get('side', ''),
                    'color': 'positive' if trade.get('side') == 'BUY' else 'negative'
                },
                'quantity': {
                    'value': trade.get('quantity', 0),
                    'format': 'decimal'
                },
                'price': {
                    'value': trade.get('price', 0),
                    'format': 'currency'
                },
                'pnl': {
                    'value': trade.get('pnl', 0),
                    'format': 'currency',
                    'color': 'positive' if trade.get('pnl', 0) >= 0 else 'negative'
                },
                'fees': {
                    'value': trade.get('fees', 0),
                    'format': 'currency'
                }
            })
        
        return {
            'widget_type': 'trade_history',
            'title': config.get('title', 'Recent Trades'),
            'content': {
                'columns': [
                    {'key': 'timestamp', 'label': 'Time', 'sortable': True},
                    {'key': 'symbol', 'label': 'Symbol', 'sortable': True},
                    {'key': 'side', 'label': 'Side', 'sortable': True},
                    {'key': 'quantity', 'label': 'Quantity', 'sortable': True},
                    {'key': 'price', 'label': 'Price', 'sortable': True},
                    {'key': 'pnl', 'label': 'P&L', 'sortable': True},
                    {'key': 'fees', 'label': 'Fees', 'sortable': True}
                ],
                'rows': formatted_trades,
                'pagination': {
                    'enabled': len(formatted_trades) > 10,
                    'page_size': config.get('page_size', 10)
                }
            },
            'size': config.get('size', 'large')
        }
    
    async def _market_overview_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Market overview widget template"""
        market_data = data.get('market_data', {})
        
        return {
            'widget_type': 'market_overview',
            'title': config.get('title', 'Market Overview'),
            'content': {
                'markets': [
                    {
                        'symbol': 'BTC/USDT',
                        'price': market_data.get('BTC_price', 50000),
                        'change_24h': market_data.get('BTC_change_24h', 2.5),
                        'volume_24h': market_data.get('BTC_volume_24h', 1000000000)
                    },
                    {
                        'symbol': 'ETH/USDT',
                        'price': market_data.get('ETH_price', 3000),
                        'change_24h': market_data.get('ETH_change_24h', 1.8),
                        'volume_24h': market_data.get('ETH_volume_24h', 500000000)
                    }
                ],
                'indices': {
                    'fear_greed': data.get('fear_greed_index', 50),
                    'market_cap': data.get('total_market_cap', 2000000000000),
                    'dominance': {
                        'btc': data.get('btc_dominance', 45),
                        'eth': data.get('eth_dominance', 18)
                    }
                }
            },
            'size': config.get('size', 'medium')
        }
    
    async def _alerts_panel_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Alerts panel widget template"""
        alerts = data.get('alerts', [])
        
        return {
            'widget_type': 'alerts_panel',
            'title': config.get('title', 'Alerts'),
            'content': {
                'alerts': [
                    {
                        'id': alert.get('id'),
                        'type': alert.get('type', 'info'),
                        'title': alert.get('title', ''),
                        'message': alert.get('message', ''),
                        'timestamp': alert.get('timestamp', ''),
                        'priority': alert.get('priority', 'medium'),
                        'read': alert.get('read', False)
                    }
                    for alert in alerts[-10:]  # Show last 10 alerts
                ],
                'summary': {
                    'total': len(alerts),
                    'unread': len([a for a in alerts if not a.get('read', False)]),
                    'critical': len([a for a in alerts if a.get('priority') == 'critical'])
                }
            },
            'size': config.get('size', 'medium')
        }
    
    async def _strategy_performance_template(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Strategy performance widget template"""
        strategies = data.get('strategies', [])
        
        return {
            'widget_type': 'strategy_performance',
            'title': config.get('title', 'Strategy Performance'),
            'content': {
                'strategies': [
                    {
                        'name': strategy.get('name', ''),
                        'status': strategy.get('status', 'active'),
                        'pnl': {
                            'value': strategy.get('pnl', 0),
                            'format': 'currency',
                            'color': 'positive' if strategy.get('pnl', 0) >= 0 else 'negative'
                        },
                        'win_rate': {
                            'value': strategy.get('win_rate', 0),
                            'format': 'percentage'
                        },
                        'trades': strategy.get('trade_count', 0),
                        'allocation': {
                            'value': strategy.get('allocation_pct', 0),
                            'format': 'percentage'
                        }
                    }
                    for strategy in strategies
                ],
                'summary': {
                    'total_strategies': len(strategies),
                    'active_strategies': len([s for s in strategies if s.get('status') == 'active']),
                    'total_pnl': sum(s.get('pnl', 0) for s in strategies)
                }
            },
            'size': config.get('size', 'large')
        }
    
    def _get_responsive_config(
        self,
        widget_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get responsive configuration for widget"""
        base_config = {
            'mobile': {'w': 12, 'h': 4},
            'tablet': {'w': 6, 'h': 4},
            'desktop': {'w': 4, 'h': 4}
        }
        
        # Widget-specific responsive overrides
        widget_responsive = {
            'pnl_summary': {
                'mobile': {'w': 12, 'h': 3},
                'tablet': {'w': 6, 'h': 3},
                'desktop': {'w': 3, 'h': 3}
            },
            'position_table': {
                'mobile': {'w': 12, 'h': 8},
                'tablet': {'w': 12, 'h': 6},
                'desktop': {'w': 8, 'h': 6}
            },
            'risk_gauge': {
                'mobile': {'w': 12, 'h': 4},
                'tablet': {'w': 6, 'h': 4},
                'desktop': {'w': 4, 'h': 4}
            }
        }
        
        return widget_responsive.get(widget_type, base_config)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description"""
        if risk_score <= 3:
            return 'Low Risk'
        elif risk_score <= 7:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def _create_error_widget(self, error_message: str) -> Dict[str, Any]:
        """Create error widget"""
        return {
            'widget_type': 'error',
            'title': 'Error',
            'content': {
                'error': True,
                'message': error_message
            },
            'size': 'small'
        }
    
    def _create_error_layout(self) -> Dict[str, Any]:
        """Create error layout"""
        return {
            'layout_id': 'error',
            'title': 'Error Loading Dashboard',
            'widgets': [self._create_error_widget('Failed to load dashboard layout')],
            'error': True
        }