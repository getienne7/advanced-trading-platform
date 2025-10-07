"""
MLOps Dashboard Configuration
Configuration for monitoring dashboards, alerts, and visualization components.
"""
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json


class MLOpsDashboardConfig:
    """Configuration for MLOps monitoring dashboards."""
    
    def __init__(self):
        # Dashboard layout configuration
        self.dashboard_layout = {
            'overview': {
                'title': 'MLOps Overview',
                'refresh_interval': 30,  # seconds
                'widgets': [
                    'system_health',
                    'model_summary',
                    'active_experiments',
                    'recent_activities'
                ]
            },
            'models': {
                'title': 'Model Management',
                'refresh_interval': 60,
                'widgets': [
                    'model_registry',
                    'model_performance',
                    'deployment_status',
                    'model_comparison'
                ]
            },
            'monitoring': {
                'title': 'Model Monitoring',
                'refresh_interval': 30,
                'widgets': [
                    'drift_detection',
                    'performance_trends',
                    'alert_summary',
                    'data_quality'
                ]
            },
            'experiments': {
                'title': 'A/B Testing',
                'refresh_interval': 60,
                'widgets': [
                    'active_tests',
                    'test_results',
                    'statistical_analysis',
                    'experiment_history'
                ]
            }
        }
        
        # Widget configurations
        self.widget_configs = {
            'system_health': {
                'type': 'status_card',
                'title': 'System Health',
                'metrics': ['status', 'uptime', 'error_rate'],
                'thresholds': {
                    'healthy': {'error_rate': '<0.01'},
                    'degraded': {'error_rate': '<0.05'},
                    'unhealthy': {'error_rate': '>=0.05'}
                }
            },
            'model_summary': {
                'type': 'summary_cards',
                'title': 'Model Summary',
                'cards': [
                    {'title': 'Total Models', 'metric': 'total_models'},
                    {'title': 'Deployed', 'metric': 'deployed_models'},
                    {'title': 'Training', 'metric': 'training_models'},
                    {'title': 'Failed', 'metric': 'failed_models'}
                ]
            },
            'active_experiments': {
                'type': 'table',
                'title': 'Active Experiments',
                'columns': [
                    'experiment_id', 'test_name', 'model_a', 'model_b',
                    'start_time', 'status', 'progress'
                ],
                'max_rows': 10
            },
            'recent_activities': {
                'type': 'timeline',
                'title': 'Recent Activities',
                'max_items': 20,
                'time_range': '24h'
            },
            'model_registry': {
                'type': 'table',
                'title': 'Model Registry',
                'columns': [
                    'model_id', 'type', 'symbol', 'version',
                    'status', 'accuracy', 'created_at'
                ],
                'sortable': True,
                'filterable': True
            },
            'model_performance': {
                'type': 'line_chart',
                'title': 'Model Performance Trends',
                'metrics': ['mse', 'accuracy', 'r2_score'],
                'time_range': '7d',
                'group_by': 'model_type'
            },
            'deployment_status': {
                'type': 'pie_chart',
                'title': 'Deployment Status',
                'metric': 'model_status',
                'colors': {
                    'deployed': '#28a745',
                    'validation': '#ffc107',
                    'training': '#17a2b8',
                    'failed': '#dc3545',
                    'deprecated': '#6c757d'
                }
            },
            'model_comparison': {
                'type': 'comparison_table',
                'title': 'Model Comparison',
                'metrics': ['mse', 'mae', 'accuracy', 'r2_score'],
                'max_models': 5
            },
            'drift_detection': {
                'type': 'gauge_chart',
                'title': 'Drift Detection',
                'metric': 'drift_score',
                'thresholds': {
                    'green': [0, 0.05],
                    'yellow': [0.05, 0.1],
                    'red': [0.1, 1.0]
                }
            },
            'performance_trends': {
                'type': 'multi_line_chart',
                'title': 'Performance Trends',
                'metrics': ['mse', 'accuracy'],
                'time_range': '30d',
                'aggregation': 'daily'
            },
            'alert_summary': {
                'type': 'alert_list',
                'title': 'Active Alerts',
                'severity_colors': {
                    'critical': '#dc3545',
                    'warning': '#ffc107',
                    'info': '#17a2b8'
                },
                'max_alerts': 10
            },
            'data_quality': {
                'type': 'progress_bars',
                'title': 'Data Quality Metrics',
                'metrics': [
                    'completeness', 'consistency', 'timeliness', 'accuracy'
                ],
                'thresholds': {
                    'good': 0.95,
                    'fair': 0.85,
                    'poor': 0.75
                }
            },
            'active_tests': {
                'type': 'card_list',
                'title': 'Active A/B Tests',
                'card_template': {
                    'title': '{test_name}',
                    'subtitle': '{model_a_id} vs {model_b_id}',
                    'metrics': ['progress', 'significance', 'duration']
                }
            },
            'test_results': {
                'type': 'comparison_chart',
                'title': 'A/B Test Results',
                'chart_type': 'bar',
                'metrics': ['mse', 'accuracy', 'precision', 'recall']
            },
            'statistical_analysis': {
                'type': 'statistical_summary',
                'title': 'Statistical Analysis',
                'metrics': [
                    'p_value', 'confidence_interval', 'effect_size',
                    'statistical_power'
                ]
            },
            'experiment_history': {
                'type': 'timeline_chart',
                'title': 'Experiment History',
                'time_range': '90d',
                'group_by': 'test_outcome'
            }
        }
        
        # Alert configurations
        self.alert_configs = {
            'model_performance_degradation': {
                'name': 'Model Performance Degradation',
                'description': 'Model performance has degraded significantly',
                'condition': 'performance_degradation > 0.15',
                'severity': 'warning',
                'notification_channels': ['email', 'slack'],
                'cooldown_minutes': 60
            },
            'data_drift_detected': {
                'name': 'Data Drift Detected',
                'description': 'Significant data drift detected in model inputs',
                'condition': 'drift_score > 0.1',
                'severity': 'warning',
                'notification_channels': ['email', 'slack'],
                'cooldown_minutes': 120
            },
            'model_training_failed': {
                'name': 'Model Training Failed',
                'description': 'Automated model training has failed',
                'condition': 'training_status == "failed"',
                'severity': 'critical',
                'notification_channels': ['email', 'slack', 'pagerduty'],
                'cooldown_minutes': 30
            },
            'deployment_failed': {
                'name': 'Model Deployment Failed',
                'description': 'Model deployment has failed',
                'condition': 'deployment_status == "failed"',
                'severity': 'critical',
                'notification_channels': ['email', 'slack', 'pagerduty'],
                'cooldown_minutes': 15
            },
            'high_prediction_latency': {
                'name': 'High Prediction Latency',
                'description': 'Model prediction latency is above threshold',
                'condition': 'prediction_latency_p95 > 1000',  # ms
                'severity': 'warning',
                'notification_channels': ['slack'],
                'cooldown_minutes': 30
            },
            'low_data_quality': {
                'name': 'Low Data Quality',
                'description': 'Data quality score is below acceptable threshold',
                'condition': 'data_quality_score < 0.8',
                'severity': 'warning',
                'notification_channels': ['email'],
                'cooldown_minutes': 240
            },
            'experiment_inconclusive': {
                'name': 'A/B Test Inconclusive',
                'description': 'A/B test has run for maximum duration without conclusive results',
                'condition': 'test_duration > max_duration AND statistical_significance > 0.05',
                'severity': 'info',
                'notification_channels': ['email'],
                'cooldown_minutes': 1440  # 24 hours
            }
        }
        
        # Notification channel configurations
        self.notification_configs = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True,
                'template_path': 'templates/email_alerts.html',
                'rate_limit': '10/hour'
            },
            'slack': {
                'enabled': True,
                'webhook_url': '',  # To be configured
                'channel': '#mlops-alerts',
                'username': 'MLOps Bot',
                'icon_emoji': ':robot_face:',
                'rate_limit': '20/hour'
            },
            'pagerduty': {
                'enabled': False,
                'integration_key': '',  # To be configured
                'severity_mapping': {
                    'critical': 'critical',
                    'warning': 'warning',
                    'info': 'info'
                }
            },
            'teams': {
                'enabled': False,
                'webhook_url': '',  # To be configured
                'rate_limit': '15/hour'
            }
        }
        
        # Metrics collection configuration
        self.metrics_config = {
            'collection_interval': 60,  # seconds
            'retention_days': 90,
            'aggregation_intervals': ['1m', '5m', '1h', '1d'],
            'metrics': {
                'system_metrics': [
                    'cpu_usage', 'memory_usage', 'disk_usage',
                    'network_io', 'api_requests_per_second'
                ],
                'model_metrics': [
                    'prediction_latency', 'prediction_throughput',
                    'model_accuracy', 'model_error_rate'
                ],
                'business_metrics': [
                    'trading_performance', 'profit_loss',
                    'sharpe_ratio', 'max_drawdown'
                ]
            }
        }
        
        # Export configurations
        self.export_configs = {
            'reports': {
                'daily_summary': {
                    'enabled': True,
                    'schedule': '0 8 * * *',  # 8 AM daily
                    'format': 'pdf',
                    'recipients': ['team@company.com'],
                    'template': 'daily_mlops_summary'
                },
                'weekly_performance': {
                    'enabled': True,
                    'schedule': '0 9 * * 1',  # 9 AM Monday
                    'format': 'pdf',
                    'recipients': ['management@company.com'],
                    'template': 'weekly_performance_report'
                },
                'monthly_analysis': {
                    'enabled': True,
                    'schedule': '0 10 1 * *',  # 10 AM first day of month
                    'format': 'pdf',
                    'recipients': ['stakeholders@company.com'],
                    'template': 'monthly_analysis_report'
                }
            },
            'data_exports': {
                'model_metrics': {
                    'enabled': True,
                    'schedule': '0 2 * * *',  # 2 AM daily
                    'format': 'csv',
                    'destination': 's3://mlops-exports/metrics/',
                    'retention_days': 365
                },
                'experiment_results': {
                    'enabled': True,
                    'schedule': '0 3 * * 0',  # 3 AM Sunday
                    'format': 'json',
                    'destination': 's3://mlops-exports/experiments/',
                    'retention_days': 730
                }
            }
        }
    
    def get_dashboard_config(self, dashboard_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dashboard."""
        return self.dashboard_layout.get(dashboard_name, {})
    
    def get_widget_config(self, widget_name: str) -> Dict[str, Any]:
        """Get configuration for a specific widget."""
        return self.widget_configs.get(widget_name, {})
    
    def get_alert_config(self, alert_name: str) -> Dict[str, Any]:
        """Get configuration for a specific alert."""
        return self.alert_configs.get(alert_name, {})
    
    def get_notification_config(self, channel: str) -> Dict[str, Any]:
        """Get configuration for a notification channel."""
        return self.notification_configs.get(channel, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dashboard_layout': self.dashboard_layout,
            'widget_configs': self.widget_configs,
            'alert_configs': self.alert_configs,
            'notification_configs': self.notification_configs,
            'metrics_config': self.metrics_config,
            'export_configs': self.export_configs
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MLOpsDashboardConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        config = cls()
        config.dashboard_layout = config_data.get('dashboard_layout', config.dashboard_layout)
        config.widget_configs = config_data.get('widget_configs', config.widget_configs)
        config.alert_configs = config_data.get('alert_configs', config.alert_configs)
        config.notification_configs = config_data.get('notification_configs', config.notification_configs)
        config.metrics_config = config_data.get('metrics_config', config.metrics_config)
        config.export_configs = config_data.get('export_configs', config.export_configs)
        
        return config


# Global dashboard configuration instance
mlops_dashboard_config = MLOpsDashboardConfig()


# Dashboard template configurations
DASHBOARD_TEMPLATES = {
    'overview': {
        'title': 'MLOps Overview Dashboard',
        'description': 'High-level overview of MLOps system health and performance',
        'layout': 'grid',
        'grid_columns': 4,
        'widgets': [
            {'name': 'system_health', 'position': [0, 0], 'size': [2, 1]},
            {'name': 'model_summary', 'position': [2, 0], 'size': [2, 1]},
            {'name': 'active_experiments', 'position': [0, 1], 'size': [4, 2]},
            {'name': 'recent_activities', 'position': [0, 3], 'size': [4, 2]}
        ]
    },
    'model_monitoring': {
        'title': 'Model Monitoring Dashboard',
        'description': 'Detailed monitoring of model performance and drift',
        'layout': 'grid',
        'grid_columns': 3,
        'widgets': [
            {'name': 'drift_detection', 'position': [0, 0], 'size': [1, 1]},
            {'name': 'performance_trends', 'position': [1, 0], 'size': [2, 1]},
            {'name': 'alert_summary', 'position': [0, 1], 'size': [1, 2]},
            {'name': 'data_quality', 'position': [1, 1], 'size': [2, 1]},
            {'name': 'model_comparison', 'position': [1, 2], 'size': [2, 1]}
        ]
    },
    'experiment_tracking': {
        'title': 'Experiment Tracking Dashboard',
        'description': 'A/B testing and experiment management',
        'layout': 'grid',
        'grid_columns': 2,
        'widgets': [
            {'name': 'active_tests', 'position': [0, 0], 'size': [1, 2]},
            {'name': 'test_results', 'position': [1, 0], 'size': [1, 1]},
            {'name': 'statistical_analysis', 'position': [1, 1], 'size': [1, 1]},
            {'name': 'experiment_history', 'position': [0, 2], 'size': [2, 1]}
        ]
    }
}


# Alert message templates
ALERT_TEMPLATES = {
    'email': {
        'subject': '[MLOps Alert] {alert_name}',
        'body': '''
        <html>
        <body>
            <h2>MLOps Alert: {alert_name}</h2>
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Details:</strong></p>
            <ul>
                {details}
            </ul>
            <p>Please investigate and take appropriate action.</p>
            <p>Best regards,<br>MLOps Monitoring System</p>
        </body>
        </html>
        '''
    },
    'slack': {
        'text': ':warning: *MLOps Alert: {alert_name}*',
        'blocks': [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': '*{alert_name}*\n{description}'
                }
            },
            {
                'type': 'section',
                'fields': [
                    {'type': 'mrkdwn', 'text': '*Severity:*\n{severity}'},
                    {'type': 'mrkdwn', 'text': '*Time:*\n{timestamp}'}
                ]
            }
        ]
    }
}