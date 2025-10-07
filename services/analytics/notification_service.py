"""
Notification Service - Email, Slack, and other notification delivery
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import aiohttp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service for sending notifications via multiple channels
    """
    
    def __init__(self):
        # Email configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'notifications@tradingplatform.com',
            'password': 'app_password',
            'from_email': 'notifications@tradingplatform.com'
        }
        
        # Slack configuration
        self.slack_config = {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'bot_token': 'xoxb-your-bot-token'
        }
        
        # SMS configuration (Twilio)
        self.sms_config = {
            'account_sid': 'your_twilio_account_sid',
            'auth_token': 'your_twilio_auth_token',
            'from_number': '+1234567890'
        }
        
        # Notification templates
        self.templates = {
            'risk_alert': self._risk_alert_template,
            'performance_update': self._performance_update_template,
            'trade_execution': self._trade_execution_template,
            'system_alert': self._system_alert_template,
            'report_ready': self._report_ready_template
        }
    
    async def send_notification(
        self,
        notification_type: str,
        recipients: List[Dict[str, Any]],
        data: Dict[str, Any],
        channels: List[str] = ['email']
    ) -> Dict[str, Any]:
        """Send notification via specified channels"""
        try:
            if notification_type not in self.templates:
                raise ValueError(f"Unknown notification type: {notification_type}")
            
            # Generate notification content
            template_func = self.templates[notification_type]
            content = await template_func(data)
            
            results = {}
            
            # Send via each channel
            for channel in channels:
                if channel == 'email':
                    email_recipients = [r['email'] for r in recipients if 'email' in r]
                    if email_recipients:
                        results['email'] = await self._send_email(content, email_recipients)
                
                elif channel == 'slack':
                    slack_recipients = [r['slack_channel'] for r in recipients if 'slack_channel' in r]
                    if slack_recipients:
                        results['slack'] = await self._send_slack(content, slack_recipients)
                
                elif channel == 'sms':
                    sms_recipients = [r['phone'] for r in recipients if 'phone' in r]
                    if sms_recipients:
                        results['sms'] = await self._send_sms(content, sms_recipients)
                
                elif channel == 'webhook':
                    webhook_recipients = [r['webhook_url'] for r in recipients if 'webhook_url' in r]
                    if webhook_recipients:
                        results['webhook'] = await self._send_webhook(content, webhook_recipients)
            
            return {
                'success': True,
                'notification_type': notification_type,
                'channels': channels,
                'results': results,
                'sent_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return {
                'success': False,
                'error': str(e),
                'notification_type': notification_type,
                'channels': channels
            }
    
    async def send_bulk_notifications(
        self,
        notifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Send multiple notifications"""
        results = []
        
        # Process notifications concurrently
        tasks = []
        for notification in notifications:
            task = self.send_notification(
                notification['type'],
                notification['recipients'],
                notification['data'],
                notification.get('channels', ['email'])
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def schedule_notification(
        self,
        notification_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule a notification for future delivery"""
        try:
            schedule_info = {
                'schedule_id': f"notif_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'notification_type': notification_config.get('type'),
                'recipients': notification_config.get('recipients', []),
                'channels': notification_config.get('channels', ['email']),
                'data': notification_config.get('data', {}),
                'scheduled_time': notification_config.get('scheduled_time'),
                'repeat': notification_config.get('repeat', False),
                'repeat_interval': notification_config.get('repeat_interval', 'daily'),
                'created_at': datetime.utcnow().isoformat(),
                'status': 'scheduled'
            }
            
            # In a real implementation, this would be stored in a database
            # and processed by a scheduler service
            
            return schedule_info
            
        except Exception as e:
            logger.error(f"Error scheduling notification: {e}")
            raise
    
    async def _risk_alert_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Risk alert notification template"""
        risk_level = data.get('risk_level', 'medium')
        risk_score = data.get('risk_score', 0)
        
        # Determine urgency and styling
        if risk_level == 'critical' or risk_score >= 8:
            urgency = 'CRITICAL'
            color = '#ff0000'
            emoji = '🚨'
        elif risk_level == 'high' or risk_score >= 6:
            urgency = 'HIGH'
            color = '#ff6600'
            emoji = '⚠️'
        else:
            urgency = 'MEDIUM'
            color = '#ffaa00'
            emoji = '⚡'
        
        return {
            'subject': f'{emoji} {urgency} Risk Alert - Trading Platform',
            'title': f'{urgency} Risk Alert',
            'message': f"""
            Risk alert triggered for your trading account.
            
            Risk Details:
            • Risk Score: {risk_score}/10
            • Risk Level: {risk_level.upper()}
            • Trigger: {data.get('trigger', 'Unknown')}
            • Current VaR: ${data.get('var_95', 0):,.2f}
            • Portfolio Value: ${data.get('portfolio_value', 0):,.2f}
            
            Recommended Actions:
            {self._format_list(data.get('recommendations', []))}
            
            Please review your positions and consider reducing risk exposure.
            """,
            'html_message': f"""
            <div style="border-left: 4px solid {color}; padding: 20px; background: #f9f9f9;">
                <h2 style="color: {color};">{emoji} {urgency} Risk Alert</h2>
                <p>Risk alert triggered for your trading account.</p>
                
                <h3>Risk Details:</h3>
                <ul>
                    <li><strong>Risk Score:</strong> {risk_score}/10</li>
                    <li><strong>Risk Level:</strong> {risk_level.upper()}</li>
                    <li><strong>Trigger:</strong> {data.get('trigger', 'Unknown')}</li>
                    <li><strong>Current VaR:</strong> ${data.get('var_95', 0):,.2f}</li>
                    <li><strong>Portfolio Value:</strong> ${data.get('portfolio_value', 0):,.2f}</li>
                </ul>
                
                <h3>Recommended Actions:</h3>
                <ul>
                    {self._format_html_list(data.get('recommendations', []))}
                </ul>
                
                <p><strong>Please review your positions and consider reducing risk exposure.</strong></p>
            </div>
            """,
            'slack_message': {
                'text': f'{emoji} {urgency} Risk Alert',
                'attachments': [
                    {
                        'color': color,
                        'fields': [
                            {'title': 'Risk Score', 'value': f'{risk_score}/10', 'short': True},
                            {'title': 'Risk Level', 'value': risk_level.upper(), 'short': True},
                            {'title': 'Current VaR', 'value': f'${data.get("var_95", 0):,.2f}', 'short': True},
                            {'title': 'Portfolio Value', 'value': f'${data.get("portfolio_value", 0):,.2f}', 'short': True}
                        ]
                    }
                ]
            },
            'priority': 'high' if urgency in ['HIGH', 'CRITICAL'] else 'medium'
        }
    
    async def _performance_update_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Performance update notification template"""
        total_return = data.get('total_return_pct', 0)
        period = data.get('period', 'daily')
        
        emoji = '📈' if total_return >= 0 else '📉'
        color = '#00aa00' if total_return >= 0 else '#aa0000'
        
        return {
            'subject': f'{emoji} {period.title()} Performance Update - {total_return:+.2%}',
            'title': f'{period.title()} Performance Update',
            'message': f"""
            Your {period} performance summary:
            
            Performance Metrics:
            • Total Return: {total_return:+.2%}
            • P&L: ${data.get('total_pnl', 0):+,.2f}
            • Sharpe Ratio: {data.get('sharpe_ratio', 0):.2f}
            • Win Rate: {data.get('win_rate', 0):.1%}
            • Max Drawdown: {data.get('max_drawdown_pct', 0):.2%}
            
            Trade Summary:
            • Total Trades: {data.get('trades_count', 0)}
            • Best Trade: ${data.get('best_trade', 0):,.2f}
            • Worst Trade: ${data.get('worst_trade', 0):,.2f}
            
            Keep up the great work!
            """,
            'html_message': f"""
            <div style="border: 1px solid {color}; padding: 20px; border-radius: 5px;">
                <h2 style="color: {color};">{emoji} {period.title()} Performance Update</h2>
                
                <h3>Performance Metrics:</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td><strong>Total Return:</strong></td><td style="color: {color};">{total_return:+.2%}</td></tr>
                    <tr><td><strong>P&L:</strong></td><td style="color: {color};">${data.get('total_pnl', 0):+,.2f}</td></tr>
                    <tr><td><strong>Sharpe Ratio:</strong></td><td>{data.get('sharpe_ratio', 0):.2f}</td></tr>
                    <tr><td><strong>Win Rate:</strong></td><td>{data.get('win_rate', 0):.1%}</td></tr>
                    <tr><td><strong>Max Drawdown:</strong></td><td>{data.get('max_drawdown_pct', 0):.2%}</td></tr>
                </table>
                
                <h3>Trade Summary:</h3>
                <ul>
                    <li><strong>Total Trades:</strong> {data.get('trades_count', 0)}</li>
                    <li><strong>Best Trade:</strong> ${data.get('best_trade', 0):,.2f}</li>
                    <li><strong>Worst Trade:</strong> ${data.get('worst_trade', 0):,.2f}</li>
                </ul>
                
                <p><em>Keep up the great work!</em></p>
            </div>
            """,
            'slack_message': {
                'text': f'{emoji} {period.title()} Performance Update',
                'attachments': [
                    {
                        'color': color,
                        'fields': [
                            {'title': 'Total Return', 'value': f'{total_return:+.2%}', 'short': True},
                            {'title': 'P&L', 'value': f'${data.get("total_pnl", 0):+,.2f}', 'short': True},
                            {'title': 'Sharpe Ratio', 'value': f'{data.get("sharpe_ratio", 0):.2f}', 'short': True},
                            {'title': 'Win Rate', 'value': f'{data.get("win_rate", 0):.1%}', 'short': True}
                        ]
                    }
                ]
            },
            'priority': 'medium'
        }
    
    async def _trade_execution_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trade execution notification template"""
        side = data.get('side', 'BUY')
        symbol = data.get('symbol', '')
        quantity = data.get('quantity', 0)
        price = data.get('price', 0)
        
        emoji = '🟢' if side == 'BUY' else '🔴'
        
        return {
            'subject': f'{emoji} Trade Executed: {side} {quantity} {symbol} @ ${price:,.2f}',
            'title': 'Trade Execution Confirmation',
            'message': f"""
            Trade executed successfully:
            
            Trade Details:
            • Symbol: {symbol}
            • Side: {side}
            • Quantity: {quantity}
            • Price: ${price:,.2f}
            • Total Value: ${quantity * price:,.2f}
            • Fees: ${data.get('fees', 0):.2f}
            • Strategy: {data.get('strategy', 'Manual')}
            
            Execution Time: {data.get('executed_at', datetime.utcnow().isoformat())}
            """,
            'slack_message': {
                'text': f'{emoji} Trade Executed',
                'attachments': [
                    {
                        'color': '#0066cc',
                        'fields': [
                            {'title': 'Symbol', 'value': symbol, 'short': True},
                            {'title': 'Side', 'value': side, 'short': True},
                            {'title': 'Quantity', 'value': str(quantity), 'short': True},
                            {'title': 'Price', 'value': f'${price:,.2f}', 'short': True}
                        ]
                    }
                ]
            },
            'priority': 'low'
        }
    
    async def _system_alert_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """System alert notification template"""
        alert_type = data.get('alert_type', 'info')
        message = data.get('message', '')
        
        if alert_type == 'error':
            emoji = '❌'
            color = '#ff0000'
        elif alert_type == 'warning':
            emoji = '⚠️'
            color = '#ffaa00'
        else:
            emoji = 'ℹ️'
            color = '#0066cc'
        
        return {
            'subject': f'{emoji} System Alert: {alert_type.upper()}',
            'title': f'System Alert - {alert_type.upper()}',
            'message': f"""
            System alert notification:
            
            Alert Type: {alert_type.upper()}
            Message: {message}
            Service: {data.get('service', 'Unknown')}
            Timestamp: {data.get('timestamp', datetime.utcnow().isoformat())}
            
            {data.get('details', '')}
            """,
            'slack_message': {
                'text': f'{emoji} System Alert: {alert_type.upper()}',
                'attachments': [
                    {
                        'color': color,
                        'text': message,
                        'fields': [
                            {'title': 'Service', 'value': data.get('service', 'Unknown'), 'short': True},
                            {'title': 'Timestamp', 'value': data.get('timestamp', datetime.utcnow().isoformat()), 'short': True}
                        ]
                    }
                ]
            },
            'priority': 'high' if alert_type == 'error' else 'medium'
        }
    
    async def _report_ready_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Report ready notification template"""
        report_type = data.get('report_type', 'trading')
        
        return {
            'subject': f'📊 Your {report_type.title()} Report is Ready',
            'title': f'{report_type.title()} Report Ready',
            'message': f"""
            Your {report_type} report has been generated and is ready for download.
            
            Report Details:
            • Report Type: {report_type.title()}
            • Period: {data.get('period', 'N/A')}
            • Generated: {data.get('generated_at', datetime.utcnow().isoformat())}
            • Pages: {data.get('pages', 'N/A')}
            
            The report has been attached to this email or can be downloaded from your dashboard.
            """,
            'slack_message': {
                'text': f'📊 Your {report_type.title()} Report is Ready',
                'attachments': [
                    {
                        'color': '#00aa00',
                        'fields': [
                            {'title': 'Report Type', 'value': report_type.title(), 'short': True},
                            {'title': 'Period', 'value': data.get('period', 'N/A'), 'short': True}
                        ]
                    }
                ]
            },
            'priority': 'low'
        }
    
    async def _send_email(
        self,
        content: Dict[str, Any],
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = content['subject']
            
            # Plain text version
            text_part = MIMEText(content['message'], 'plain')
            msg.attach(text_part)
            
            # HTML version (if available)
            if 'html_message' in content:
                html_part = MIMEText(content['html_message'], 'html')
                msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.sendmail(self.email_config['from_email'], recipients, msg.as_string())
            server.quit()
            
            return {
                'success': True,
                'recipients_count': len(recipients),
                'message': 'Email sent successfully'
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_slack(
        self,
        content: Dict[str, Any],
        channels: List[str]
    ) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            results = []
            
            for channel in channels:
                slack_payload = content.get('slack_message', {
                    'text': content['message']
                })
                
                # Add channel to payload
                slack_payload['channel'] = channel
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.slack_config['webhook_url'],
                        json=slack_payload
                    ) as response:
                        if response.status == 200:
                            results.append({'channel': channel, 'success': True})
                        else:
                            results.append({
                                'channel': channel,
                                'success': False,
                                'error': f'HTTP {response.status}'
                            })
            
            success_count = sum(1 for r in results if r['success'])
            
            return {
                'success': success_count > 0,
                'results': results,
                'success_count': success_count,
                'total_count': len(channels)
            }
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_sms(
        self,
        content: Dict[str, Any],
        phone_numbers: List[str]
    ) -> Dict[str, Any]:
        """Send SMS notification"""
        try:
            # This would integrate with Twilio or another SMS service
            # For now, return a mock response
            
            return {
                'success': True,
                'recipients_count': len(phone_numbers),
                'message': 'SMS sent successfully (mock)'
            }
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_webhook(
        self,
        content: Dict[str, Any],
        webhook_urls: List[str]
    ) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            results = []
            
            webhook_payload = {
                'notification_type': content.get('notification_type', 'unknown'),
                'title': content['title'],
                'message': content['message'],
                'priority': content.get('priority', 'medium'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for url in webhook_urls:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=webhook_payload) as response:
                        if response.status == 200:
                            results.append({'url': url, 'success': True})
                        else:
                            results.append({
                                'url': url,
                                'success': False,
                                'error': f'HTTP {response.status}'
                            })
            
            success_count = sum(1 for r in results if r['success'])
            
            return {
                'success': success_count > 0,
                'results': results,
                'success_count': success_count,
                'total_count': len(webhook_urls)
            }
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_list(self, items: List[str]) -> str:
        """Format list items for plain text"""
        return '\n'.join(f'• {item}' for item in items)
    
    def _format_html_list(self, items: List[str]) -> str:
        """Format list items for HTML"""
        return '\n'.join(f'<li>{item}</li>' for item in items)