"""
Report Generator - Automated report generation with PDF output
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import io
import base64
from pathlib import Path
import json

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart

# Email and notifications
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Plotting for report charts
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Automated report generation system with PDF output and email delivery
    """
    
    def __init__(self):
        self.report_templates = {
            'daily': self._daily_report_template,
            'weekly': self._weekly_report_template,
            'monthly': self._monthly_report_template,
            'custom': self._custom_report_template
        }
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Email configuration (would be loaded from environment)
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'reports@tradingplatform.com',
            'password': 'app_password',  # Use app password for Gmail
            'from_email': 'reports@tradingplatform.com'
        }
        
        # Report storage path
        self.reports_path = Path("reports")
        self.reports_path.mkdir(exist_ok=True)
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.darkgreen,
            alignment=1  # Center alignment
        ))
        
        self.styles.add(ParagraphStyle(
            name='NegativeMetric',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.red,
            alignment=1  # Center alignment
        ))
    
    async def generate_report(
        self,
        report_type: str,
        user_id: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a report based on type and data"""
        try:
            if report_type not in self.report_templates:
                raise ValueError(f"Unknown report type: {report_type}")
            
            config = config or {}
            template_func = self.report_templates[report_type]
            
            # Generate report content
            report_content = await template_func(user_id, data, config)
            
            # Generate PDF
            pdf_path = await self._generate_pdf(report_content, user_id, report_type)
            
            # Generate HTML version (optional)
            html_content = await self._generate_html(report_content)
            
            report_info = {
                'report_id': f"{report_type}_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'report_type': report_type,
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat(),
                'pdf_path': str(pdf_path),
                'html_content': html_content,
                'metadata': {
                    'period': data.get('period', {}),
                    'total_pages': report_content.get('total_pages', 1),
                    'sections': list(report_content.get('sections', {}).keys())
                }
            }
            
            return report_info
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def schedule_report(
        self,
        report_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule automated report generation"""
        try:
            schedule_info = {
                'schedule_id': f"schedule_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'report_type': report_config.get('report_type'),
                'user_id': report_config.get('user_id'),
                'frequency': report_config.get('frequency', 'daily'),  # daily, weekly, monthly
                'time': report_config.get('time', '09:00'),  # Time to generate report
                'recipients': report_config.get('recipients', []),
                'enabled': report_config.get('enabled', True),
                'next_run': self._calculate_next_run(
                    report_config.get('frequency', 'daily'),
                    report_config.get('time', '09:00')
                ),
                'created_at': datetime.utcnow().isoformat()
            }
            
            # In a real implementation, this would be stored in a database
            # and processed by a scheduler service
            
            return schedule_info
            
        except Exception as e:
            logger.error(f"Error scheduling report: {e}")
            raise
    
    async def send_report(
        self,
        report_info: Dict[str, Any],
        recipients: List[str],
        delivery_method: str = 'email'
    ) -> Dict[str, Any]:
        """Send report to recipients"""
        try:
            if delivery_method == 'email':
                result = await self._send_email_report(report_info, recipients)
            elif delivery_method == 'slack':
                result = await self._send_slack_report(report_info, recipients)
            else:
                raise ValueError(f"Unknown delivery method: {delivery_method}")
            
            return {
                'success': True,
                'delivery_method': delivery_method,
                'recipients': recipients,
                'sent_at': datetime.utcnow().isoformat(),
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error sending report: {e}")
            return {
                'success': False,
                'error': str(e),
                'delivery_method': delivery_method,
                'recipients': recipients
            }
    
    async def _daily_report_template(
        self,
        user_id: str,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Daily report template"""
        return {
            'title': f'Daily Trading Report - {datetime.utcnow().strftime("%Y-%m-%d")}',
            'sections': {
                'summary': {
                    'title': 'Daily Summary',
                    'content': {
                        'total_pnl': data.get('total_pnl', 0),
                        'realized_pnl': data.get('realized_pnl', 0),
                        'unrealized_pnl': data.get('unrealized_pnl', 0),
                        'trades_count': data.get('trades_count', 0),
                        'win_rate': data.get('win_rate', 0),
                        'best_trade': data.get('best_trade', 0),
                        'worst_trade': data.get('worst_trade', 0)
                    }
                },
                'performance': {
                    'title': 'Performance Metrics',
                    'content': {
                        'daily_return': data.get('daily_return_pct', 0),
                        'volatility': data.get('volatility', 0),
                        'sharpe_ratio': data.get('sharpe_ratio', 0),
                        'max_drawdown': data.get('max_drawdown', 0)
                    }
                },
                'positions': {
                    'title': 'Current Positions',
                    'content': {
                        'positions': data.get('positions', []),
                        'total_exposure': data.get('total_exposure', 0),
                        'leverage_ratio': data.get('leverage_ratio', 1)
                    }
                },
                'risk': {
                    'title': 'Risk Metrics',
                    'content': {
                        'var_95': data.get('var_95', 0),
                        'risk_score': data.get('risk_score', 0),
                        'concentration_risk': data.get('concentration_risk', 0)
                    }
                }
            },
            'charts': {
                'pnl_chart': data.get('pnl_chart_data'),
                'performance_chart': data.get('performance_chart_data')
            }
        }
    
    async def _weekly_report_template(
        self,
        user_id: str,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Weekly report template"""
        return {
            'title': f'Weekly Trading Report - Week of {datetime.utcnow().strftime("%Y-%m-%d")}',
            'sections': {
                'summary': {
                    'title': 'Weekly Summary',
                    'content': {
                        'total_pnl': data.get('total_pnl', 0),
                        'weekly_return': data.get('weekly_return_pct', 0),
                        'trades_count': data.get('trades_count', 0),
                        'win_rate': data.get('win_rate', 0),
                        'profit_factor': data.get('profit_factor', 0)
                    }
                },
                'performance': {
                    'title': 'Performance Analysis',
                    'content': {
                        'sharpe_ratio': data.get('sharpe_ratio', 0),
                        'sortino_ratio': data.get('sortino_ratio', 0),
                        'calmar_ratio': data.get('calmar_ratio', 0),
                        'max_drawdown': data.get('max_drawdown', 0),
                        'recovery_factor': data.get('recovery_factor', 0)
                    }
                },
                'strategy_performance': {
                    'title': 'Strategy Performance',
                    'content': {
                        'strategies': data.get('strategies', [])
                    }
                },
                'market_analysis': {
                    'title': 'Market Analysis',
                    'content': {
                        'market_conditions': data.get('market_conditions', {}),
                        'correlation_analysis': data.get('correlation_analysis', {}),
                        'volatility_analysis': data.get('volatility_analysis', {})
                    }
                }
            },
            'charts': {
                'weekly_pnl': data.get('weekly_pnl_chart'),
                'strategy_comparison': data.get('strategy_comparison_chart'),
                'risk_analysis': data.get('risk_analysis_chart')
            }
        }
    
    async def _monthly_report_template(
        self,
        user_id: str,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monthly report template"""
        return {
            'title': f'Monthly Trading Report - {datetime.utcnow().strftime("%B %Y")}',
            'sections': {
                'executive_summary': {
                    'title': 'Executive Summary',
                    'content': {
                        'monthly_return': data.get('monthly_return_pct', 0),
                        'total_pnl': data.get('total_pnl', 0),
                        'trades_count': data.get('trades_count', 0),
                        'win_rate': data.get('win_rate', 0),
                        'best_strategy': data.get('best_strategy', {}),
                        'key_achievements': data.get('key_achievements', [])
                    }
                },
                'performance_analysis': {
                    'title': 'Detailed Performance Analysis',
                    'content': {
                        'risk_adjusted_returns': data.get('risk_adjusted_returns', {}),
                        'benchmark_comparison': data.get('benchmark_comparison', {}),
                        'attribution_analysis': data.get('attribution_analysis', {})
                    }
                },
                'risk_management': {
                    'title': 'Risk Management Review',
                    'content': {
                        'var_analysis': data.get('var_analysis', {}),
                        'stress_test_results': data.get('stress_test_results', {}),
                        'risk_incidents': data.get('risk_incidents', [])
                    }
                },
                'strategy_review': {
                    'title': 'Strategy Review',
                    'content': {
                        'strategy_performance': data.get('strategy_performance', []),
                        'new_strategies': data.get('new_strategies', []),
                        'strategy_recommendations': data.get('strategy_recommendations', [])
                    }
                },
                'outlook': {
                    'title': 'Market Outlook & Recommendations',
                    'content': {
                        'market_outlook': data.get('market_outlook', ''),
                        'recommendations': data.get('recommendations', []),
                        'risk_warnings': data.get('risk_warnings', [])
                    }
                }
            },
            'charts': {
                'monthly_performance': data.get('monthly_performance_chart'),
                'strategy_allocation': data.get('strategy_allocation_chart'),
                'risk_metrics': data.get('risk_metrics_chart'),
                'correlation_heatmap': data.get('correlation_heatmap')
            }
        }
    
    async def _custom_report_template(
        self,
        user_id: str,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Custom report template based on configuration"""
        sections = {}
        
        # Build sections based on config
        for section_config in config.get('sections', []):
            section_name = section_config['name']
            sections[section_name] = {
                'title': section_config.get('title', section_name.title()),
                'content': data.get(section_name, {})
            }
        
        return {
            'title': config.get('title', 'Custom Trading Report'),
            'sections': sections,
            'charts': data.get('charts', {})
        }
    
    async def _generate_pdf(
        self,
        report_content: Dict[str, Any],
        user_id: str,
        report_type: str
    ) -> Path:
        """Generate PDF report"""
        try:
            # Create filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_type}_report_{user_id}_{timestamp}.pdf"
            pdf_path = self.reports_path / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            story = []
            
            # Title
            title = Paragraph(report_content['title'], self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Generated timestamp
            timestamp_text = f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            story.append(Paragraph(timestamp_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Sections
            for section_name, section_data in report_content.get('sections', {}).items():
                # Section header
                story.append(Paragraph(section_data['title'], self.styles['SectionHeader']))
                story.append(Spacer(1, 12))
                
                # Section content
                content = section_data.get('content', {})
                
                if section_name == 'summary':
                    story.extend(self._create_summary_section(content))
                elif section_name == 'performance':
                    story.extend(self._create_performance_section(content))
                elif section_name == 'positions':
                    story.extend(self._create_positions_section(content))
                elif section_name == 'risk':
                    story.extend(self._create_risk_section(content))
                else:
                    # Generic section
                    story.extend(self._create_generic_section(content))
                
                story.append(Spacer(1, 20))
            
            # Charts (if any)
            charts = report_content.get('charts', {})
            if charts:
                story.append(Paragraph('Charts and Visualizations', self.styles['SectionHeader']))
                story.append(Spacer(1, 12))
                
                for chart_name, chart_data in charts.items():
                    if chart_data:
                        chart_image = await self._create_chart_image(chart_data)
                        if chart_image:
                            story.append(chart_image)
                            story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
    
    def _create_summary_section(self, content: Dict[str, Any]) -> List:
        """Create summary section elements"""
        elements = []
        
        # Create summary table
        data = [
            ['Metric', 'Value'],
            ['Total P&L', f"${content.get('total_pnl', 0):,.2f}"],
            ['Realized P&L', f"${content.get('realized_pnl', 0):,.2f}"],
            ['Unrealized P&L', f"${content.get('unrealized_pnl', 0):,.2f}"],
            ['Trades Count', str(content.get('trades_count', 0))],
            ['Win Rate', f"{content.get('win_rate', 0):.1%}"],
            ['Best Trade', f"${content.get('best_trade', 0):,.2f}"],
            ['Worst Trade', f"${content.get('worst_trade', 0):,.2f}"]
        ]
        
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        return elements
    
    def _create_performance_section(self, content: Dict[str, Any]) -> List:
        """Create performance section elements"""
        elements = []
        
        data = [
            ['Performance Metric', 'Value'],
            ['Daily Return', f"{content.get('daily_return', 0):.2%}"],
            ['Volatility', f"{content.get('volatility', 0):.2%}"],
            ['Sharpe Ratio', f"{content.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{content.get('max_drawdown', 0):.2%}"]
        ]
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        return elements
    
    def _create_positions_section(self, content: Dict[str, Any]) -> List:
        """Create positions section elements"""
        elements = []
        
        positions = content.get('positions', [])
        if positions:
            # Create positions table
            data = [['Symbol', 'Side', 'Size', 'Entry Price', 'Current Price', 'P&L']]
            
            for pos in positions[:10]:  # Limit to 10 positions
                data.append([
                    pos.get('symbol', ''),
                    pos.get('side', ''),
                    f"{pos.get('quantity', 0):.4f}",
                    f"${pos.get('entry_price', 0):,.2f}",
                    f"${pos.get('current_price', 0):,.2f}",
                    f"${pos.get('unrealized_pnl', 0):,.2f}"
                ])
            
            table = Table(data, colWidths=[1*inch, 0.7*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            elements.append(table)
        else:
            elements.append(Paragraph("No open positions", self.styles['Normal']))
        
        return elements
    
    def _create_risk_section(self, content: Dict[str, Any]) -> List:
        """Create risk section elements"""
        elements = []
        
        data = [
            ['Risk Metric', 'Value'],
            ['VaR 95%', f"${content.get('var_95', 0):,.2f}"],
            ['Risk Score', f"{content.get('risk_score', 0):.1f}/10"],
            ['Concentration Risk', f"{content.get('concentration_risk', 0):.1%}"]
        ]
        
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.pink),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        return elements
    
    def _create_generic_section(self, content: Dict[str, Any]) -> List:
        """Create generic section elements"""
        elements = []
        
        if isinstance(content, dict):
            for key, value in content.items():
                text = f"{key.replace('_', ' ').title()}: {value}"
                elements.append(Paragraph(text, self.styles['Normal']))
        elif isinstance(content, list):
            for item in content:
                elements.append(Paragraph(f"• {item}", self.styles['Normal']))
        else:
            elements.append(Paragraph(str(content), self.styles['Normal']))
        
        return elements
    
    async def _create_chart_image(self, chart_data: Dict[str, Any]) -> Optional[Image]:
        """Create chart image for PDF"""
        try:
            # This is a simplified implementation
            # In a real system, you would convert Plotly charts to images
            
            # For now, return None (no chart)
            return None
            
        except Exception as e:
            logger.error(f"Error creating chart image: {e}")
            return None
    
    async def _generate_html(self, report_content: Dict[str, Any]) -> str:
        """Generate HTML version of the report"""
        try:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .section {{ margin: 30px 0; }}
                    .section-title {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
                    .positive {{ color: #27ae60; }}
                    .negative {{ color: #e74c3c; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: center; }}
                    th {{ background: #3498db; color: white; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on: {timestamp}</p>
                </div>
                {sections_html}
            </body>
            </html>
            """
            
            sections_html = ""
            for section_name, section_data in report_content.get('sections', {}).items():
                sections_html += f"""
                <div class="section">
                    <h2 class="section-title">{section_data['title']}</h2>
                    <div class="section-content">
                        {self._format_section_html(section_data.get('content', {}))}
                    </div>
                </div>
                """
            
            html_content = html_template.format(
                title=report_content['title'],
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                sections_html=sections_html
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def _format_section_html(self, content: Dict[str, Any]) -> str:
        """Format section content as HTML"""
        html = ""
        
        for key, value in content.items():
            if isinstance(value, (int, float)):
                css_class = "positive" if value >= 0 else "negative"
                if key.endswith('_pct') or 'rate' in key:
                    formatted_value = f"{value:.2%}"
                elif 'pnl' in key or 'price' in key or key.startswith('var'):
                    formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"{value:,.2f}"
                
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> <span class="{css_class}">{formatted_value}</span></div>'
            else:
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        
        return html
    
    async def _send_email_report(
        self,
        report_info: Dict[str, Any],
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send report via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Trading Report - {report_info['report_type'].title()} - {datetime.utcnow().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
            Dear Trader,
            
            Please find attached your {report_info['report_type']} trading report.
            
            Report Details:
            - Report Type: {report_info['report_type'].title()}
            - Generated: {report_info['generated_at']}
            - Period: {report_info.get('metadata', {}).get('period', 'N/A')}
            
            Best regards,
            Trading Platform Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            pdf_path = report_info.get('pdf_path')
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {Path(pdf_path).name}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], recipients, text)
            server.quit()
            
            return {
                'success': True,
                'message': f'Email sent to {len(recipients)} recipients'
            }
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_slack_report(
        self,
        report_info: Dict[str, Any],
        recipients: List[str]
    ) -> Dict[str, Any]:
        """Send report via Slack"""
        try:
            # This would integrate with Slack API
            # For now, return a mock response
            
            return {
                'success': True,
                'message': f'Slack notification sent to {len(recipients)} channels'
            }
            
        except Exception as e:
            logger.error(f"Error sending Slack report: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_next_run(self, frequency: str, time: str) -> str:
        """Calculate next scheduled run time"""
        try:
            now = datetime.utcnow()
            hour, minute = map(int, time.split(':'))
            
            if frequency == 'daily':
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
            elif frequency == 'weekly':
                # Run on Mondays
                days_ahead = 0 - now.weekday()  # Monday is 0
                if days_ahead <= 0:
                    days_ahead += 7
                next_run = now + timedelta(days=days_ahead)
                next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            elif frequency == 'monthly':
                # Run on first day of next month
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1, hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_run = now + timedelta(days=1)
            
            return next_run.isoformat()
            
        except Exception as e:
            logger.error(f"Error calculating next run: {e}")
            return (datetime.utcnow() + timedelta(days=1)).isoformat()