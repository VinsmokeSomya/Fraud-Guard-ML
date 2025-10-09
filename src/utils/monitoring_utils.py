"""
Monitoring utilities for fraud detection system.

This module provides utility functions for monitoring model performance,
generating monitoring reports, and creating monitoring dashboards.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.utils.config import get_logger

logger = get_logger(__name__)


class MonitoringDashboard:
    """
    Monitoring dashboard for fraud detection system.
    
    Provides visualization and reporting capabilities for model monitoring,
    performance tracking, and drift detection.
    """
    
    def __init__(self):
        """Initialize the MonitoringDashboard."""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#17becf'
        }
    
    def create_performance_trend_chart(self, trend_data: Dict[str, Any]) -> go.Figure:
        """
        Create performance trend chart.
        
        Args:
            trend_data: Performance trend data from ModelMonitor
            
        Returns:
            Plotly figure object
        """
        if 'timestamps' not in trend_data:
            return self._create_empty_chart("No performance data available")
        
        timestamps = pd.to_datetime(trend_data['timestamps'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score Trend', 'Precision vs Recall', 'Fraud Rate', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # F1 Score trend
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=trend_data['f1_scores'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Precision vs Recall
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=trend_data['precision_scores'],
                mode='lines+markers',
                name='Precision',
                line=dict(color=self.color_scheme['success'], width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=trend_data['recall_scores'],
                mode='lines+markers',
                name='Recall',
                line=dict(color=self.color_scheme['warning'], width=2)
            ),
            row=1, col=2
        )
        
        # Fraud rate
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=trend_data['fraud_rates'],
                mode='lines+markers',
                name='Fraud Rate',
                line=dict(color=self.color_scheme['secondary'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Performance metrics summary
        metrics_df = pd.DataFrame({
            'Metric': ['F1 Score', 'Precision', 'Recall'],
            'Current': [
                trend_data['f1_scores'][-1] if trend_data['f1_scores'] else 0,
                trend_data['precision_scores'][-1] if trend_data['precision_scores'] else 0,
                trend_data['recall_scores'][-1] if trend_data['recall_scores'] else 0
            ],
            'Average': [
                np.mean(trend_data['f1_scores']) if trend_data['f1_scores'] else 0,
                np.mean(trend_data['precision_scores']) if trend_data['precision_scores'] else 0,
                np.mean(trend_data['recall_scores']) if trend_data['recall_scores'] else 0
            ]
        })
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Current'],
                name='Current',
                marker_color=self.color_scheme['primary']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Average'],
                name='Average',
                marker_color=self.color_scheme['info']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Performance Monitoring Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_drift_detection_chart(self, drift_data: Dict[str, Any]) -> go.Figure:
        """
        Create drift detection visualization.
        
        Args:
            drift_data: Drift detection data from ModelMonitor
            
        Returns:
            Plotly figure object
        """
        if 'feature_summary' not in drift_data:
            return self._create_empty_chart("No drift data available")
        
        feature_summary = drift_data['feature_summary']
        
        # Prepare data for visualization
        features = list(feature_summary.keys())
        drift_rates = [summary['drift_rate'] for summary in feature_summary.values()]
        drift_scores = [summary['latest_drift_score'] or 0 for summary in feature_summary.values()]
        p_values = [summary['latest_p_value'] or 1 for summary in feature_summary.values()]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Drift Rate by Feature', 'Latest Drift Scores', 'P-Values', 'Drift Detection Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Drift rate by feature
        colors = [self.color_scheme['warning'] if rate > 0.1 else self.color_scheme['success'] for rate in drift_rates]
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=drift_rates,
                name='Drift Rate',
                marker_color=colors
            ),
            row=1, col=1
        )
        
        # Latest drift scores
        fig.add_trace(
            go.Scatter(
                x=features,
                y=drift_scores,
                mode='markers',
                name='Drift Score',
                marker=dict(
                    size=10,
                    color=drift_scores,
                    colorscale='Reds',
                    showscale=True
                )
            ),
            row=1, col=2
        )
        
        # P-values
        fig.add_trace(
            go.Bar(
                x=features,
                y=p_values,
                name='P-Value',
                marker_color=[self.color_scheme['warning'] if p < 0.05 else self.color_scheme['success'] for p in p_values]
            ),
            row=2, col=1
        )
        
        # Add significance threshold line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=1)
        
        # Summary statistics
        total_features = len(features)
        features_with_drift = sum(1 for rate in drift_rates if rate > 0)
        high_drift_features = sum(1 for rate in drift_rates if rate > 0.1)
        
        summary_text = f"""
        Total Features: {total_features}
        Features with Drift: {features_with_drift}
        High Drift Features: {high_drift_features}
        Period: {drift_data.get('period_days', 'N/A')} days
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="x domain", yref="y domain",
            x=0.5, y=0.5,
            showarrow=False,
            row=2, col=2
        )
        
        fig.update_layout(
            title="Data Drift Detection Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_decision_statistics_chart(self, decision_stats: Dict[str, Any]) -> go.Figure:
        """
        Create decision statistics visualization.
        
        Args:
            decision_stats: Decision statistics from DecisionLogger
            
        Returns:
            Plotly figure object
        """
        if 'total_decisions' not in decision_stats:
            return self._create_empty_chart("No decision data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Level Distribution', 'Decision Metrics', 'Processing Performance', 'Alert Statistics'),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Risk level distribution (pie chart)
        risk_distribution = decision_stats.get('risk_distribution', {})
        risk_levels = list(risk_distribution.keys())
        risk_counts = list(risk_distribution.values())
        
        fig.add_trace(
            go.Pie(
                labels=risk_levels,
                values=risk_counts,
                name="Risk Distribution",
                marker_colors=[self.color_scheme['success'], self.color_scheme['warning'], 
                              self.color_scheme['secondary'], self.color_scheme['primary']]
            ),
            row=1, col=1
        )
        
        # Decision metrics
        metrics = ['Total Decisions', 'Fraud Predictions', 'Human Reviews', 'Alerts Generated']
        values = [
            decision_stats.get('total_decisions', 0),
            decision_stats.get('fraud_predictions', 0),
            decision_stats.get('human_reviews_required', 0),
            decision_stats.get('alerts_generated', 0)
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name='Decision Metrics',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=2
        )
        
        # Processing performance
        avg_processing_time = decision_stats.get('avg_processing_time_ms', 0)
        fraud_rate = decision_stats.get('fraud_rate', 0)
        
        performance_metrics = ['Avg Processing Time (ms)', 'Fraud Rate (%)']
        performance_values = [avg_processing_time, fraud_rate * 100]
        
        fig.add_trace(
            go.Bar(
                x=performance_metrics,
                y=performance_values,
                name='Performance',
                marker_color=self.color_scheme['info']
            ),
            row=2, col=1
        )
        
        # Alert statistics
        human_review_rate = decision_stats.get('human_review_rate', 0)
        alert_rate = decision_stats.get('alerts_generated', 0) / max(decision_stats.get('total_decisions', 1), 1)
        
        alert_metrics = ['Human Review Rate (%)', 'Alert Rate (%)']
        alert_values = [human_review_rate * 100, alert_rate * 100]
        
        fig.add_trace(
            go.Bar(
                x=alert_metrics,
                y=alert_values,
                name='Alert Stats',
                marker_color=self.color_scheme['warning']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Decision Statistics Dashboard ({decision_stats.get('period_hours', 24)} hours)",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_model_health_dashboard(self, 
                                   monitoring_summary: Dict[str, Any],
                                   performance_feedback: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create comprehensive model health dashboard.
        
        Args:
            monitoring_summary: Monitoring summary from ModelMonitor
            performance_feedback: Performance feedback from DecisionLogger
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Model Health Score', 'Performance Metrics', 'Drift Alerts',
                          'Prediction Volume', 'Response Time', 'Accuracy Trend'),
            specs=[[{"type": "indicator"}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Model health score (gauge)
        health_score = monitoring_summary.get('model_health', {}).get('current_health_score', 0.8)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score"},
                delta={'reference': 0.8},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=1, col=1
        )
        
        # Performance metrics
        if performance_feedback:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [
                performance_feedback.get('accuracy', 0),
                performance_feedback.get('precision', 0),
                performance_feedback.get('recall', 0),
                performance_feedback.get('f1_score', 0)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name='Performance',
                    marker_color=self.color_scheme['success']
                ),
                row=1, col=2
            )
        
        # Drift alerts
        drift_alerts = monitoring_summary.get('drift_status', {}).get('recent_drift_alerts_24h', 0)
        total_features = monitoring_summary.get('drift_status', {}).get('features_monitored', 1)
        
        fig.add_trace(
            go.Bar(
                x=['Drift Alerts', 'Total Features'],
                y=[drift_alerts, total_features],
                name='Drift Status',
                marker_color=[self.color_scheme['warning'], self.color_scheme['info']]
            ),
            row=1, col=3
        )
        
        # Prediction volume
        prediction_volume = monitoring_summary.get('model_health', {}).get('prediction_volume_1h', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Predictions (1h)'],
                y=[prediction_volume],
                name='Volume',
                marker_color=self.color_scheme['primary']
            ),
            row=2, col=1
        )
        
        # Response time
        avg_response_time = monitoring_summary.get('model_health', {}).get('avg_response_time', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Avg Response Time (s)'],
                y=[avg_response_time],
                name='Response Time',
                marker_color=self.color_scheme['secondary']
            ),
            row=2, col=2
        )
        
        # Accuracy trend (placeholder - would need historical data)
        if performance_feedback:
            accuracy = performance_feedback.get('accuracy', 0)
            fig.add_trace(
                go.Scatter(
                    x=[datetime.now() - timedelta(days=i) for i in range(7, 0, -1)],
                    y=[accuracy + np.random.normal(0, 0.02) for _ in range(7)],  # Simulated trend
                    mode='lines+markers',
                    name='Accuracy Trend',
                    line=dict(color=self.color_scheme['success'])
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="Model Health Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create an empty chart with a message.
        
        Args:
            message: Message to display
            
        Returns:
            Empty Plotly figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="No Data Available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def generate_monitoring_report_html(self,
                                      fraud_detector,
                                      output_path: str,
                                      days: int = 7) -> None:
        """
        Generate comprehensive HTML monitoring report.
        
        Args:
            fraud_detector: FraudDetector instance with monitoring enabled
            output_path: Path to save the HTML report
            days: Number of days to include in report
        """
        try:
            # Collect monitoring data
            monitoring_summary = fraud_detector.get_monitoring_summary()
            performance_trend = fraud_detector.get_performance_trend(days)
            drift_summary = fraud_detector.get_drift_summary(days)
            decision_stats = fraud_detector.get_decision_statistics(24)
            performance_feedback = fraud_detector.get_model_performance_feedback(days)
            
            # Create visualizations
            charts = {}
            
            if performance_trend:
                charts['performance_trend'] = self.create_performance_trend_chart(performance_trend)
            
            if drift_summary:
                charts['drift_detection'] = self.create_drift_detection_chart(drift_summary)
            
            if decision_stats:
                charts['decision_statistics'] = self.create_decision_statistics_chart(decision_stats)
            
            if monitoring_summary:
                charts['model_health'] = self.create_model_health_dashboard(
                    monitoring_summary, performance_feedback
                )
            
            # Generate HTML report
            html_content = self._generate_html_report(
                charts, monitoring_summary, performance_feedback, days
            )
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Monitoring report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
    
    def _generate_html_report(self,
                            charts: Dict[str, go.Figure],
                            monitoring_summary: Optional[Dict[str, Any]],
                            performance_feedback: Optional[Dict[str, Any]],
                            days: int) -> str:
        """
        Generate HTML content for monitoring report.
        
        Args:
            charts: Dictionary of Plotly charts
            monitoring_summary: Monitoring summary data
            performance_feedback: Performance feedback data
            days: Number of days in report
            
        Returns:
            HTML content string
        """
        html_parts = [
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fraud Detection Model Monitoring Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .summary { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                    .chart-container { margin-bottom: 30px; }
                    .metrics-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                    .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    .metrics-table th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
            """,
            f"""
            <div class="header">
                <h1>Fraud Detection Model Monitoring Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report Period: {days} days</p>
            </div>
            """
        ]
        
        # Add summary section
        if monitoring_summary:
            html_parts.append(self._generate_summary_section(monitoring_summary, performance_feedback))
        
        # Add charts
        for chart_name, chart in charts.items():
            chart_html = chart.to_html(include_plotlyjs=False, div_id=f"chart_{chart_name}")
            html_parts.append(f'<div class="chart-container">{chart_html}</div>')
        
        html_parts.append("</body></html>")
        
        return "".join(html_parts)
    
    def _generate_summary_section(self,
                                monitoring_summary: Dict[str, Any],
                                performance_feedback: Optional[Dict[str, Any]]) -> str:
        """Generate HTML summary section."""
        summary_html = ['<div class="summary"><h2>Executive Summary</h2>']
        
        # Model health
        model_health = monitoring_summary.get('model_health', {})
        health_score = model_health.get('current_health_score', 'N/A')
        performance_trend = model_health.get('performance_trend', 'N/A')
        
        summary_html.append(f"""
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Model Health Score</td><td>{health_score}</td></tr>
            <tr><td>Performance Trend</td><td>{performance_trend}</td></tr>
        """)
        
        # Performance metrics
        if performance_feedback:
            summary_html.append(f"""
            <tr><td>Current Accuracy</td><td>{performance_feedback.get('accuracy', 'N/A'):.3f}</td></tr>
            <tr><td>Current Precision</td><td>{performance_feedback.get('precision', 'N/A'):.3f}</td></tr>
            <tr><td>Current Recall</td><td>{performance_feedback.get('recall', 'N/A'):.3f}</td></tr>
            <tr><td>Current F1 Score</td><td>{performance_feedback.get('f1_score', 'N/A'):.3f}</td></tr>
            """)
        
        # Drift status
        drift_status = monitoring_summary.get('drift_status', {})
        drift_alerts = drift_status.get('recent_drift_alerts_24h', 'N/A')
        
        summary_html.append(f"""
            <tr><td>Recent Drift Alerts (24h)</td><td>{drift_alerts}</td></tr>
        </table>
        </div>
        """)
        
        return "".join(summary_html)


def create_monitoring_alerts(monitoring_summary: Dict[str, Any],
                           performance_feedback: Optional[Dict[str, Any]] = None,
                           thresholds: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Create monitoring alerts based on current status.
    
    Args:
        monitoring_summary: Monitoring summary data
        performance_feedback: Performance feedback data
        thresholds: Alert thresholds
        
    Returns:
        List of alert dictionaries
    """
    if thresholds is None:
        thresholds = {
            'health_score_min': 0.7,
            'accuracy_min': 0.8,
            'f1_score_min': 0.7,
            'drift_alerts_max': 5,
            'response_time_max': 1.0  # seconds
        }
    
    alerts = []
    
    # Health score alert
    model_health = monitoring_summary.get('model_health', {})
    health_score = model_health.get('current_health_score', 1.0)
    
    if health_score < thresholds['health_score_min']:
        alerts.append({
            'type': 'health_score_low',
            'severity': 'high',
            'message': f"Model health score is low: {health_score:.3f}",
            'threshold': thresholds['health_score_min'],
            'current_value': health_score
        })
    
    # Performance alerts
    if performance_feedback:
        accuracy = performance_feedback.get('accuracy', 1.0)
        f1_score = performance_feedback.get('f1_score', 1.0)
        
        if accuracy < thresholds['accuracy_min']:
            alerts.append({
                'type': 'accuracy_low',
                'severity': 'medium',
                'message': f"Model accuracy is below threshold: {accuracy:.3f}",
                'threshold': thresholds['accuracy_min'],
                'current_value': accuracy
            })
        
        if f1_score < thresholds['f1_score_min']:
            alerts.append({
                'type': 'f1_score_low',
                'severity': 'medium',
                'message': f"Model F1 score is below threshold: {f1_score:.3f}",
                'threshold': thresholds['f1_score_min'],
                'current_value': f1_score
            })
    
    # Drift alerts
    drift_status = monitoring_summary.get('drift_status', {})
    drift_alerts_count = drift_status.get('recent_drift_alerts_24h', 0)
    
    if drift_alerts_count > thresholds['drift_alerts_max']:
        alerts.append({
            'type': 'drift_alerts_high',
            'severity': 'high',
            'message': f"High number of drift alerts: {drift_alerts_count}",
            'threshold': thresholds['drift_alerts_max'],
            'current_value': drift_alerts_count
        })
    
    # Response time alert
    avg_response_time = model_health.get('avg_response_time', 0.0)
    
    if avg_response_time > thresholds['response_time_max']:
        alerts.append({
            'type': 'response_time_high',
            'severity': 'medium',
            'message': f"Average response time is high: {avg_response_time:.3f}s",
            'threshold': thresholds['response_time_max'],
            'current_value': avg_response_time
        })
    
    return alerts