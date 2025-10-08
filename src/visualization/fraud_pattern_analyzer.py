"""
FraudPatternAnalyzer module for analyzing fraud patterns and risk factors.

This module provides the FraudPatternAnalyzer class for creating comprehensive
fraud pattern analysis including time series analysis, customer behavior analysis,
and risk factor identification and ranking.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FraudPatternAnalyzer:
    """
    FraudPatternAnalyzer class for analyzing fraud patterns and risk factors.
    
    This class provides comprehensive fraud pattern analysis capabilities including
    time series analysis of fraud occurrence, customer behavior analysis, and
    risk factor identification and ranking.
    """
    
    def __init__(self, 
                 style: str = 'whitegrid',
                 palette: str = 'Set2',
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 100):
        """
        Initialize FraudPatternAnalyzer with styling options.
        
        Args:
            style: Seaborn style for matplotlib plots
            palette: Color palette for visualizations
            figure_size: Default figure size for matplotlib plots
            dpi: DPI for matplotlib figures
        """
        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Set up plotting style
        sns.set_style(self.style)
        sns.set_palette(self.palette)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        
        logger.info(f"FraudPatternAnalyzer initialized with style: {style}, palette: {palette}")
    
    def plot_fraud_time_series(self, 
                              df: pd.DataFrame,
                              time_unit: str = 'hour',
                              aggregation: str = 'count',
                              interactive: bool = False,
                              show_trend: bool = True) -> Dict[str, Any]:
        """
        Create time series plots of fraud occurrence.
        
        Args:
            df: DataFrame containing transaction data with 'step' and 'isFraud' columns
            time_unit: Time unit for aggregation ('hour', 'day', 'step')
            aggregation: Type of aggregation ('count', 'rate', 'amount')
            interactive: Whether to create interactive plotly charts
            show_trend: Whether to show trend line
            
        Returns:
            Dictionary containing time series data and statistics
        """
        logger.info(f"Creating fraud time series plot with {time_unit} aggregation")
        
        if 'isFraud' not in df.columns:
            logger.error("DataFrame must contain 'isFraud' column")
            return {'error': 'Missing isFraud column'}
        
        if 'step' not in df.columns:
            logger.error("DataFrame must contain 'step' column")
            return {'error': 'Missing step column'}
        
        # Create time feature based on step
        df_temp = df.copy()
        if time_unit == 'hour':
            df_temp['time_feature'] = df_temp['step'] % 24
            xlabel = 'Hour of Day'
            title = 'Fraud Occurrence by Hour of Day'
        elif time_unit == 'day':
            df_temp['time_feature'] = df_temp['step'] // 24
            xlabel = 'Day'
            title = 'Fraud Occurrence by Day'
        else:  # step
            df_temp['time_feature'] = df_temp['step']
            xlabel = 'Time Step'
            title = 'Fraud Occurrence by Time Step'
        
        # Calculate fraud statistics by time
        if aggregation == 'count':
            fraud_stats = df_temp[df_temp['isFraud'] == 1].groupby('time_feature').size()
            ylabel = 'Fraud Count'
            title += ' (Count)'
        elif aggregation == 'rate':
            total_by_time = df_temp.groupby('time_feature').size()
            fraud_by_time = df_temp[df_temp['isFraud'] == 1].groupby('time_feature').size()
            fraud_stats = (fraud_by_time / total_by_time * 100).fillna(0)
            ylabel = 'Fraud Rate (%)'
            title += ' (Rate)'
        elif aggregation == 'amount':
            if 'amount' not in df.columns:
                logger.error("DataFrame must contain 'amount' column for amount aggregation")
                return {'error': 'Missing amount column'}
            fraud_stats = df_temp[df_temp['isFraud'] == 1].groupby('time_feature')['amount'].sum()
            ylabel = 'Fraud Amount'
            title += ' (Amount)'
        else:
            logger.error(f"Unknown aggregation type: {aggregation}")
            return {'error': f'Unknown aggregation type: {aggregation}'}
        
        # Ensure all time periods are represented
        if time_unit == 'hour':
            all_times = pd.Series(range(24))
        elif time_unit == 'day':
            max_day = df_temp['time_feature'].max()
            all_times = pd.Series(range(max_day + 1))
        else:
            max_step = df_temp['time_feature'].max()
            all_times = pd.Series(range(max_step + 1))
        
        fraud_stats = fraud_stats.reindex(all_times, fill_value=0)
        
        if interactive:
            return self._plot_fraud_time_series_interactive(fraud_stats, xlabel, ylabel, title, show_trend)
        else:
            return self._plot_fraud_time_series_static(fraud_stats, xlabel, ylabel, title, show_trend)
    
    def analyze_customer_behavior(self, 
                                 df: pd.DataFrame,
                                 interactive: bool = False) -> Dict[str, Any]:
        """
        Build customer behavior analysis charts.
        
        Args:
            df: DataFrame containing transaction data
            interactive: Whether to create interactive plotly charts
            
        Returns:
            Dictionary containing customer behavior analysis results
        """
        logger.info("Analyzing customer behavior patterns")
        
        required_cols = ['nameOrig', 'nameDest', 'oldbalanceOrg', 'newbalanceOrig', 
                        'oldbalanceDest', 'newbalanceDest', 'amount', 'isFraud']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {'error': f'Missing required columns: {missing_cols}'}
        
        # Calculate balance changes
        df_temp = df.copy()
        df_temp['balance_change_orig'] = df_temp['newbalanceOrig'] - df_temp['oldbalanceOrg']
        df_temp['balance_change_dest'] = df_temp['newbalanceDest'] - df_temp['oldbalanceDest']
        
        # Identify merchant accounts (names starting with 'M')
        df_temp['is_merchant_orig'] = df_temp['nameOrig'].str.startswith('M')
        df_temp['is_merchant_dest'] = df_temp['nameDest'].str.startswith('M')
        
        # Calculate unusual balance changes (detect anomalies)
        balance_change_stats = {
            'orig_mean': df_temp['balance_change_orig'].mean(),
            'orig_std': df_temp['balance_change_orig'].std(),
            'dest_mean': df_temp['balance_change_dest'].mean(),
            'dest_std': df_temp['balance_change_dest'].std()
        }
        
        # Flag unusual balance changes (beyond 2 standard deviations)
        df_temp['unusual_balance_orig'] = np.abs(df_temp['balance_change_orig'] - balance_change_stats['orig_mean']) > 2 * balance_change_stats['orig_std']
        df_temp['unusual_balance_dest'] = np.abs(df_temp['balance_change_dest'] - balance_change_stats['dest_mean']) > 2 * balance_change_stats['dest_std']
        
        if interactive:
            return self._analyze_customer_behavior_interactive(df_temp, balance_change_stats)
        else:
            return self._analyze_customer_behavior_static(df_temp, balance_change_stats)
    
    def identify_risk_factors(self, 
                             df: pd.DataFrame,
                             top_n: int = 15) -> Dict[str, Any]:
        """
        Add risk factor identification and ranking.
        
        Args:
            df: DataFrame containing transaction data
            top_n: Number of top risk factors to return
            
        Returns:
            Dictionary containing risk factor analysis results
        """
        logger.info("Identifying and ranking risk factors")
        
        if 'isFraud' not in df.columns:
            logger.error("DataFrame must contain 'isFraud' column")
            return {'error': 'Missing isFraud column'}
        
        risk_factors = {}
        
        # 1. Transaction type risk
        if 'type' in df.columns:
            type_risk = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            type_risk.columns = ['type', 'total_count', 'fraud_count', 'fraud_rate']
            type_risk = type_risk.sort_values('fraud_rate', ascending=False)
            risk_factors['transaction_type'] = type_risk.to_dict('records')
        
        # 2. Amount-based risk factors
        if 'amount' in df.columns:
            # Large transfer risk (>200,000 as per business rules)
            large_transfers = df[df['amount'] > 200000]
            if len(large_transfers) > 0:
                large_transfer_fraud_rate = large_transfers['isFraud'].mean()
                risk_factors['large_transfers'] = {
                    'threshold': 200000,
                    'total_count': len(large_transfers),
                    'fraud_count': large_transfers['isFraud'].sum(),
                    'fraud_rate': large_transfer_fraud_rate
                }
            
            # Amount percentile risk
            amount_percentiles = [50, 75, 90, 95, 99]
            amount_risk = []
            for percentile in amount_percentiles:
                threshold = np.percentile(df['amount'], percentile)
                high_amount_txns = df[df['amount'] >= threshold]
                if len(high_amount_txns) > 0:
                    fraud_rate = high_amount_txns['isFraud'].mean()
                    amount_risk.append({
                        'percentile': percentile,
                        'threshold': threshold,
                        'total_count': len(high_amount_txns),
                        'fraud_count': high_amount_txns['isFraud'].sum(),
                        'fraud_rate': fraud_rate
                    })
            risk_factors['amount_percentiles'] = amount_risk
        
        # 3. Balance-based risk factors
        balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        available_balance_cols = [col for col in balance_cols if col in df.columns]
        
        if len(available_balance_cols) >= 2:
            # Zero balance risk
            zero_balance_risk = []
            for col in available_balance_cols:
                zero_balance_txns = df[df[col] == 0]
                if len(zero_balance_txns) > 0:
                    fraud_rate = zero_balance_txns['isFraud'].mean()
                    zero_balance_risk.append({
                        'balance_type': col,
                        'total_count': len(zero_balance_txns),
                        'fraud_count': zero_balance_txns['isFraud'].sum(),
                        'fraud_rate': fraud_rate
                    })
            risk_factors['zero_balance'] = zero_balance_risk
            
            # Balance change patterns
            if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
                df_temp = df.copy()
                df_temp['balance_change_orig'] = df_temp['newbalanceOrig'] - df_temp['oldbalanceOrg']
                
                # Exact balance depletion (newbalance = 0 and oldbalance > 0)
                balance_depletion = df_temp[(df_temp['oldbalanceOrg'] > 0) & (df_temp['newbalanceOrig'] == 0)]
                if len(balance_depletion) > 0:
                    risk_factors['balance_depletion'] = {
                        'total_count': len(balance_depletion),
                        'fraud_count': balance_depletion['isFraud'].sum(),
                        'fraud_rate': balance_depletion['isFraud'].mean()
                    }
        
        # 4. Account type risk (merchant vs customer)
        if 'nameOrig' in df.columns and 'nameDest' in df.columns:
            df_temp = df.copy()
            df_temp['is_merchant_orig'] = df_temp['nameOrig'].str.startswith('M')
            df_temp['is_merchant_dest'] = df_temp['nameDest'].str.startswith('M')
            
            # Customer to merchant vs customer to customer
            account_patterns = []
            patterns = [
                ('customer_to_customer', ~df_temp['is_merchant_orig'] & ~df_temp['is_merchant_dest']),
                ('customer_to_merchant', ~df_temp['is_merchant_orig'] & df_temp['is_merchant_dest']),
                ('merchant_to_customer', df_temp['is_merchant_orig'] & ~df_temp['is_merchant_dest']),
                ('merchant_to_merchant', df_temp['is_merchant_orig'] & df_temp['is_merchant_dest'])
            ]
            
            for pattern_name, pattern_mask in patterns:
                pattern_txns = df_temp[pattern_mask]
                if len(pattern_txns) > 0:
                    account_patterns.append({
                        'pattern': pattern_name,
                        'total_count': len(pattern_txns),
                        'fraud_count': pattern_txns['isFraud'].sum(),
                        'fraud_rate': pattern_txns['isFraud'].mean()
                    })
            
            risk_factors['account_patterns'] = sorted(account_patterns, 
                                                    key=lambda x: x['fraud_rate'], 
                                                    reverse=True)
        
        # 5. Time-based risk factors
        if 'step' in df.columns:
            df_temp = df.copy()
            df_temp['hour'] = df_temp['step'] % 24
            df_temp['day'] = df_temp['step'] // 24
            
            # Hour-based risk
            hour_risk = df_temp.groupby('hour')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            hour_risk.columns = ['hour', 'total_count', 'fraud_count', 'fraud_rate']
            hour_risk = hour_risk.sort_values('fraud_rate', ascending=False)
            risk_factors['hourly_patterns'] = hour_risk.head(top_n).to_dict('records')
        
        # Rank all risk factors by fraud rate
        ranked_factors = self._rank_risk_factors(risk_factors)
        
        return {
            'risk_factors': risk_factors,
            'ranked_factors': ranked_factors[:top_n],
            'summary_stats': self._calculate_risk_summary_stats(risk_factors)
        }
    
    def create_comprehensive_fraud_analysis(self,
                                          df: pd.DataFrame,
                                          interactive: bool = False,
                                          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive fraud pattern analysis report.
        
        Args:
            df: DataFrame containing transaction data
            interactive: Whether to create interactive plotly charts
            save_path: Path to save the report (if None, just display)
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Creating comprehensive fraud pattern analysis")
        
        results = {}
        
        # 1. Time series analysis
        print("1. Analyzing fraud time series patterns...")
        results['time_series_hourly'] = self.plot_fraud_time_series(df, time_unit='hour', 
                                                                   aggregation='count', 
                                                                   interactive=interactive)
        results['time_series_daily'] = self.plot_fraud_time_series(df, time_unit='day', 
                                                                  aggregation='rate', 
                                                                  interactive=interactive)
        
        # 2. Customer behavior analysis
        print("2. Analyzing customer behavior patterns...")
        results['customer_behavior'] = self.analyze_customer_behavior(df, interactive=interactive)
        
        # 3. Risk factor identification
        print("3. Identifying and ranking risk factors...")
        results['risk_factors'] = self.identify_risk_factors(df, top_n=15)
        
        # 4. Create summary dashboard if not interactive
        if not interactive:
            self._create_summary_dashboard(df, results, save_path)
        
        return results   
 
    # Static plotting methods (matplotlib/seaborn)
    def _plot_fraud_time_series_static(self, fraud_stats: pd.Series, xlabel: str, 
                                      ylabel: str, title: str, show_trend: bool) -> Dict[str, Any]:
        """Create static fraud time series plot."""
        plt.figure(figsize=self.figure_size)
        
        # Create line plot
        plt.plot(fraud_stats.index, fraud_stats.values, marker='o', linewidth=2, markersize=6)
        
        # Add trend line if requested
        if show_trend and len(fraud_stats) > 2:
            z = np.polyfit(fraud_stats.index, fraud_stats.values, 1)
            p = np.poly1d(z)
            plt.plot(fraud_stats.index, p(fraud_stats.index), "--", alpha=0.7, color='red', 
                    label=f'Trend (slope: {z[0]:.3f})')
            plt.legend()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'time_data': fraud_stats.index.tolist(),
            'fraud_data': fraud_stats.values.tolist(),
            'xlabel': xlabel,
            'ylabel': ylabel,
            'title': title,
            'trend_slope': np.polyfit(fraud_stats.index, fraud_stats.values, 1)[0] if len(fraud_stats) > 2 else None
        }
    
    def _analyze_customer_behavior_static(self, df: pd.DataFrame, 
                                        balance_change_stats: Dict[str, float]) -> Dict[str, Any]:
        """Create static customer behavior analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Balance change distribution by fraud status
        ax1 = axes[0, 0]
        fraud_data = df[df['isFraud'] == 1]['balance_change_orig']
        legit_data = df[df['isFraud'] == 0]['balance_change_orig']
        
        ax1.hist([legit_data, fraud_data], bins=50, alpha=0.7, 
                label=['Legitimate', 'Fraud'], color=['lightblue', 'red'])
        ax1.set_xlabel('Origin Balance Change')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Origin Balance Change Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Merchant vs Customer patterns
        ax2 = axes[0, 1]
        merchant_patterns = df.groupby(['is_merchant_dest', 'isFraud']).size().unstack(fill_value=0)
        merchant_patterns.plot(kind='bar', ax=ax2, color=['lightblue', 'red'])
        ax2.set_xlabel('Is Destination Merchant')
        ax2.set_ylabel('Transaction Count')
        ax2.set_title('Fraud Patterns: Customer vs Merchant Destinations')
        ax2.legend(['Legitimate', 'Fraud'])
        ax2.tick_params(axis='x', rotation=0)
        
        # 3. Unusual balance changes
        ax3 = axes[1, 0]
        unusual_balance = df.groupby(['unusual_balance_orig', 'isFraud']).size().unstack(fill_value=0)
        unusual_balance.plot(kind='bar', ax=ax3, color=['lightblue', 'red'])
        ax3.set_xlabel('Unusual Origin Balance Change')
        ax3.set_ylabel('Transaction Count')
        ax3.set_title('Fraud Patterns: Unusual Balance Changes')
        ax3.legend(['Legitimate', 'Fraud'])
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Amount vs Balance Change scatter
        ax4 = axes[1, 1]
        fraud_mask = df['isFraud'] == 1
        ax4.scatter(df[~fraud_mask]['amount'], df[~fraud_mask]['balance_change_orig'], 
                   alpha=0.5, s=10, color='lightblue', label='Legitimate')
        ax4.scatter(df[fraud_mask]['amount'], df[fraud_mask]['balance_change_orig'], 
                   alpha=0.7, s=15, color='red', label='Fraud')
        ax4.set_xlabel('Transaction Amount')
        ax4.set_ylabel('Origin Balance Change')
        ax4.set_title('Amount vs Balance Change Pattern')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Customer Behavior Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()
        
        # Calculate behavior statistics
        behavior_stats = {
            'unusual_balance_fraud_rate': df[df['unusual_balance_orig']]['isFraud'].mean(),
            'merchant_dest_fraud_rate': df[df['is_merchant_dest']]['isFraud'].mean(),
            'customer_dest_fraud_rate': df[~df['is_merchant_dest']]['isFraud'].mean(),
            'balance_change_stats': balance_change_stats
        }
        
        return behavior_stats
    
    # Interactive plotting methods (plotly)
    def _plot_fraud_time_series_interactive(self, fraud_stats: pd.Series, xlabel: str, 
                                           ylabel: str, title: str, show_trend: bool) -> Dict[str, Any]:
        """Create interactive fraud time series plot."""
        fig = go.Figure()
        
        # Add main line plot
        fig.add_trace(go.Scatter(
            x=fraud_stats.index,
            y=fraud_stats.values,
            mode='lines+markers',
            name='Fraud Data',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # Add trend line if requested
        if show_trend and len(fraud_stats) > 2:
            z = np.polyfit(fraud_stats.index, fraud_stats.values, 1)
            p = np.poly1d(z)
            trend_y = p(fraud_stats.index)
            
            fig.add_trace(go.Scatter(
                x=fraud_stats.index,
                y=trend_y,
                mode='lines',
                name=f'Trend (slope: {z[0]:.3f})',
                line=dict(dash='dash', color='red', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=900, height=600,
            hovermode='x unified'
        )
        
        fig.show()
        
        return {
            'time_data': fraud_stats.index.tolist(),
            'fraud_data': fraud_stats.values.tolist(),
            'xlabel': xlabel,
            'ylabel': ylabel,
            'title': title,
            'trend_slope': np.polyfit(fraud_stats.index, fraud_stats.values, 1)[0] if len(fraud_stats) > 2 else None
        }
    
    def _analyze_customer_behavior_interactive(self, df: pd.DataFrame, 
                                             balance_change_stats: Dict[str, float]) -> Dict[str, Any]:
        """Create interactive customer behavior analysis plots."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Origin Balance Change Distribution',
                'Fraud Patterns: Customer vs Merchant',
                'Unusual Balance Changes',
                'Amount vs Balance Change Pattern'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Balance change distribution
        fraud_data = df[df['isFraud'] == 1]['balance_change_orig']
        legit_data = df[df['isFraud'] == 0]['balance_change_orig']
        
        fig.add_trace(go.Histogram(x=legit_data, name='Legitimate', 
                                  marker_color='lightblue', opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=fraud_data, name='Fraud', 
                                  marker_color='red', opacity=0.7), row=1, col=1)
        
        # 2. Merchant patterns
        merchant_stats = df.groupby(['is_merchant_dest', 'isFraud']).size().reset_index(name='count')
        merchant_stats['dest_type'] = merchant_stats['is_merchant_dest'].map({True: 'Merchant', False: 'Customer'})
        merchant_stats['fraud_status'] = merchant_stats['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
        
        for fraud_status in ['Legitimate', 'Fraud']:
            data = merchant_stats[merchant_stats['fraud_status'] == fraud_status]
            color = 'lightblue' if fraud_status == 'Legitimate' else 'red'
            fig.add_trace(go.Bar(x=data['dest_type'], y=data['count'], 
                               name=fraud_status, marker_color=color, 
                               showlegend=False), row=1, col=2)
        
        # 3. Unusual balance changes
        unusual_stats = df.groupby(['unusual_balance_orig', 'isFraud']).size().reset_index(name='count')
        unusual_stats['balance_type'] = unusual_stats['unusual_balance_orig'].map({True: 'Unusual', False: 'Normal'})
        unusual_stats['fraud_status'] = unusual_stats['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
        
        for fraud_status in ['Legitimate', 'Fraud']:
            data = unusual_stats[unusual_stats['fraud_status'] == fraud_status]
            color = 'lightblue' if fraud_status == 'Legitimate' else 'red'
            fig.add_trace(go.Bar(x=data['balance_type'], y=data['count'], 
                               name=fraud_status, marker_color=color, 
                               showlegend=False), row=2, col=1)
        
        # 4. Scatter plot
        fraud_mask = df['isFraud'] == 1
        fig.add_trace(go.Scatter(x=df[~fraud_mask]['amount'], 
                               y=df[~fraud_mask]['balance_change_orig'],
                               mode='markers', name='Legitimate', 
                               marker=dict(color='lightblue', size=4, opacity=0.6),
                               showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=df[fraud_mask]['amount'], 
                               y=df[fraud_mask]['balance_change_orig'],
                               mode='markers', name='Fraud', 
                               marker=dict(color='red', size=6, opacity=0.8),
                               showlegend=False), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Customer Behavior Analysis", 
                         barmode='group')
        fig.show()
        
        # Calculate behavior statistics
        behavior_stats = {
            'unusual_balance_fraud_rate': df[df['unusual_balance_orig']]['isFraud'].mean(),
            'merchant_dest_fraud_rate': df[df['is_merchant_dest']]['isFraud'].mean(),
            'customer_dest_fraud_rate': df[~df['is_merchant_dest']]['isFraud'].mean(),
            'balance_change_stats': balance_change_stats
        }
        
        return behavior_stats
    
    # Helper methods
    def _rank_risk_factors(self, risk_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank all risk factors by fraud rate."""
        ranked_factors = []
        
        # Transaction type risks
        if 'transaction_type' in risk_factors:
            for item in risk_factors['transaction_type']:
                ranked_factors.append({
                    'factor_type': 'transaction_type',
                    'factor_name': item['type'],
                    'fraud_rate': item['fraud_rate'],
                    'fraud_count': item['fraud_count'],
                    'total_count': item['total_count']
                })
        
        # Large transfer risk
        if 'large_transfers' in risk_factors:
            item = risk_factors['large_transfers']
            ranked_factors.append({
                'factor_type': 'large_transfers',
                'factor_name': f"Amount > {item['threshold']}",
                'fraud_rate': item['fraud_rate'],
                'fraud_count': item['fraud_count'],
                'total_count': item['total_count']
            })
        
        # Amount percentile risks
        if 'amount_percentiles' in risk_factors:
            for item in risk_factors['amount_percentiles']:
                ranked_factors.append({
                    'factor_type': 'amount_percentile',
                    'factor_name': f"Amount >= {item['percentile']}th percentile",
                    'fraud_rate': item['fraud_rate'],
                    'fraud_count': item['fraud_count'],
                    'total_count': item['total_count']
                })
        
        # Zero balance risks
        if 'zero_balance' in risk_factors:
            for item in risk_factors['zero_balance']:
                ranked_factors.append({
                    'factor_type': 'zero_balance',
                    'factor_name': f"Zero {item['balance_type']}",
                    'fraud_rate': item['fraud_rate'],
                    'fraud_count': item['fraud_count'],
                    'total_count': item['total_count']
                })
        
        # Balance depletion risk
        if 'balance_depletion' in risk_factors:
            item = risk_factors['balance_depletion']
            ranked_factors.append({
                'factor_type': 'balance_depletion',
                'factor_name': 'Complete balance depletion',
                'fraud_rate': item['fraud_rate'],
                'fraud_count': item['fraud_count'],
                'total_count': item['total_count']
            })
        
        # Account pattern risks
        if 'account_patterns' in risk_factors:
            for item in risk_factors['account_patterns']:
                ranked_factors.append({
                    'factor_type': 'account_pattern',
                    'factor_name': item['pattern'].replace('_', ' ').title(),
                    'fraud_rate': item['fraud_rate'],
                    'fraud_count': item['fraud_count'],
                    'total_count': item['total_count']
                })
        
        # Hourly pattern risks (top 5 only)
        if 'hourly_patterns' in risk_factors:
            for item in risk_factors['hourly_patterns'][:5]:
                ranked_factors.append({
                    'factor_type': 'hourly_pattern',
                    'factor_name': f"Hour {item['hour']}",
                    'fraud_rate': item['fraud_rate'],
                    'fraud_count': item['fraud_count'],
                    'total_count': item['total_count']
                })
        
        # Sort by fraud rate
        return sorted(ranked_factors, key=lambda x: x['fraud_rate'], reverse=True)
    
    def _calculate_risk_summary_stats(self, risk_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for risk factors."""
        summary = {
            'total_risk_factors': 0,
            'highest_risk_factor': None,
            'average_fraud_rate': 0,
            'risk_factor_categories': []
        }
        
        all_rates = []
        highest_rate = 0
        highest_factor = None
        
        for category, data in risk_factors.items():
            if isinstance(data, list):
                category_rates = [item.get('fraud_rate', 0) for item in data if 'fraud_rate' in item]
                if category_rates:
                    all_rates.extend(category_rates)
                    max_rate = max(category_rates)
                    if max_rate > highest_rate:
                        highest_rate = max_rate
                        highest_factor = category
                    
                    summary['risk_factor_categories'].append({
                        'category': category,
                        'count': len(category_rates),
                        'max_fraud_rate': max_rate,
                        'avg_fraud_rate': np.mean(category_rates)
                    })
            elif isinstance(data, dict) and 'fraud_rate' in data:
                rate = data['fraud_rate']
                all_rates.append(rate)
                if rate > highest_rate:
                    highest_rate = rate
                    highest_factor = category
                
                summary['risk_factor_categories'].append({
                    'category': category,
                    'count': 1,
                    'max_fraud_rate': rate,
                    'avg_fraud_rate': rate
                })
        
        summary['total_risk_factors'] = len(all_rates)
        summary['highest_risk_factor'] = highest_factor
        summary['average_fraud_rate'] = np.mean(all_rates) if all_rates else 0
        
        return summary
    
    def _create_summary_dashboard(self, df: pd.DataFrame, results: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> None:
        """Create a summary dashboard with all fraud pattern analysis."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Fraud time series (hourly)
        plt.subplot(3, 3, 1)
        if 'time_series_hourly' in results and 'fraud_data' in results['time_series_hourly']:
            hourly_data = results['time_series_hourly']
            plt.plot(hourly_data['time_data'], hourly_data['fraud_data'], 'o-')
            plt.title('Fraud Count by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Fraud Count')
            plt.grid(True, alpha=0.3)
        
        # 2. Fraud time series (daily rate)
        plt.subplot(3, 3, 2)
        if 'time_series_daily' in results and 'fraud_data' in results['time_series_daily']:
            daily_data = results['time_series_daily']
            plt.plot(daily_data['time_data'], daily_data['fraud_data'], 'o-', color='red')
            plt.title('Fraud Rate by Day')
            plt.xlabel('Day')
            plt.ylabel('Fraud Rate (%)')
            plt.grid(True, alpha=0.3)
        
        # 3. Top risk factors
        plt.subplot(3, 3, 3)
        if 'risk_factors' in results and 'ranked_factors' in results['risk_factors']:
            top_factors = results['risk_factors']['ranked_factors'][:10]
            if top_factors:
                factor_names = [f['factor_name'][:20] + '...' if len(f['factor_name']) > 20 
                               else f['factor_name'] for f in top_factors]
                fraud_rates = [f['fraud_rate'] * 100 for f in top_factors]
                
                plt.barh(range(len(factor_names)), fraud_rates)
                plt.yticks(range(len(factor_names)), factor_names)
                plt.xlabel('Fraud Rate (%)')
                plt.title('Top Risk Factors')
                plt.gca().invert_yaxis()
        
        # 4. Transaction type fraud rates
        plt.subplot(3, 3, 4)
        if 'type' in df.columns:
            type_fraud_rates = df.groupby('type')['isFraud'].mean() * 100
            type_fraud_rates.plot(kind='bar', color='lightcoral')
            plt.title('Fraud Rate by Transaction Type')
            plt.xlabel('Transaction Type')
            plt.ylabel('Fraud Rate (%)')
            plt.xticks(rotation=45)
        
        # 5. Amount distribution (fraud vs legitimate)
        plt.subplot(3, 3, 5)
        if 'amount' in df.columns:
            fraud_amounts = df[df['isFraud'] == 1]['amount']
            legit_amounts = df[df['isFraud'] == 0]['amount']
            plt.hist([legit_amounts, fraud_amounts], bins=30, alpha=0.7, 
                    label=['Legitimate', 'Fraud'], color=['lightblue', 'red'])
            plt.xlabel('Transaction Amount (log scale)')
            plt.ylabel('Frequency')
            plt.title('Amount Distribution')
            plt.xscale('log')
            plt.legend()
        
        # 6. Balance change patterns
        plt.subplot(3, 3, 6)
        if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig']):
            df_temp = df.copy()
            df_temp['balance_change'] = df_temp['newbalanceOrig'] - df_temp['oldbalanceOrg']
            fraud_balance_change = df_temp[df_temp['isFraud'] == 1]['balance_change']
            legit_balance_change = df_temp[df_temp['isFraud'] == 0]['balance_change']
            plt.hist([legit_balance_change, fraud_balance_change], bins=30, alpha=0.7,
                    label=['Legitimate', 'Fraud'], color=['lightblue', 'red'])
            plt.xlabel('Balance Change')
            plt.ylabel('Frequency')
            plt.title('Balance Change Distribution')
            plt.legend()
        
        # 7. Merchant vs Customer patterns
        plt.subplot(3, 3, 7)
        if 'nameDest' in df.columns:
            df_temp = df.copy()
            df_temp['is_merchant_dest'] = df_temp['nameDest'].str.startswith('M')
            merchant_fraud = df_temp.groupby(['is_merchant_dest', 'isFraud']).size().unstack(fill_value=0)
            merchant_fraud.plot(kind='bar', color=['lightblue', 'red'])
            plt.title('Fraud: Customer vs Merchant Destinations')
            plt.xlabel('Is Merchant Destination')
            plt.ylabel('Count')
            plt.legend(['Legitimate', 'Fraud'])
            plt.xticks(rotation=0)
        
        # 8. Large transfer analysis
        plt.subplot(3, 3, 8)
        if 'amount' in df.columns:
            large_transfers = df[df['amount'] > 200000]
            if len(large_transfers) > 0:
                large_transfer_fraud_rate = large_transfers['isFraud'].mean() * 100
                normal_transfer_fraud_rate = df[df['amount'] <= 200000]['isFraud'].mean() * 100
                
                plt.bar(['Normal Transfers\n(≤200K)', 'Large Transfers\n(>200K)'], 
                       [normal_transfer_fraud_rate, large_transfer_fraud_rate],
                       color=['lightblue', 'red'])
                plt.ylabel('Fraud Rate (%)')
                plt.title('Large Transfer Risk')
        
        # 9. Risk factor summary
        plt.subplot(3, 3, 9)
        if 'risk_factors' in results and 'summary_stats' in results['risk_factors']:
            summary_stats = results['risk_factors']['summary_stats']
            categories = [cat['category'].replace('_', ' ').title() 
                         for cat in summary_stats['risk_factor_categories']]
            avg_rates = [cat['avg_fraud_rate'] * 100 
                        for cat in summary_stats['risk_factor_categories']]
            
            if categories and avg_rates:
                plt.bar(categories, avg_rates, color='orange', alpha=0.7)
                plt.ylabel('Average Fraud Rate (%)')
                plt.title('Risk Factor Categories')
                plt.xticks(rotation=45)
        
        plt.suptitle('Comprehensive Fraud Pattern Analysis Dashboard', fontsize=20, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fraud pattern analysis dashboard saved to {save_path}")
        
        plt.show()