"""
DataVisualizer module for exploratory data analysis and fraud pattern visualization.

This module provides the DataVisualizer class for creating comprehensive visualizations
of transaction data and fraud patterns to support exploratory data analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    DataVisualizer class for creating exploratory data analysis visualizations.
    
    This class provides comprehensive visualization capabilities for fraud detection
    datasets, including transaction distributions, fraud patterns, and correlation analysis.
    """
    
    def __init__(self, 
                 style: str = 'whitegrid',
                 palette: str = 'Set2',
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 100):
        """
        Initialize DataVisualizer with styling options.
        
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
        
        logger.info(f"DataVisualizer initialized with style: {style}, palette: {palette}")
    
    def plot_transaction_distribution_by_type(self, 
                                            df: pd.DataFrame, 
                                            show_fraud: bool = True,
                                            interactive: bool = False) -> None:
        """
        Create transaction distribution plots by transaction type.
        
        Args:
            df: DataFrame containing transaction data
            show_fraud: Whether to show fraud vs legitimate breakdown
            interactive: Whether to create interactive plotly charts
        """
        logger.info("Creating transaction distribution by type visualization")
        
        if interactive:
            self._plot_transaction_type_interactive(df, show_fraud)
        else:
            self._plot_transaction_type_static(df, show_fraud)
    
    def plot_transaction_distribution_by_amount(self, 
                                              df: pd.DataFrame,
                                              bins: int = 50,
                                              log_scale: bool = True,
                                              show_fraud: bool = True,
                                              interactive: bool = False) -> None:
        """
        Create transaction amount distribution plots.
        
        Args:
            df: DataFrame containing transaction data
            bins: Number of bins for histogram
            log_scale: Whether to use log scale for amount
            show_fraud: Whether to show fraud vs legitimate breakdown
            interactive: Whether to create interactive plotly charts
        """
        logger.info("Creating transaction distribution by amount visualization")
        
        if interactive:
            self._plot_amount_distribution_interactive(df, bins, log_scale, show_fraud)
        else:
            self._plot_amount_distribution_static(df, bins, log_scale, show_fraud)
    
    def plot_transaction_distribution_by_time(self, 
                                            df: pd.DataFrame,
                                            time_unit: str = 'hour',
                                            show_fraud: bool = True,
                                            interactive: bool = False) -> None:
        """
        Create transaction distribution plots by time.
        
        Args:
            df: DataFrame containing transaction data
            time_unit: Time unit for aggregation ('hour', 'day', 'step')
            show_fraud: Whether to show fraud vs legitimate breakdown
            interactive: Whether to create interactive plotly charts
        """
        logger.info(f"Creating transaction distribution by {time_unit} visualization")
        
        if interactive:
            self._plot_time_distribution_interactive(df, time_unit, show_fraud)
        else:
            self._plot_time_distribution_static(df, time_unit, show_fraud)
    
    def plot_fraud_patterns_by_type(self, 
                                   df: pd.DataFrame,
                                   interactive: bool = False) -> None:
        """
        Create fraud pattern visualizations by transaction type.
        
        Args:
            df: DataFrame containing transaction data
            interactive: Whether to create interactive plotly charts
        """
        logger.info("Creating fraud patterns by transaction type visualization")
        
        if interactive:
            self._plot_fraud_patterns_interactive(df)
        else:
            self._plot_fraud_patterns_static(df)
    
    def plot_correlation_heatmap(self, 
                                df: pd.DataFrame,
                                features: Optional[List[str]] = None,
                                method: str = 'pearson',
                                interactive: bool = False) -> None:
        """
        Create correlation heatmap for numerical features.
        
        Args:
            df: DataFrame containing transaction data
            features: List of features to include (if None, use all numerical)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            interactive: Whether to create interactive plotly charts
        """
        logger.info(f"Creating correlation heatmap using {method} correlation")
        
        # Select numerical features
        if features is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target variables from correlation analysis
            features = [col for col in numerical_cols if col not in ['isFraud', 'isFlaggedFraud']]
        
        if len(features) < 2:
            logger.warning("Not enough numerical features for correlation analysis")
            return
        
        if interactive:
            self._plot_correlation_heatmap_interactive(df[features], method)
        else:
            self._plot_correlation_heatmap_static(df[features], method)
    
    def plot_feature_distributions(self, 
                                  df: pd.DataFrame,
                                  features: Optional[List[str]] = None,
                                  show_fraud_comparison: bool = True,
                                  interactive: bool = False) -> None:
        """
        Create feature distribution plots.
        
        Args:
            df: DataFrame containing transaction data
            features: List of features to plot (if None, use key numerical features)
            show_fraud_comparison: Whether to compare fraud vs legitimate distributions
            interactive: Whether to create interactive plotly charts
        """
        logger.info("Creating feature distribution visualizations")
        
        if features is None:
            # Select key features for distribution analysis
            features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            features = [f for f in features if f in df.columns]
        
        if interactive:
            self._plot_feature_distributions_interactive(df, features, show_fraud_comparison)
        else:
            self._plot_feature_distributions_static(df, features, show_fraud_comparison)
    
    def create_comprehensive_eda_report(self, 
                                      df: pd.DataFrame,
                                      save_path: Optional[str] = None,
                                      interactive: bool = False) -> None:
        """
        Create a comprehensive EDA report with all visualizations.
        
        Args:
            df: DataFrame containing transaction data
            save_path: Path to save the report (if None, just display)
            interactive: Whether to create interactive plotly charts
        """
        logger.info("Creating comprehensive EDA report")
        
        if not interactive:
            # Create a multi-panel figure for static plots
            fig = plt.figure(figsize=(20, 24))
            
            # Transaction type distribution
            plt.subplot(4, 2, 1)
            self._plot_transaction_type_static(df, show_fraud=True, subplot=True)
            
            # Amount distribution
            plt.subplot(4, 2, 2)
            self._plot_amount_distribution_static(df, bins=30, log_scale=True, show_fraud=True, subplot=True)
            
            # Time distribution
            plt.subplot(4, 2, 3)
            self._plot_time_distribution_static(df, time_unit='hour', show_fraud=True, subplot=True)
            
            # Fraud patterns
            plt.subplot(4, 2, 4)
            self._plot_fraud_patterns_static(df, subplot=True)
            
            # Feature distributions (amount)
            if 'amount' in df.columns:
                plt.subplot(4, 2, 5)
                self._plot_single_feature_distribution(df, 'amount', show_fraud_comparison=True, subplot=True)
            
            # Balance distributions
            if 'oldbalanceOrg' in df.columns:
                plt.subplot(4, 2, 6)
                self._plot_single_feature_distribution(df, 'oldbalanceOrg', show_fraud_comparison=True, subplot=True)
            
            # Correlation heatmap
            plt.subplot(4, 2, 7)
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [col for col in numerical_cols if col not in ['isFraud', 'isFlaggedFraud']][:6]  # Limit for readability
            if len(features) >= 2:
                self._plot_correlation_heatmap_static(df[features], method='pearson', subplot=True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"EDA report saved to {save_path}")
            
            plt.show()
        else:
            # Create individual interactive plots
            self.plot_transaction_distribution_by_type(df, interactive=True)
            self.plot_transaction_distribution_by_amount(df, interactive=True)
            self.plot_transaction_distribution_by_time(df, interactive=True)
            self.plot_fraud_patterns_by_type(df, interactive=True)
            self.plot_correlation_heatmap(df, interactive=True)
            self.plot_feature_distributions(df, interactive=True)
    
    # Static plotting methods (matplotlib/seaborn)
    def _plot_transaction_type_static(self, df: pd.DataFrame, show_fraud: bool = True, subplot: bool = False) -> None:
        """Create static transaction type distribution plot."""
        if not subplot:
            plt.figure(figsize=self.figure_size)
        
        if show_fraud and 'isFraud' in df.columns:
            # Create stacked bar chart
            fraud_by_type = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
            fraud_by_type.plot(kind='bar', stacked=True, 
                             color=['lightblue', 'red'], 
                             alpha=0.7)
            plt.title('Transaction Distribution by Type (Fraud vs Legitimate)')
            plt.legend(['Legitimate', 'Fraud'])
        else:
            # Simple count plot
            sns.countplot(data=df, x='type')
            plt.title('Transaction Distribution by Type')
        
        plt.xlabel('Transaction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if not subplot:
            plt.tight_layout()
            plt.show()
    
    def _plot_amount_distribution_static(self, df: pd.DataFrame, bins: int = 50, 
                                       log_scale: bool = True, show_fraud: bool = True, 
                                       subplot: bool = False) -> None:
        """Create static amount distribution plot."""
        if not subplot:
            plt.figure(figsize=self.figure_size)
        
        if show_fraud and 'isFraud' in df.columns:
            # Separate fraud and legitimate transactions
            fraud_amounts = df[df['isFraud'] == 1]['amount']
            legit_amounts = df[df['isFraud'] == 0]['amount']
            
            plt.hist([legit_amounts, fraud_amounts], bins=bins, alpha=0.7, 
                    label=['Legitimate', 'Fraud'], color=['lightblue', 'red'])
            plt.legend()
        else:
            plt.hist(df['amount'], bins=bins, alpha=0.7, color='lightblue')
        
        if log_scale:
            plt.xscale('log')
            plt.xlabel('Transaction Amount (log scale)')
        else:
            plt.xlabel('Transaction Amount')
        
        plt.ylabel('Frequency')
        plt.title('Transaction Amount Distribution')
        
        if not subplot:
            plt.tight_layout()
            plt.show()
    
    def _plot_time_distribution_static(self, df: pd.DataFrame, time_unit: str = 'hour', 
                                     show_fraud: bool = True, subplot: bool = False) -> None:
        """Create static time distribution plot."""
        if not subplot:
            plt.figure(figsize=self.figure_size)
        
        # Create time feature based on step
        if time_unit == 'hour':
            df_temp = df.copy()
            df_temp['time_feature'] = df_temp['step'] % 24
            xlabel = 'Hour of Day'
            title = 'Transaction Distribution by Hour of Day'
        elif time_unit == 'day':
            df_temp = df.copy()
            df_temp['time_feature'] = df_temp['step'] // 24
            xlabel = 'Day'
            title = 'Transaction Distribution by Day'
        else:  # step
            df_temp = df.copy()
            df_temp['time_feature'] = df_temp['step']
            xlabel = 'Time Step'
            title = 'Transaction Distribution by Time Step'
        
        if show_fraud and 'isFraud' in df.columns:
            # Create grouped bar chart
            fraud_by_time = df_temp.groupby(['time_feature', 'isFraud']).size().unstack(fill_value=0)
            fraud_by_time.plot(kind='bar', color=['lightblue', 'red'], alpha=0.7, width=0.8)
            plt.legend(['Legitimate', 'Fraud'])
        else:
            time_counts = df_temp['time_feature'].value_counts().sort_index()
            time_counts.plot(kind='bar', color='lightblue', alpha=0.7)
        
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(rotation=45)
        
        if not subplot:
            plt.tight_layout()
            plt.show()
    
    def _plot_fraud_patterns_static(self, df: pd.DataFrame, subplot: bool = False) -> None:
        """Create static fraud patterns plot."""
        if 'isFraud' not in df.columns:
            logger.warning("No fraud labels available for fraud pattern analysis")
            return
        
        if not subplot:
            plt.figure(figsize=self.figure_size)
        
        # Calculate fraud rates by transaction type
        fraud_rates = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_rates.columns = ['type', 'total_transactions', 'fraud_count', 'fraud_rate']
        
        # Create bar plot of fraud rates
        bars = plt.bar(fraud_rates['type'], fraud_rates['fraud_rate'] * 100, 
                      color='red', alpha=0.7)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, fraud_rates['fraud_count'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Transaction Type')
        plt.ylabel('Fraud Rate (%)')
        plt.title('Fraud Rate by Transaction Type')
        plt.xticks(rotation=45)
        
        if not subplot:
            plt.tight_layout()
            plt.show()
    
    def _plot_correlation_heatmap_static(self, df: pd.DataFrame, method: str = 'pearson', 
                                       subplot: bool = False) -> None:
        """Create static correlation heatmap."""
        if not subplot:
            plt.figure(figsize=self.figure_size)
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method=method)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        plt.title(f'Feature Correlation Heatmap ({method.capitalize()})')
        
        if not subplot:
            plt.tight_layout()
            plt.show()
    
    def _plot_feature_distributions_static(self, df: pd.DataFrame, features: List[str], 
                                         show_fraud_comparison: bool = True) -> None:
        """Create static feature distribution plots."""
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            self._plot_single_feature_distribution(df, feature, show_fraud_comparison, ax=ax)
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_single_feature_distribution(self, df: pd.DataFrame, feature: str, 
                                        show_fraud_comparison: bool = True, 
                                        subplot: bool = False, ax=None) -> None:
        """Plot distribution of a single feature."""
        if ax is None and not subplot:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
        elif ax is None:
            ax = plt.gca()
        
        if show_fraud_comparison and 'isFraud' in df.columns:
            # Plot distributions for fraud vs legitimate
            fraud_data = df[df['isFraud'] == 1][feature].dropna()
            legit_data = df[df['isFraud'] == 0][feature].dropna()
            
            ax.hist([legit_data, fraud_data], bins=30, alpha=0.7, 
                   label=['Legitimate', 'Fraud'], color=['lightblue', 'red'])
            ax.legend()
        else:
            ax.hist(df[feature].dropna(), bins=30, alpha=0.7, color='lightblue')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        
        if not subplot and ax == plt.gca():
            plt.tight_layout()
            plt.show()    
   
 # Interactive plotting methods (plotly)
    def _plot_transaction_type_interactive(self, df: pd.DataFrame, show_fraud: bool = True) -> None:
        """Create interactive transaction type distribution plot."""
        if show_fraud and 'isFraud' in df.columns:
            # Create grouped bar chart
            fraud_by_type = df.groupby(['type', 'isFraud']).size().reset_index(name='count')
            fraud_by_type['fraud_status'] = fraud_by_type['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            
            fig = px.bar(fraud_by_type, x='type', y='count', color='fraud_status',
                        title='Transaction Distribution by Type (Fraud vs Legitimate)',
                        labels={'type': 'Transaction Type', 'count': 'Count'},
                        color_discrete_map={'Legitimate': 'lightblue', 'Fraud': 'red'})
        else:
            type_counts = df['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            
            fig = px.bar(type_counts, x='type', y='count',
                        title='Transaction Distribution by Type',
                        labels={'type': 'Transaction Type', 'count': 'Count'})
        
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    
    def _plot_amount_distribution_interactive(self, df: pd.DataFrame, bins: int = 50, 
                                            log_scale: bool = True, show_fraud: bool = True) -> None:
        """Create interactive amount distribution plot."""
        if show_fraud and 'isFraud' in df.columns:
            df_temp = df.copy()
            df_temp['fraud_status'] = df_temp['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            
            fig = px.histogram(df_temp, x='amount', color='fraud_status', nbins=bins,
                             title='Transaction Amount Distribution',
                             labels={'amount': 'Transaction Amount', 'count': 'Frequency'},
                             color_discrete_map={'Legitimate': 'lightblue', 'Fraud': 'red'})
        else:
            fig = px.histogram(df, x='amount', nbins=bins,
                             title='Transaction Amount Distribution',
                             labels={'amount': 'Transaction Amount', 'count': 'Frequency'})
        
        if log_scale:
            fig.update_xaxes(type="log", title="Transaction Amount (log scale)")
        
        fig.show()
    
    def _plot_time_distribution_interactive(self, df: pd.DataFrame, time_unit: str = 'hour', 
                                          show_fraud: bool = True) -> None:
        """Create interactive time distribution plot."""
        df_temp = df.copy()
        
        # Create time feature based on step
        if time_unit == 'hour':
            df_temp['time_feature'] = df_temp['step'] % 24
            xlabel = 'Hour of Day'
            title = 'Transaction Distribution by Hour of Day'
        elif time_unit == 'day':
            df_temp['time_feature'] = df_temp['step'] // 24
            xlabel = 'Day'
            title = 'Transaction Distribution by Day'
        else:  # step
            df_temp['time_feature'] = df_temp['step']
            xlabel = 'Time Step'
            title = 'Transaction Distribution by Time Step'
        
        if show_fraud and 'isFraud' in df.columns:
            df_temp['fraud_status'] = df_temp['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            
            fig = px.histogram(df_temp, x='time_feature', color='fraud_status',
                             title=title,
                             labels={'time_feature': xlabel, 'count': 'Count'},
                             color_discrete_map={'Legitimate': 'lightblue', 'Fraud': 'red'})
        else:
            fig = px.histogram(df_temp, x='time_feature',
                             title=title,
                             labels={'time_feature': xlabel, 'count': 'Count'})
        
        fig.show()
    
    def _plot_fraud_patterns_interactive(self, df: pd.DataFrame) -> None:
        """Create interactive fraud patterns plot."""
        if 'isFraud' not in df.columns:
            logger.warning("No fraud labels available for fraud pattern analysis")
            return
        
        # Calculate fraud rates by transaction type
        fraud_rates = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
        fraud_rates.columns = ['type', 'total_transactions', 'fraud_count', 'fraud_rate']
        fraud_rates['fraud_rate_pct'] = fraud_rates['fraud_rate'] * 100
        
        # Create bar plot with hover information
        fig = px.bar(fraud_rates, x='type', y='fraud_rate_pct',
                    title='Fraud Rate by Transaction Type',
                    labels={'type': 'Transaction Type', 'fraud_rate_pct': 'Fraud Rate (%)'},
                    hover_data={'total_transactions': True, 'fraud_count': True})
        
        fig.update_traces(marker_color='red', opacity=0.7)
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    
    def _plot_correlation_heatmap_interactive(self, df: pd.DataFrame, method: str = 'pearson') -> None:
        """Create interactive correlation heatmap."""
        # Calculate correlation matrix
        corr_matrix = df.corr(method=method)
        
        # Create interactive heatmap
        fig = px.imshow(corr_matrix, 
                       title=f'Feature Correlation Heatmap ({method.capitalize()})',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        
        # Add correlation values as text
        fig.update_traces(text=np.around(corr_matrix.values, decimals=2), texttemplate="%{text}")
        fig.update_layout(width=800, height=600)
        fig.show()
    
    def _plot_feature_distributions_interactive(self, df: pd.DataFrame, features: List[str], 
                                              show_fraud_comparison: bool = True) -> None:
        """Create interactive feature distribution plots."""
        n_features = len(features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        subplot_titles = [f'Distribution of {feature}' for feature in features]
        fig = make_subplots(rows=n_rows, cols=n_cols, 
                           subplot_titles=subplot_titles)
        
        for i, feature in enumerate(features):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            if show_fraud_comparison and 'isFraud' in df.columns:
                # Add fraud and legitimate distributions
                fraud_data = df[df['isFraud'] == 1][feature].dropna()
                legit_data = df[df['isFraud'] == 0][feature].dropna()
                
                fig.add_trace(
                    go.Histogram(x=legit_data, name='Legitimate', 
                               marker_color='lightblue', opacity=0.7,
                               showlegend=(i == 0)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Histogram(x=fraud_data, name='Fraud', 
                               marker_color='red', opacity=0.7,
                               showlegend=(i == 0)),
                    row=row, col=col
                )
            else:
                fig.add_trace(
                    go.Histogram(x=df[feature].dropna(), 
                               marker_color='lightblue', opacity=0.7,
                               showlegend=False),
                    row=row, col=col
                )
        
        fig.update_layout(height=400 * n_rows, 
                         title_text="Feature Distributions",
                         barmode='overlay')
        fig.show()
    
    def get_data_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics for the dataset.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Dict containing summary statistics
        """
        logger.info("Calculating data summary statistics")
        
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'min_step': df['step'].min() if 'step' in df.columns else None,
                'max_step': df['step'].max() if 'step' in df.columns else None,
                'total_steps': df['step'].nunique() if 'step' in df.columns else None
            },
            'transaction_types': {},
            'fraud_statistics': {},
            'amount_statistics': {},
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Transaction type distribution
        if 'type' in df.columns:
            summary['transaction_types'] = df['type'].value_counts().to_dict()
        
        # Fraud statistics
        if 'isFraud' in df.columns:
            fraud_count = df['isFraud'].sum()
            fraud_rate = fraud_count / len(df) * 100
            summary['fraud_statistics'] = {
                'total_fraud': int(fraud_count),
                'fraud_rate_percent': round(fraud_rate, 4),
                'fraud_by_type': df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).to_dict() if 'type' in df.columns else {}
            }
        
        # Amount statistics
        if 'amount' in df.columns:
            summary['amount_statistics'] = {
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max(),
                'q25': df['amount'].quantile(0.25),
                'q75': df['amount'].quantile(0.75)
            }
        
        return summary
    
    def print_data_summary(self, df: pd.DataFrame) -> None:
        """
        Print a formatted summary of the dataset.
        
        Args:
            df: DataFrame containing transaction data
        """
        summary = self.get_data_summary_stats(df)
        
        print("=" * 60)
        print("FRAUD DETECTION DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\nDataset Overview:")
        print(f"  Total Transactions: {summary['total_transactions']:,}")
        
        if summary['date_range']['min_step'] is not None:
            print(f"  Time Range: Step {summary['date_range']['min_step']} to {summary['date_range']['max_step']}")
            print(f"  Total Time Steps: {summary['date_range']['total_steps']}")
        
        print(f"\nTransaction Types:")
        for trans_type, count in summary['transaction_types'].items():
            percentage = (count / summary['total_transactions']) * 100
            print(f"  {trans_type}: {count:,} ({percentage:.2f}%)")
        
        if summary['fraud_statistics']:
            print(f"\nFraud Statistics:")
            print(f"  Total Fraudulent: {summary['fraud_statistics']['total_fraud']:,}")
            print(f"  Fraud Rate: {summary['fraud_statistics']['fraud_rate_percent']:.4f}%")
            
            if summary['fraud_statistics']['fraud_by_type']:
                print(f"\n  Fraud Rate by Transaction Type:")
                fraud_by_type = summary['fraud_statistics']['fraud_by_type']
                for trans_type in summary['transaction_types'].keys():
                    if trans_type in fraud_by_type['mean']:
                        fraud_rate = fraud_by_type['mean'][trans_type] * 100
                        fraud_count = fraud_by_type['sum'][trans_type]
                        print(f"    {trans_type}: {fraud_rate:.4f}% ({fraud_count:,} fraudulent)")
        
        if summary['amount_statistics']:
            print(f"\nAmount Statistics:")
            stats = summary['amount_statistics']
            print(f"  Mean: ${stats['mean']:,.2f}")
            print(f"  Median: ${stats['median']:,.2f}")
            print(f"  Std Dev: ${stats['std']:,.2f}")
            print(f"  Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}")
            print(f"  IQR: ${stats['q25']:,.2f} - ${stats['q75']:,.2f}")
        
        # Check for missing values
        missing_values = {k: v for k, v in summary['missing_values'].items() if v > 0}
        if missing_values:
            print(f"\nMissing Values:")
            for col, count in missing_values.items():
                percentage = (count / summary['total_transactions']) * 100
                print(f"  {col}: {count:,} ({percentage:.2f}%)")
        else:
            print(f"\nNo missing values found.")
        
        print("=" * 60)